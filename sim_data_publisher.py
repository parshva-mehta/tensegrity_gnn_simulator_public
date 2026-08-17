"""Stream EKF rod state estimates to ROS Noetic via rosbridge + roslibpy.

See ``docs/ekf_ros_integration_design.md`` for the architecture. In short: this
process holds no ROS dependency at all -- it speaks JSON-over-websocket to a
``rosbridge_websocket`` server running inside a ROS Noetic container, which
republishes onto real ROS topics. One ``nav_msgs/Odometry`` topic per rod,
named ``/tensegrity/<rod_name>/odom``.

The EKF state is 13 values per rod: pos (3), quat (4), linvel (3), angvel (3).
Two conventions differ between this repo and ROS and are handled here:

* **Quaternion order.** This repo stores ``(w, x, y, z)`` (see
  ``utilities/torch_quaternion.py``); ``geometry_msgs/Quaternion`` is
  ``(x, y, z, w)``.
* **Twist frame.** The repo integrates ``pos += linear_vel * dt`` and
  ``quat = quat_exp(0.5*dt*ang_vel) (x) quat`` -- left-multiplication, i.e. both
  velocities are **world-frame**. ``nav_msgs/Odometry.twist`` is conventionally
  expressed in ``child_frame_id`` (body) axes, so by default the twist is
  rotated into the body frame before publishing. Pass ``twist_frame="world"`` to
  publish the raw world-frame velocities instead (a documented deviation from
  the ROS convention).

Typical use::

    from sim_data_publisher import RodStatePublisher, rod_names_from_simulator

    with RodStatePublisher(rod_names=rod_names_from_simulator(simulator)) as pub:
        frames = run_ekf_rollout(simulator, gt_data, extra_gt_data, dt,
                                 publisher=pub)

``roslibpy`` is imported lazily, so this module (and every pure helper in it)
can be imported and unit-tested without ROS or roslibpy installed.
"""

import math
import os
import time as _time

# 13 = pos(3) + quat(4) + linvel(3) + angvel(3)
STATE_DIM_PER_ROD = 13

DEFAULT_ROSBRIDGE_URL = "ws://localhost:9090"
ODOMETRY_MSG_TYPE = "nav_msgs/Odometry"
DEFAULT_TOPIC_NAMESPACE = "/tensegrity"
DEFAULT_FRAME_ID = "world"

# nav_msgs/Odometry carries 6x6 pose and twist covariances. The EKF's covariance
# is 13-dim per rod (quaternion included) and has no exact closed-form
# projection onto the 3-position + 3-small-angle ordering ROS expects, so v1
# leaves them zeroed. See the design doc's "Covariance mismatch" note.
_ZERO_COVARIANCE = [0.0] * 36


def _as_floats(value, n, name):
    """Coerce a torch tensor / numpy array / sequence into a list of n floats."""
    # Avoid importing torch or numpy: both expose tolist() / flatten-able shapes.
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):  # tensor / ndarray
        value = value.reshape(-1).tolist()
    else:
        value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} must have {n} elements, got {len(value)}")
    return [float(v) for v in value]


def quat_wxyz_to_ros(quat):
    """Reorder a repo ``(w, x, y, z)`` quaternion into a ROS Quaternion dict."""
    w, x, y, z = _as_floats(quat, 4, "quat")
    return {"x": x, "y": y, "z": z, "w": w}


def _vec3(vec):
    x, y, z = vec
    return {"x": x, "y": y, "z": z}


def rotate_world_to_body(quat, vec):
    """Rotate a world-frame vector into the body frame of ``quat``.

    Equivalent to ``R(q).T @ vec``, i.e. ``q* (x) (0, vec) (x) q`` with the
    repo's ``(w, x, y, z)`` layout. Mirrors
    ``torch_quaternion.rotate_vec_quat(inverse_unit_quat(q), vec)``; the unit
    test asserts agreement with that reference implementation.
    """
    w, x, y, z = _as_floats(quat, 4, "quat")
    vx, vy, vz = _as_floats(vec, 3, "vec")

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("cannot rotate by a zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Rows of R(q); the transpose is applied by dotting columns below.
    r00 = 2 * (w * w + x * x) - 1
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 2 * (w * w + y * y) - 1
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 2 * (w * w + z * z) - 1

    return [
        r00 * vx + r10 * vy + r20 * vz,
        r01 * vx + r11 * vy + r21 * vz,
        r02 * vx + r12 * vy + r22 * vz,
    ]


def ros_time_from_seconds(seconds):
    """Split float seconds into a ROS ``{secs, nsecs}`` stamp."""
    secs = int(math.floor(seconds))
    nsecs = int(round((seconds - secs) * 1e9))
    if nsecs >= 1_000_000_000:  # rounding carried into the next second
        secs += 1
        nsecs -= 1_000_000_000
    return {"secs": secs, "nsecs": nsecs}


def build_odometry_msg(rod_name, stamp_seconds, pos, quat, linvel, angvel,
                       frame_id=DEFAULT_FRAME_ID, twist_frame="body"):
    """Build a ``nav_msgs/Odometry`` message dict for one rod.

    Args:
        rod_name: Rod name, used as ``child_frame_id`` (e.g. ``"rod_01"``).
        stamp_seconds: Header stamp in float seconds.
        pos: World-frame position, 3 elements.
        quat: Orientation as repo-order ``(w, x, y, z)``, 4 elements.
        linvel: World-frame linear velocity, 3 elements.
        angvel: World-frame angular velocity, 3 elements.
        frame_id: Fixed world frame for ``header.frame_id``.
        twist_frame: ``"body"`` (ROS convention, rotates twist by the inverse of
            ``quat``) or ``"world"`` (publish world-frame velocities as-is).

    Returns:
        A dict matching the ``nav_msgs/Odometry`` layout rosbridge expects.
    """
    if twist_frame not in ("body", "world"):
        raise ValueError(f"twist_frame must be 'body' or 'world', got {twist_frame!r}")

    pos = _as_floats(pos, 3, "pos")
    linvel = _as_floats(linvel, 3, "linvel")
    angvel = _as_floats(angvel, 3, "angvel")

    if twist_frame == "body":
        linvel = rotate_world_to_body(quat, linvel)
        angvel = rotate_world_to_body(quat, angvel)

    return {
        "header": {
            "stamp": ros_time_from_seconds(stamp_seconds),
            "frame_id": frame_id,
        },
        "child_frame_id": rod_name,
        "pose": {
            "pose": {
                "position": _vec3(pos),
                "orientation": quat_wxyz_to_ros(quat),
            },
            "covariance": list(_ZERO_COVARIANCE),
        },
        "twist": {
            "twist": {
                "linear": _vec3(linvel),
                "angular": _vec3(angvel),
            },
            "covariance": list(_ZERO_COVARIANCE),
        },
    }


def split_rod_states(state):
    """Split a flat EKF state into per-rod ``(pos, quat, linvel, angvel)`` tuples."""
    flat = _as_floats(state, _flat_len(state), "state")
    if len(flat) % STATE_DIM_PER_ROD != 0:
        raise ValueError(
            f"state length {len(flat)} is not a multiple of {STATE_DIM_PER_ROD}"
        )
    rods = []
    for i in range(len(flat) // STATE_DIM_PER_ROD):
        b = flat[i * STATE_DIM_PER_ROD:(i + 1) * STATE_DIM_PER_ROD]
        rods.append((b[0:3], b[3:7], b[7:10], b[10:13]))
    return rods


def _flat_len(value):
    if hasattr(value, "numel"):  # torch.Tensor
        return int(value.numel())
    if hasattr(value, "size") and not callable(value.size):  # numpy.ndarray
        return int(value.size)
    return len(list(value))


def rod_names_from_simulator(simulator):
    """Read ordered rod names from a simulator's robot config.

    The EKF state is laid out in ``simulator.robot.rods`` order, so this keeps
    topic names aligned with state blocks (e.g. ``rod_01``, ``rod_23``,
    ``rod_45`` for the 3-bar config).
    """
    return list(simulator.robot.rods.keys())


class RodStatePublisher:
    """Publishes per-rod EKF estimates as ``nav_msgs/Odometry`` over rosbridge.

    Args:
        url: rosbridge websocket URL. Defaults to the ``ROSBRIDGE_URL``
            environment variable, else ``ws://localhost:9090``.
        rod_names: Ordered rod names matching the EKF state layout. If omitted,
            names are derived lazily as ``rod_0 ... rod_{n-1}`` from the first
            published state; prefer passing ``rod_names_from_simulator(sim)``.
        frame_id: Fixed world frame for ``header.frame_id``.
        topic_namespace: Topic prefix; topics are ``<ns>/<rod_name>/odom``.
        stamp_source: ``"wall"`` publishes wall-clock time (safe default -- works
            without ``/clock`` or ``use_sim_time``); ``"sim"`` publishes the
            rollout's simulated time, which requires the ROS side to run with
            ``use_sim_time`` and a ``/clock`` source to display sensibly.
        twist_frame: ``"body"`` (ROS convention) or ``"world"``. See module docs.
        queue_size: Per-topic rosbridge queue size.
        connect_timeout: Seconds to wait for the websocket handshake.
    """

    def __init__(self, url=None, rod_names=None, frame_id=DEFAULT_FRAME_ID,
                 topic_namespace=DEFAULT_TOPIC_NAMESPACE, stamp_source="wall",
                 twist_frame="body", queue_size=10, connect_timeout=10.0):
        if stamp_source not in ("wall", "sim"):
            raise ValueError(
                f"stamp_source must be 'wall' or 'sim', got {stamp_source!r}"
            )
        if twist_frame not in ("body", "world"):
            raise ValueError(
                f"twist_frame must be 'body' or 'world', got {twist_frame!r}"
            )

        self.url = url or os.environ.get("ROSBRIDGE_URL", DEFAULT_ROSBRIDGE_URL)
        self.rod_names = list(rod_names) if rod_names is not None else None
        self.frame_id = frame_id
        self.topic_namespace = topic_namespace.rstrip("/")
        self.stamp_source = stamp_source
        self.twist_frame = twist_frame
        self.queue_size = queue_size
        self.connect_timeout = connect_timeout

        self._ros = None
        self._topics = {}

    # -- connection lifecycle ------------------------------------------------

    def connect(self):
        """Open the rosbridge websocket. Idempotent."""
        if self._ros is not None:
            return self

        import roslibpy  # lazy: keeps this module importable without roslibpy

        url = self.url
        if "://" not in url:
            url = "ws://" + url
        scheme, _, hostport = url.partition("://")
        host, _, port = hostport.partition(":")
        ros = roslibpy.Ros(
            host=host,
            port=int(port) if port else 9090,
            is_secure=(scheme == "wss"),
        )
        ros.run(timeout=self.connect_timeout)
        if not ros.is_connected:
            raise ConnectionError(f"could not connect to rosbridge at {self.url}")
        self._ros = ros
        return self

    def close(self):
        """Unadvertise topics and close the websocket. Idempotent."""
        for topic in self._topics.values():
            try:
                topic.unadvertise()
            except Exception:  # noqa: BLE001 - best effort during teardown
                pass
        self._topics.clear()
        if self._ros is not None:
            try:
                self._ros.terminate()
            finally:
                self._ros = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- publishing ----------------------------------------------------------

    def topic_name(self, rod_name):
        return f"{self.topic_namespace}/{rod_name}/odom"

    def _topic(self, rod_name):
        if rod_name not in self._topics:
            if self._ros is None:
                raise RuntimeError("not connected; call connect() first")
            import roslibpy

            topic = roslibpy.Topic(
                self._ros,
                self.topic_name(rod_name),
                ODOMETRY_MSG_TYPE,
                queue_size=self.queue_size,
            )
            topic.advertise()
            self._topics[rod_name] = topic
        return self._topics[rod_name]

    def _stamp(self, sim_time):
        return sim_time if self.stamp_source == "sim" else _time.time()

    def publish_rod_state(self, rod_name, time, pos, quat, linvel, angvel):
        """Publish one rod's state. ``time`` is the rollout's simulated time."""
        msg = build_odometry_msg(
            rod_name,
            self._stamp(time),
            pos,
            quat,
            linvel,
            angvel,
            frame_id=self.frame_id,
            twist_frame=self.twist_frame,
        )
        import roslibpy

        self._topic(rod_name).publish(roslibpy.Message(msg))
        return msg

    def publish_state(self, time, state):
        """Publish every rod in one flat EKF state vector.

        This is the hook ``run_ekf_rollout`` calls once per timestep.
        """
        rods = split_rod_states(state)
        if self.rod_names is None:
            self.rod_names = [f"rod_{i}" for i in range(len(rods))]
        if len(self.rod_names) != len(rods):
            raise ValueError(
                f"have {len(self.rod_names)} rod names but state holds "
                f"{len(rods)} rods"
            )
        return [
            self.publish_rod_state(name, time, pos, quat, linvel, angvel)
            for name, (pos, quat, linvel, angvel) in zip(self.rod_names, rods)
        ]
