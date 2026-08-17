# Design: Streaming EKF State to ROS Noetic (Docker) via roslibpy

## Context

`ekf.py`'s `run_ekf_rollout()` produces, at every timestep, a state estimate for
each rod: 13 values = position (3) + quaternion (4) + linear velocity (3) +
angular velocity (3) (see `_ekf_step_gtsam` / the `frames` list it returns).
Right now that state only exists as torch tensors inside the Python process
that also does the GNN-based simulation/linearization (torch, gtsam — no ROS
installed). The goal is to get each rod's estimated pose+velocity into ROS
Noetic, which will run in a separate Docker container (e.g. for visualization
in RViz, or consumption by other ROS nodes), without installing ROS/catkin
into this repo's environment.

This is a **design/research document** — no code has been implemented yet.
It captures the architecture, message mapping, and integration points so a
future implementation pass can be scoped precisely.

## Chosen architecture: roslibpy + rosbridge_suite

```
┌─────────────────────────────┐        websocket (port 9090)        ┌──────────────────────────────┐
│ This repo (host or its own  │ ───────────────────────────────────▶│ Docker: ROS Noetic container   │
│ container). Python 3.10,    │   roslibpy.Topic(...).publish(msg)  │  - roscore                     │
│ torch, gtsam. No ROS needed.│                                      │  - rosbridge_server (rosbridge_│
│  ekf.py -> sim_data_        │                                      │    websocket, port 9090)       │
│  publisher.py               │                                      │  - RViz / other ROS nodes      │
└─────────────────────────────┘                                      └──────────────────────────────┘
```

- The Noetic container runs `roscore` plus `rosbridge_websocket` (from
  `rosbridge_suite`), exposing port 9090.
- This repo's process depends only on the pure-Python `roslibpy` package
  (add to `requirements.txt`) — it talks JSON-over-websocket to rosbridge,
  which republishes onto real ROS topics inside the container. No rospy,
  no catkin workspace, no Python-version coupling to Noetic's Python 3.8.
- Docker networking: if the publisher runs on the host, the container needs
  `-p 9090:9090` published; if it runs in a sibling container, both go on a
  shared `docker-compose` network and connect via the service name
  (e.g. `ws://ros-noetic:9090`).

## Message design: `nav_msgs/Odometry`, one topic per rod

Each rod already behaves like a free rigid body, and `nav_msgs/Odometry` is
built for exactly this: pose (position+orientation) plus twist
(linear+angular velocity), each with an optional covariance. No custom
`.msg` needs to be compiled/sourced inside the Noetic image — this is a
stock message type.

Topic naming: `/tensegrity/<rod_name>/odom`, using the rod names already
present in the robot config (e.g. `rod_01`, `rod_23` from
`simulators/configs/3_bar_tensegrity_gnn_sim_config.json`).

Field mapping, per rod, per EKF state block (`state[13*r : 13*r+13]`):

| EKF state slice | Odometry field | Notes |
|---|---|---|
| `pos = state[0:3]` | `pose.pose.position.{x,y,z}` | direct copy |
| `quat = state[3:7]` | `pose.pose.orientation.{x,y,z,w}` | **reorder required**: repo's `Quaternion` (see `utilities/torch_quaternion.py`) stores `(w, x, y, z)`; ROS `geometry_msgs/Quaternion` is `(x, y, z, w)`. |
| `linvel = state[7:10]` | `twist.twist.linear.{x,y,z}` | see frame-convention note below |
| `angvel = state[10:13]` | `twist.twist.angular.{x,y,z}` | see frame-convention note below |
| — | `header.stamp` | ROS time at publish, or simulated time = rollout `time` (frame's `"time"` key) — decide per whether this is live wall-clock or `/use_sim_time` playback |
| — | `header.frame_id` | fixed world frame, e.g. `"world"` |
| — | `child_frame_id` | rod name, e.g. `"rod_01"` |

Two things flagged for verification before implementation (not resolved by
this design pass):

1. **Velocity frame convention.** `nav_msgs/Odometry.twist` is conventionally
   expressed in `child_frame_id` (body-fixed) axes. `state_objects/composite_body.py`
   computes per-point velocity as `linear_vel + cross(ang_vel, body_vec)`,
   which reads as world-frame `ang_vel`/`linear_vel` composed with a
   world-frame lever arm. If the EKF's `linvel`/`angvel` are in fact
   world-frame (not body-frame), they need rotation into the body frame by
   the inverse of `quat` before publishing, or `twist_frame_id`-equivalent
   documentation needs to note they're expressed in the world/header frame
   instead (Odometry has no explicit field for that, so it'd be a documented
   deviation from convention).
2. **Covariance mismatch.** GTSAM's `KalmanFilter` state covariance `P` is
   13-dim per rod (quaternion included), but `Odometry.pose.covariance` is a
   6x6 (3 position + 3 *rotation-as-small-angle*, not quaternion). There's no
   exact closed-form copy from the quaternion covariance block into that
   6x6 — options are (a) leave covariance unpopulated/zeroed for v1, (b) a
   small-angle approximation dropping the quaternion's redundant DOF. Recommend
   (a) for the first implementation and revisit if downstream consumers need it.

## Publish timing: live, inside `run_ekf_rollout`

Publishing should happen **live** as each frame is produced, not just from
the saved `frames` list after the fact. Concretely, the hook point is the
end of the per-timestep loop in `run_ekf_rollout` (`ekf.py:528-556`), right
after `state_for_frame` / `pose` / `time` are computed and appended to
`frames` — a publisher call would be added there, one call per rod per step.
The design keeps the publisher decoupled (a small class taking
`(time, rod_name, pos, quat, linvel, angvel)` and doing the roslibpy
publish), so `ekf.py` doesn't need to import roslibpy directly; it would
depend on a thin interface that `sim_data_publisher.py` implements.

## Where implementation would live (for a future pass)

- `sim_data_publisher.py` (currently an empty stub) — becomes the roslibpy
  client: connects to the rosbridge websocket, holds one
  `roslibpy.Topic` per rod (created lazily from rod names), and exposes a
  `publish_rod_state(rod_name, time, pos, quat, linvel, angvel)` method that
  does the quaternion reorder + Odometry message construction + publish.
- `ekf.py` — `run_ekf_rollout` would gain an optional `publisher=None`
  parameter; when provided, it's called once per rod at the point noted
  above. No behavior change when `publisher` is `None` (keeps existing
  offline/eval usage, e.g. `eval.py`, untouched).
- `requirements.txt` — add `roslibpy`.
- New (documentation-only) Docker note: a minimal `docker-compose.yml`
  sketch showing a `ros:noetic` service running
  `roscore & roslaunch rosbridge_server rosbridge_websocket.launch`, with
  port 9090 exposed — this is ROS-side setup, not part of this repo's
  Python code, but worth documenting since nothing like it exists yet.

## Open questions to confirm before implementation

- Confirm whether `linvel`/`angvel` in the state vector are world-frame or
  body-frame (affects whether a rotation-by-quat step is needed before
  publishing twist).
- Confirm `header.stamp` policy: real wall-clock time (simplest) vs.
  ROS `/clock` + `use_sim_time` if this is meant to look like a live sensor
  driver during offline rollout replay.
- Confirm whether covariance should be populated at all in v1 (recommend: no).
- Confirm the rosbridge host/port the Noetic container will expose, so the
  publisher's default connection target can be set sensibly (e.g. via env
  var `ROSBRIDGE_URL`, default `ws://localhost:9090`).

## Verification (once implemented)

- Unit-level: mock `roslibpy.Ros`/`Topic`, assert quaternion reorder and
  field mapping are correct for a synthetic state vector.
- Integration: run `docker-compose up` (Noetic + rosbridge), run a short
  EKF rollout with the publisher attached, and confirm with
  `rostopic echo /tensegrity/rod_01/odom` (from inside the container) that
  messages arrive at the expected rate with sane values, and that RViz can
  render the rod's `Odometry` display without frame errors.
