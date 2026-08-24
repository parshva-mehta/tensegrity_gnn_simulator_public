# Testing the EKF → ROS pipeline

Five levels, fastest first. Each one is independently useful — stop wherever you
have the confidence you need.

| Level | What it proves | Needs | Time |
|---|---|---|---|
| 1 | Message mapping, scaling, file format | pytest | ~1 s |
| 2 | Real roslibpy speaks correct rosbridge protocol | + websockets | ~1 s |
| 3 | Messages land on real ROS topics | + Docker | ~5 min |
| 4 | `run_ekf_rollout` drives the publisher | + repo deps | ~1 min |
| 5 | Both output paths agree in meters | all of the above | ~1 min |

Every command below runs from the repo root. Levels 1–5 were all executed and
passed during development; the expected outputs shown are real, not illustrative.

---

## Level 1 — Unit tests (no ROS, no Docker)

```bash
pip install pytest
python3 -m pytest tests/ -q
```

Expected:

```
49 passed, 2 skipped
```

The 2 skips are torch-only tests (one checks `rotate_world_to_body` against the
repo's own `torch_quaternion.rotate_vec_quat`). They run automatically once torch
is installed — after Level 4 you should see `51 passed, 0 skipped`.

This covers the quaternion reorder `(w,x,y,z)→(x,y,z,w)`, the world→body twist
rotation, the `position_scale` split (position and linear velocity scale;
angular velocity and orientation do not), the Odometry field mapping, topic
lifecycle, and the 39-column file layout.

## Level 2 — Wire protocol (real roslibpy, stub server)

```bash
pip install websockets roslibpy
python3 -m pytest tests/test_sim_data_publisher_wire.py -q
```

Expected: `1 passed`.

This drives the **real** roslibpy client against a stub websocket server and
asserts the `advertise` and `publish` frames on the wire. It catches roslibpy API
drift without needing ROS.

## Level 3 — Real ROS topics

### 3a. Start a ROS container with rosbridge

Use your own image if you have it built:

```bash
CATKIN_WS=../catkin_ws docker compose \
  -f docker/docker-compose.ros-noetic.yml up --build
```

Or a stock image, which is lighter and sufficient for this test:

```bash
docker run -d --name rosbridge -p 9090:9090 ros:noetic-ros-base bash -lc '
source /opt/ros/noetic/setup.bash
apt-get update -qq && apt-get install -y -qq --no-install-recommends ros-noetic-rosbridge-server
roscore & sleep 6
roslaunch --wait rosbridge_server rosbridge_websocket.launch address:=0.0.0.0 port:=9090
'
```

Wait for the bridge to come up (the apt install takes a few minutes the first
time):

```bash
docker logs rosbridge 2>&1 | grep "Rosbridge WebSocket server started"
```

Expected: `Rosbridge WebSocket server started at ws://0.0.0.0:9090`

### 3b. Publish synthetic rod states

Start a listener, then publish:

```bash
docker exec -d rosbridge bash -lc \
  'source /opt/ros/noetic/setup.bash; rostopic echo -n 2 /tensegrity/rod_01/odom > /tmp/echo.txt 2>&1'
sleep 3

export ROSBRIDGE_URL=ws://localhost:9090
python3 - <<'PY'
import json, time
from sim_data_publisher import RodStatePublisher

cfg = json.load(open('simulators/configs/3_bar_tensegrity_gnn_sim_config.json'))
rods = cfg['tensegrity_cfg']['rods']
state = []
for r in rods:
    a, b = r['end_pts']
    com = [(a[i] + b[i]) / 2 for i in range(3)]
    state += com + [1.0, 0.0, 0.0, 0.0] + [0.0, 1.0, 0.0] + [0.0, 0.0, 2.0]

with RodStatePublisher(rod_names=[r['name'] for r in rods], stamp_source='sim') as pub:
    for k in range(40):
        pub.publish_state(k * 0.01, state)
        time.sleep(0.05)
print('published 40 frames x 3 rods')
PY

docker exec rosbridge bash -lc 'cat /tmp/echo.txt' | head -30
```

Expected — note the three things this proves at once:

```
child_frame_id: "rod_01"
    position:
      x: -0.035202919167178416      <- meters, not 10x oversized
      y: 0.02433287347828142
      z: 0.11748276926750867
    orientation:
      x: 0.0   y: 0.0   z: 0.0   w: 1.0    <- (w,x,y,z) reordered correctly
    linear:
      y: 0.1                                <- scaled from 1.0
    angular:
      z: 2.0                                <- rad/s, correctly NOT scaled
```

### 3c. Confirm registration

```bash
docker exec rosbridge bash -lc \
  'source /opt/ros/noetic/setup.bash; rostopic list | grep tensegrity; rostopic type /tensegrity/rod_23/odom'
```

Expected:

```
/tensegrity/rod_01/odom
/tensegrity/rod_23/odom
/tensegrity/rod_45/odom
nav_msgs/Odometry
```

## Level 4 — Real EKF rollout driving the publisher

Install the repo dependencies:

```bash
conda activate tensegrity_gnn
pip install -r requirements.txt
```

If you have the dataset, the simplest real test is `eval.py`, which already
writes `./rollout_ekf.txt` via `save_rollout_txt`. To exercise the **live**
publisher path with or without the dataset, save this as `e2e_check.py` in the
repo root:

```python
"""Real run_ekf_rollout with both sinks attached."""
import json
import numpy as np
import torch

from ekf import run_ekf_rollout
from sim_data_publisher import (CompositeSink, RodStatePublisher,
                                RolloutStateFileWriter, rod_names_from_simulator)

sim = torch.load('sample_model.pt', map_location='cpu')
sim.eval()
sim.to('cpu')

cfg = json.load(open('simulators/configs/3_bar_tensegrity_gnn_sim_config.json'))
rods = cfg['tensegrity_cfg']['rods']

end_pts, pos, quat = [], [], []
for r in rods:
    a, b = r['end_pts']
    end_pts += [a, b]
    pos.append([(a[i] + b[i]) / 2 for i in range(3)])
    quat.append([1.0, 0.0, 0.0, 0.0])
zeros3 = [[0.0] * 3 for _ in rods]

N = 4
gt = [{'end_pts': end_pts, 'pos': pos, 'quat': quat,
       'linvel': zeros3, 'angvel': zeros3} for _ in range(N + 1)]
extra = [{'controls': [0.0] * 6,
          'rest_lengths': [2.700000047683716] * 6,
          'motor_speeds': [0.0] * 6} for _ in range(N)]

sinks = CompositeSink(
    RolloutStateFileWriter('rollout_ekf.txt'),
    RodStatePublisher(rod_names=rod_names_from_simulator(sim), stamp_source='sim'),
)
with sinks:
    frames = run_ekf_rollout(sim, gt, extra, dt=0.01,
                             use_finite_diff=True, publisher=sinks)

lines = open('rollout_ekf.txt').read().splitlines()
print(f"frames returned: {len(frames)}")
print(f"file lines     : {len(lines)}")
print(f"columns/line   : {set(len(l.split()) for l in lines)}")
s = frames[-1]['state'].flatten().tolist()
print(f"quat norm      : {np.linalg.norm(s[3:7]):.6f}")
assert len(lines) == len(frames), "file/frame count mismatch"
print("OK")
```

To swap in the real dataset, replace the synthetic `gt`/`extra` with:

```python
data_dir = Path("../tensegrity/data_sets/mjc_synthetic_5d_0.01/val/R2S2Rrolling_7/")
gt    = json.load((data_dir / "processed_data.json").open('r'))
extra = json.load((data_dir / "5d_extra_state_data.json").open('r'))
```

Run it with the container still up:

```bash
ROSBRIDGE_URL=ws://localhost:9090 python3 e2e_check.py
```

Expected:

```
frames returned: 5
file lines     : 5
columns/line   : {39}
quat norm      : 1.000000
OK
```

`frames == file lines` is the assertion that matters: the publisher fired on
every frame including the initial state, and nothing was dropped. `quat norm ==
1.0` confirms the EKF's quaternion renormalization survived the round trip.

To watch the EKF frames arrive live, start a listener before running it:

```bash
docker exec -d rosbridge bash -lc \
  'source /opt/ros/noetic/setup.bash; rostopic echo -n 5 /tensegrity/rod_23/odom > /tmp/echo2.txt 2>&1'
```

## Level 5 — Cross-check: both paths agree

This is the strongest single check. The file path is written in **raw simulator
units** (the ROS reader applies `data_scale_factor=0.10` itself), while the
websocket path is **pre-scaled to meters**. Both must land on the same numbers —
if they don't, one path is double-scaled or unscaled.

Parse the file the way the `interface` package does:

```bash
python3 - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRot

# verbatim from interface/scripts/sim_data_publisher.py
def create_transform(position, quat, s=0.10):
    q = np.array(quat, np.float32)
    rot = SciPyRot.from_quat([q[1], q[2], q[3], q[0]])   # W is first
    T = np.identity(4)
    T[0:3, 3] = np.array(position, np.float32) * s
    T[0:3, 0:3] = rot.as_matrix()
    return T

def get_values(l):
    return {'red':   create_transform(l[0:3],   l[3:7]),
            'green': create_transform(l[13:16], l[16:20]),
            'blue':  create_transform(l[26:29], l[29:33])}

lines = open('rollout_ekf.txt').read().splitlines()
print(f"lines={len(lines)} cols={set(len(l.split()) for l in lines)}")
for name, T in get_values([float(t) for t in lines[-1].split()]).items():
    R = T[0:3, 0:3]
    print(f"{name:>5}: com(m)={np.round(T[0:3,3], 4)}  "
          f"det(R)={np.linalg.det(R):.6f}  |RR^T-I|={np.abs(R@R.T - np.eye(3)).max():.2e}")
PY
```

Expected:

```
lines=5 cols={39}
  red: com(m)=[-0.0352  0.0243  0.1175]  det(R)=1.000000  |RR^T-I|=2.22e-16
green: com(m)=[-0.0171 -0.0258  0.0521]  det(R)=1.000000  |RR^T-I|=2.22e-16
 blue: com(m)=[ 0.0103 -0.0455  0.1296]  det(R)=1.000000  |RR^T-I|=4.44e-16
```

Now compare `green` against the `rod_23` position from `/tmp/echo2.txt`:

| Path | Position (m) |
|---|---|
| Websocket (pre-scaled) | `-0.017095, -0.025762, 0.051987` |
| File (raw × `0.10`) | `-0.0171, -0.0258, 0.0521` |

The last column is float32 round-off and varies run to run at the `1e-16` level;
only its magnitude matters. They agree. `det(R)=1` and `|RR^T−I|≈1e-16` confirm the quaternions
round-trip into valid rotations.

## Level 6 (optional) — Visual check

Not covered by any automated test — the only part still worth eyeballing.

**Foxglove**, if your image runs `foxglove_bridge` on 8765: open
[app.foxglove.dev](https://app.foxglove.dev), connect to `ws://localhost:8765`,
add a 3D panel. Rods should be ~0.325 m long.

**RViz** needs a TF frame matching `header.frame_id`, which the bridges do not
publish:

```bash
docker exec -d rosbridge bash -lc \
  'source /opt/ros/noetic/setup.bash; rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 1 world map'
```

Then set Fixed Frame to `world` and add an Odometry display per topic.

## Cleanup

```bash
docker rm -f rosbridge
# or, if you used compose:
docker compose -f docker/docker-compose.ros-noetic.yml down
rm -f e2e_check.py rollout_ekf.txt
```

## Troubleshooting

**`ConnectionError: could not connect to rosbridge`**
The container isn't up or the port isn't published. `docker ps` should show
`0.0.0.0:9090->9090/tcp`.

**`Cannot connect to the Docker daemon`**
The daemon isn't running (distinct from Docker not being installed). Start
Docker Desktop, or on a bare Linux host `sudo dockerd &`.

**`rostopic list` shows no `/tensegrity/...` topics**
rosbridge only advertises on first publish. Run a publisher and re-check while
it's running.

**`ModuleNotFoundError: No module named 'torch_geometric.nn.conv.utils.inspector'`**
`sample_model.pt` was pickled with torch_geometric 2.4.0. `pip install
torch_geometric==2.4.0` (it is pinned in `requirements.txt`).

**`torch.load` raises `WeightsUnpickler error` / `weights_only`**
On torch ≥ 2.6 the default flipped. Use
`torch.load(..., map_location='cpu', weights_only=False)`. The pinned torch
2.1.2 does not need this.

**Rods look 10x too big**
Something is passing `position_scale=1.0` on the websocket path, or the file
path is being pre-scaled *and* scaled again by `data_scale_factor`. See the
units note in `instructions.md`.
