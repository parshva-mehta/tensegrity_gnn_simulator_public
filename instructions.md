# Streaming EKF state to ROS over the rosbridge websocket

How to set up and run the live websocket link between this repo's EKF and a ROS
Noetic container. Design rationale lives in
[`docs/ekf_ros_integration_design.md`](docs/ekf_ros_integration_design.md).

## How it works

```
┌─────────────────────────────────┐   websocket :9090   ┌──────────────────────────────┐
│ This repo (host)                │ ──────────────────► │ Docker: ROS Noetic            │
│ python 3.10, torch, gtsam       │  roslibpy publishes │  roscore                      │
│ NO ROS installed                │  JSON over ws       │  rosbridge_websocket  :9090   │
│                                 │                     │  foxglove_bridge      :8765   │
│ ekf.py ──► sim_data_publisher   │                     │  ► /tensegrity/<rod>/odom     │
└─────────────────────────────────┘                     └──────────────────────────────┘
```

Nothing ROS-related is installed on the host side — the only dependency is the
pure-Python `roslibpy` package. rosbridge republishes the JSON onto real ROS
topics inside the container as `nav_msgs/Odometry`, one topic per rod:

| Rod index | Rod name | Color (ROS side) | Topic |
|---|---|---|---|
| 0 | `rod_01` | red | `/tensegrity/rod_01/odom` |
| 1 | `rod_23` | green | `/tensegrity/rod_23/odom` |
| 2 | `rod_45` | blue | `/tensegrity/rod_45/odom` |

## Prerequisites

- Docker with Compose v2 (`docker compose`, not `docker-compose`).
- A checkout of the companion catkin workspace
  ([`parshva-mehta/catkin_ws`](https://github.com/parshva-mehta/catkin_ws)),
  which holds the `Dockerfile` that builds the ROS image.
- This repo's Python env (see `README.md`).

## 1. Install the host-side dependency

```bash
conda activate tensegrity_gnn
pip install roslibpy          # already listed in requirements.txt
```

## 2. Start the ROS container

The image is defined by the catkin workspace's `Dockerfile`. It already installs
`ros-noetic-rosbridge-server` and `ros-noetic-foxglove-bridge`, and its default
command (`start_bridges.sh`) starts `roscore`, rosbridge on **9090**, and
foxglove_bridge on **8765**. You do **not** need to build the catkin workspace
for this path — `nav_msgs/Odometry` is a stock ROS message.

### Option A — compose (from this repo)

```bash
CATKIN_WS=../catkin_ws docker compose \
  -f docker/docker-compose.ros-noetic.yml up --build
```

Set `CATKIN_WS` to your checkout. Adjust ports with `ROSBRIDGE_PORT` /
`FOXGLOVE_PORT` if 9090 or 8765 are taken.

### Option B — plain docker run (from the catkin workspace)

```bash
cd /path/to/catkin_ws
docker build --pull --platform linux/amd64 -t tensegrity:noetic .
docker run --rm -it --platform linux/amd64 \
  -p 9090:9090 -p 8765:8765 \
  -v "$(pwd)":/ws \
  tensegrity:noetic
```

`--platform linux/amd64` is required on Apple Silicon (the image builds GTSAM and
pulls CPU LibTorch, both amd64-only) and harmless elsewhere.

You should see rosbridge announce itself:

```
[INFO] [...]: Rosbridge WebSocket server started on port 9090
```

## 3. Confirm the bridge is up

From another shell:

```bash
docker compose -f docker/docker-compose.ros-noetic.yml exec ros-noetic \
  bash -lc 'rostopic list'
```

`/rosout` and friends should appear. If you used Option B, use
`docker exec -it <container> bash -lc 'rostopic list'`.

## 4. Smoke-test the websocket

Before wiring up the EKF, confirm the link end to end with synthetic data. This
needs neither torch nor a dataset:

```bash
export ROSBRIDGE_URL=ws://localhost:9090
python3 - <<'PY'
import math, time
from sim_data_publisher import RodStatePublisher

ROD_NAMES = ["rod_01", "rod_23", "rod_45"]

with RodStatePublisher(rod_names=ROD_NAMES) as pub:
    for k in range(500):
        t = k * 0.01
        state = []
        for r in range(3):
            state += [
                math.sin(t) + r, math.cos(t), 1.0,   # pos
                1.0, 0.0, 0.0, 0.0,                  # quat (w, x, y, z)
                0.0, 0.0, 0.0,                       # linvel
                0.0, 0.0, 0.0,                       # angvel
            ]
        pub.publish_state(t, state)
        time.sleep(0.01)
PY
```

While it runs, watch the topic from inside the container:

```bash
docker compose -f docker/docker-compose.ros-noetic.yml exec ros-noetic \
  bash -lc 'rostopic echo /tensegrity/rod_01/odom'

# or just the rate
docker compose -f docker/docker-compose.ros-noetic.yml exec ros-noetic \
  bash -lc 'rostopic hz /tensegrity/rod_01/odom'
```

You should see ~100 Hz of `Odometry` messages with `child_frame_id: "rod_01"`.

## 5. Run the EKF with the publisher attached

```python
import json
from pathlib import Path

import torch

from ekf import run_ekf_rollout
from sim_data_publisher import RodStatePublisher, rod_names_from_simulator

model_path = Path("sample_model.pt")
data_dir = Path("../tensegrity/data_sets/mjc_synthetic_5d_0.01/val/R2S2Rrolling_7/")

simulator = torch.load(model_path, map_location="cpu")
simulator.eval()
simulator.to("cpu")

gt_data = json.load((data_dir / "processed_data.json").open("r"))
extra_gt_data = json.load((data_dir / "5d_extra_state_data.json").open("r"))

# rod_names_from_simulator keeps topic names aligned with the state layout.
with RodStatePublisher(rod_names=rod_names_from_simulator(simulator)) as pub:
    frames = run_ekf_rollout(
        simulator,
        gt_data,
        extra_gt_data,
        dt=0.01,
        publisher=pub,
    )

print(f"published {len(frames)} frames")
```

`run_ekf_rollout` publishes once for the initial state and once per timestep. With
`publisher=None` (the default) behavior is unchanged, so existing offline callers
such as `eval.py` are unaffected.

## Configuration

`RodStatePublisher` options, all optional:

| Argument | Default | Notes |
|---|---|---|
| `url` | `$ROSBRIDGE_URL`, else `ws://localhost:9090` | Use `ws://ros-noetic:9090` from a sibling container on the compose network. `wss://` works too. |
| `rod_names` | derived as `rod_0…rod_N` | Pass `rod_names_from_simulator(simulator)` to match the config. |
| `frame_id` | `"world"` | `header.frame_id`. |
| `topic_namespace` | `"/tensegrity"` | Topics are `<ns>/<rod_name>/odom`. |
| `stamp_source` | `"wall"` | `"sim"` stamps with the rollout's simulated time — only useful with `/use_sim_time` and a `/clock` source, otherwise messages look ancient to RViz. |
| `twist_frame` | `"body"` | Rotates the twist into body axes, per the `Odometry` convention. `"world"` publishes the raw world-frame velocities. |
| `queue_size` | `10` | Per-topic rosbridge queue size. |
| `connect_timeout` | `10.0` | Seconds for the websocket handshake. |

## Visualizing

**Foxglove** (easiest — the image already runs it on 8765): open
[app.foxglove.dev](https://app.foxglove.dev), connect to
`ws://localhost:8765`, and add a 3D panel. It renders `Odometry` natively.

**RViz** needs a TF frame matching `header.frame_id`, which
`start_bridges.sh` does not publish on its own. Add one:

```bash
docker compose -f docker/docker-compose.ros-noetic.yml exec ros-noetic \
  bash -lc 'rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 1 world map'
```

Then set RViz's Fixed Frame to `world` and add an Odometry display per topic.

> **Units.** Positions go out in **raw simulator units**, matching the file-based
> path where the ROS side applies its own `data_scale_factor` (default `0.10`) to
> convert to meters. Nothing scales the websocket path, so rods appear ~10x
> oversized in RViz/Foxglove. Scale on the viewer side, or multiply positions by
> `0.1` before publishing if you want true meters.

## Troubleshooting

**`ConnectionError: could not connect to rosbridge at ...`**
The container isn't up, or the port isn't published. Check `docker ps` shows
`0.0.0.0:9090->9090/tcp`, and that rosbridge logged "server started".

**Connects, but `rostopic list` shows no `/tensegrity/...` topics**
rosbridge only advertises a topic on first publish. Run the smoke test in step 4
and re-check while it's running.

**`rostopic echo` prints nothing**
Check for a namespace mismatch — `topic_namespace` on the publisher must match
what you're echoing. `rostopic list | grep tensegrity` shows what actually exists.

**Messages arrive but RViz shows "Fixed Frame [world] does not exist"**
No TF is being published; add the `static_transform_publisher` above.

**Timestamps look wrong / RViz drops messages as too old**
You're on `stamp_source="sim"` without a `/clock` source. Use the default
`"wall"`, or launch with `sim_clock:=true` on the ROS side.

**Apple Silicon: image fails to build or exits immediately**
Missing `--platform linux/amd64`. The compose file sets this already; override
with `DOCKER_PLATFORM` if needed.

**Port 9090 already in use**
`ROSBRIDGE_PORT=9091 CATKIN_WS=../catkin_ws docker compose ... up`, then
`export ROSBRIDGE_URL=ws://localhost:9091`.

## The file-based alternative

The websocket is not the only path, and for the existing ROS workflow it isn't
the shortest one. The `interface` package's own `sim_data_publisher.py` replays a
39-column text file whose format is exactly this repo's EKF state layout, and
`eval.py` already writes it via `save_rollout_txt`:

```bash
python3 eval.py                      # writes ./rollout_ekf.txt
```

```bash
roslaunch interface simulated_data.launch data_file:=/ws/rollout_ekf.txt
```

That path needs no websocket and no ROS-side changes, and it feeds the existing
`TensegrityBars` visualization. Use the websocket when you want the estimate
*live*, or when you need the velocity estimates — `TensegrityBars` has no twist
fields, `Odometry` does.

`sim_data_publisher.RolloutStateFileWriter` writes the same format incrementally
during a rollout (rather than after it, as `save_rollout_txt` does), at full
float64 precision, and `CompositeSink` runs it alongside the live publisher:

```python
from sim_data_publisher import (
    CompositeSink, RodStatePublisher, RolloutStateFileWriter,
    rod_names_from_simulator,
)

sinks = CompositeSink(
    RolloutStateFileWriter("rollout_ekf.txt"),
    RodStatePublisher(rod_names=rod_names_from_simulator(simulator)),
)
with sinks:
    run_ekf_rollout(simulator, gt_data, extra_gt_data, dt=0.01, publisher=sinks)
```

## Running the tests

```bash
pip install pytest websockets
python3 -m pytest tests/ -q
```

The suite mocks `roslibpy` and additionally drives the real client against a stub
rosbridge websocket server, so it needs no ROS installation.
