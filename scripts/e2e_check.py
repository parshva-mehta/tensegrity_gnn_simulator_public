#!/usr/bin/env python3
"""End-to-end check for the EKF -> ROS pipeline.

Runs a real `run_ekf_rollout` with the sinks attached, writes the ROS-readable
rollout file, and optionally streams live `nav_msgs/Odometry` to rosbridge.
Works with or without the dataset -- without `--data-dir` it synthesizes
`gt_data`/`extra_gt_data` from the robot config so the full code path still runs.

Examples:
    # file only, synthetic data
    python3 scripts/e2e_check.py

    # also stream to a running rosbridge
    python3 scripts/e2e_check.py --ros

    # use the real dataset
    python3 scripts/e2e_check.py --ros \
        --data-dir ../tensegrity/data_sets/mjc_synthetic_5d_0.01/val/R2S2Rrolling_7/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as `python3 scripts/e2e_check.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ekf import run_ekf_rollout
from sim_data_publisher import (STATE_DIM_PER_ROD, CompositeSink,
                                RodStatePublisher, RolloutStateFileWriter,
                                rod_names_from_simulator)

DEFAULT_CONFIG = "simulators/configs/3_bar_tensegrity_gnn_sim_config.json"


def load_simulator(model_path):
    try:  # torch >= 2.6 defaults weights_only=True, which cannot unpickle this
        sim = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.4 has no weights_only kwarg
        sim = torch.load(model_path, map_location="cpu")
    sim.eval()
    sim.to("cpu")
    return sim


def synthetic_data(config_path, simulator, n_steps):
    """Build a minimal gt/extra pair from the robot config (no dataset needed)."""
    cfg = json.load(open(config_path))
    rods = cfg["tensegrity_cfg"]["rods"]

    end_pts, pos, quat = [], [], []
    for rod in rods:
        a, b = rod["end_pts"]
        end_pts += [a, b]
        pos.append([(a[i] + b[i]) / 2 for i in range(3)])
        quat.append([1.0, 0.0, 0.0, 0.0])
    zeros3 = [[0.0] * 3 for _ in rods]

    cables = list(simulator.robot.actuated_cables.values())
    rest_lengths = [float(c._rest_length) for c in cables]

    gt = [{"end_pts": end_pts, "pos": pos, "quat": quat,
           "linvel": zeros3, "angvel": zeros3} for _ in range(n_steps + 1)]
    extra = [{"controls": [0.0] * len(cables),
              "rest_lengths": rest_lengths,
              "motor_speeds": [0.0] * len(cables)} for _ in range(n_steps)]
    return gt, extra


def load_dataset(data_dir, n_steps):
    data_dir = Path(data_dir)
    gt = json.load((data_dir / "processed_data.json").open("r"))
    extra = json.load((data_dir / "5d_extra_state_data.json").open("r"))
    if n_steps > 0:
        gt, extra = gt[:n_steps + 1], extra[:n_steps]
    return gt, extra


def check_file(path, n_frames, n_rods):
    """Validate the written file the way the ROS interface package reads it."""
    lines = Path(path).read_text().splitlines()
    widths = {len(line.split()) for line in lines}
    expected_cols = n_rods * STATE_DIM_PER_ROD

    ok = True
    print(f"  frames returned : {n_frames}")
    print(f"  file lines      : {len(lines)}")
    print(f"  columns/line    : {widths} (expected {{{expected_cols}}})")
    if len(lines) != n_frames:
        print("  FAIL: file line count != frame count"); ok = False
    if widths != {expected_cols}:
        print(f"  FAIL: unexpected column count"); ok = False

    try:
        from scipy.spatial.transform import Rotation as SciPyRot
    except ImportError:
        print("  (scipy missing -- skipping rotation validity check)")
        return ok

    values = [float(t) for t in lines[-1].split()]
    names = ["red", "green", "blue"]
    for r in range(n_rods):
        b = values[r * STATE_DIM_PER_ROD:(r + 1) * STATE_DIM_PER_ROD]
        q = np.array(b[3:7], np.float32)
        # create_transform() in interface/scripts/sim_data_publisher.py: W first
        R = SciPyRot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        com_m = np.array(b[0:3], np.float32) * 0.10  # ROS data_scale_factor
        resid = np.abs(R @ R.T - np.eye(3)).max()
        label = names[r] if r < len(names) else f"rod{r}"
        print(f"  {label:>5}: com(m)={np.round(com_m, 4)}  "
              f"det(R)={np.linalg.det(R):.6f}  |RR^T-I|={resid:.2e}")
        if abs(np.linalg.det(R) - 1.0) > 1e-5 or resid > 1e-5:
            print(f"  FAIL: {label} rotation is not orthonormal"); ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="sample_model.pt")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None,
                    help="Real dataset directory; omit to use synthetic data.")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--out", default="rollout_ekf.txt")
    ap.add_argument("--ros", action="store_true",
                    help="Also stream live Odometry to rosbridge.")
    ap.add_argument("--rosbridge-url", default=None,
                    help="Overrides $ROSBRIDGE_URL (default ws://localhost:9090).")
    ap.add_argument("--gnn-jacobian", action="store_true",
                    help="Linearize with the GNN Jacobian instead of finite differences.")
    args = ap.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"model not found: {args.model} (run from the repo root)")

    print(f"loading {args.model} ...")
    sim = load_simulator(args.model)
    rod_names = rod_names_from_simulator(sim)
    print(f"  simulator : {type(sim).__name__}")
    print(f"  rods      : {rod_names}")

    if args.data_dir:
        gt, extra = load_dataset(args.data_dir, args.steps)
        print(f"  data      : {args.data_dir} ({len(extra)} steps)")
    else:
        gt, extra = synthetic_data(args.config, sim, args.steps)
        print(f"  data      : synthetic from {args.config} ({len(extra)} steps)")

    sinks = [RolloutStateFileWriter(args.out, expected_n_rods=len(rod_names))]
    if args.ros:
        sinks.append(RodStatePublisher(url=args.rosbridge_url,
                                       rod_names=rod_names, stamp_source="sim"))
        print(f"  rosbridge : {sinks[-1].url}")

    print("\nrunning EKF rollout ...")
    with CompositeSink(*sinks) as sink:
        frames = run_ekf_rollout(sim, gt, extra, args.dt,
                                 use_finite_diff=not args.gnn_jacobian,
                                 publisher=sink)

    print(f"\nchecking {args.out} ...")
    ok = check_file(args.out, len(frames), len(rod_names))

    norm = np.linalg.norm(frames[-1]["state"].flatten().tolist()[3:7])
    print(f"  quat norm       : {norm:.6f}")
    if abs(norm - 1.0) > 1e-4:
        print("  FAIL: quaternion is not normalized"); ok = False

    print("\n" + ("PASS: rollout ran, every frame reached every sink."
                  if ok else "FAIL: see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
