import argparse
import json
from pathlib import Path

import numpy as np

from .adapter import TensegrityAdapter
from .factors import require_gtsam
from .measurements import compute_cable_lengths
from .runner import run_from_rollout


def _print_stats(results):
    cable_rmse = results["cable_rmse"]
    innovations = results["innovation_norms"]
    pos_rmse = results["position_rmse"]
    sources = results["measurement_sources"]

    print(f"GTSAM KF: enabled ({require_gtsam().__name__})")
    print(f"Measurement source: {sources[0] if sources else 'unknown'}")
    if cable_rmse.size:
        print(f"Cable-length RMSE: mean={cable_rmse.mean():.6f}, std={cable_rmse.std():.6f}")
    if innovations.size:
        print(f"Innovation norm: mean={innovations.mean():.6f}, std={innovations.std():.6f}")
    if pos_rmse.size:
        print(
            "Position RMSE (centroid-aligned, gauge-free): "
            f"mean={pos_rmse.mean():.6f}, std={pos_rmse.std():.6f}"
        )


def _load_training_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {path}")
    with path.open("r") as handle:
        return json.load(handle)


def _select_data_dir(config, override_path=None):
    if override_path is not None:
        return Path(override_path)
    for key in ("val_data_paths", "train_data_paths"):
        if key in config and config[key]:
            return Path(config[key][0])
    raise RuntimeError(
        "No data directories found in training config "
        "(expected val_data_paths or train_data_paths)."
    )


def _load_edges_from_sim_config(sim_config_path, num_nodes=None):
    sim_config_path = Path(sim_config_path)
    if not sim_config_path.exists():
        raise FileNotFoundError(f"sim_config not found: {sim_config_path}")
    with sim_config_path.open("r") as handle:
        sim_config = json.load(handle)

    topology = sim_config["tensegrity_cfg"]["system_topology"]
    site_names = list(topology["sites"].keys())
    site_index = {name: idx for idx, name in enumerate(site_names)}

    edges = []
    for cable in sim_config["tensegrity_cfg"]["cables"]:
        end_pts = cable["end_pts"]
        edges.append([site_index[end_pts[0]], site_index[end_pts[1]]])

    edges = np.asarray(edges, dtype=int)
    if num_nodes is not None and edges.max() >= num_nodes:
        raise ValueError(
            f"edges contain node indices outside [0, {num_nodes - 1}]"
        )
    return edges


def _load_positions_from_processed(data_path):
    processed_path = Path(data_path) / "processed_data.json"
    if not processed_path.exists():
        raise FileNotFoundError(f"processed_data.json not found in {data_path}")
    with processed_path.open("r") as handle:
        data = json.load(handle)

    positions = []
    for entry in data:
        if "end_pts" in entry:
            pos = np.asarray(entry["end_pts"], dtype=float).reshape(-1, 3)
        elif "node_pos" in entry:
            pos = np.asarray(entry["node_pos"], dtype=float).reshape(-1, 3)
        elif "positions" in entry:
            pos = np.asarray(entry["positions"], dtype=float).reshape(-1, 3)
        else:
            raise RuntimeError(
                "processed_data.json does not contain end_pts or node positions."
            )
        positions.append(pos)
    return np.asarray(positions)


def _build_rollout_from_data_dir(train_config, data_dir):
    positions = _load_positions_from_processed(data_dir)
    edges = _load_edges_from_sim_config(
        train_config["sim_config"], num_nodes=positions.shape[1]
    )
    lengths = np.stack(
        [compute_cable_lengths(p, edges) for p in positions], axis=0
    )
    return {
        "positions": positions,
        "edges": edges,
        "lengths": lengths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a GTSAM-based KF on tensegrity rollouts."
    )
    parser.add_argument("--dt", type=float, required=True, help="Timestep size")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of KF steps")
    parser.add_argument("--rollout_path", type=str, default=None, help="Path to rollout (.npz/.pt)")
    parser.add_argument(
        "--real_measurements_path",
        type=str,
        default=None,
        help="Optional cable-length measurements (.npz/.pt/.json)",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="training/configs/3_bar_train_config.json",
        help="GNN training config used to locate data directories",
    )
    parser.add_argument(
        "--data_dir_path",
        type=str,
        default=None,
        help="Override data directory from training config",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Alias for --train_config (kept for compatibility)",
    )
    parser.add_argument("--measurement_noise_sigma", type=float, default=0.01)
    parser.add_argument("--motion_noise_sigma", type=float, default=0.1)
    parser.add_argument("--initial_covariance", type=float, default=1.0)
    args = parser.parse_args()

    require_gtsam()

    rollout = None
    if args.rollout_path:
        rollout = TensegrityAdapter.load_rollout(args.rollout_path)
    else:
        train_config_path = args.config or args.train_config
        train_config = _load_training_config(train_config_path)
        data_dir = _select_data_dir(train_config, args.data_dir_path)
        rollout = _build_rollout_from_data_dir(train_config, data_dir)

    real_measurements = None
    if args.real_measurements_path:
        real_measurements = TensegrityAdapter.load_measurements(
            args.real_measurements_path
        )
        if isinstance(real_measurements, dict):
            lengths = real_measurements.get("lengths", None)
            cable_ids = real_measurements.get("cable_ids", None)
            if cable_ids is not None:
                real_measurements = {"lengths": lengths, "cable_ids": cable_ids}
            else:
                real_measurements = lengths

    if isinstance(real_measurements, dict):
        lengths = real_measurements.get("lengths", None)
        cable_ids = real_measurements.get("cable_ids", None)
        rollout = dict(rollout)
        rollout["lengths"] = lengths
        if cable_ids is not None:
            rollout["cable_ids"] = cable_ids
        real_measurements = None

    results = run_from_rollout(
        rollout=rollout,
        dt=args.dt,
        real_measurements=real_measurements,
        measurement_noise_sigma=args.measurement_noise_sigma,
        motion_noise_sigma=args.motion_noise_sigma,
        initial_covariance=args.initial_covariance,
        num_steps=args.num_steps,
    )
    _print_stats(results)


if __name__ == "__main__":
    main()
