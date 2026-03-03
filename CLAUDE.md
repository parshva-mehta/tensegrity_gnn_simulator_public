# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for the paper "Learning Differentiable Tensegrity Dynamics with Graph Neural Networks" ([arXiv:2410.12216](https://arxiv.org/abs/2410.12216)). Implements differentiable physics simulation for tensegrity robots via an Encode-Process-Decode GNN, with an Extended Kalman Filter (EKF) for state estimation.

## Setup

```bash
conda create --name tensegrity_gnn python=3.10
conda activate tensegrity_gnn
pip install -r requirements.txt
```

Dataset must be downloaded separately (Google Drive link in README.md) and placed at `../tensegrity/data_sets/`.

## Common Commands

```bash
# Train model (3-bar config by default)
python3 train.py

# Evaluate model (set model/data paths in eval.py __main__ block)
python3 eval.py
```

There are no test files in this codebase.

## Architecture

### Simulator Hierarchy

There are two parallel simulator families that share `AbstractSimulator` as a base:

1. **Physics-based** (`simulators/tensegrity_simulator.py`):
   `TensegrityRobotSimulator` → implements the full rigid-body loop: forces → torques → accelerations → time integration → contact resolution.

2. **GNN-based** (`simulators/tensegrity_gnn_simulator.py`):
   - `TensegrityGNNSimulator`: pure learned simulator; maps state → graph → GNN → next state.
   - `TensegrityHybridGNNSimulator`: extends the physics simulator; uses first-principles for passive forces then calls the GNN in `resolve_contacts` to predict contact corrections.

`AbstractSimulator.step()` defines the fixed call sequence: `update_state → apply_control → compute_forces → compute_torques → compute_accelerations → time_integration → compute_contact_deltas → resolve_contacts`.

### State Representation

Each rod's state is a 13-dimensional vector: `[pos(3), quat(4), linvel(3), angvel(3)]`. A full robot state is the horizontal concatenation of all rod states, shaped `(batch, 13 * n_rods, 1)`.

### GNN Pipeline (`gnn_physics/`)

1. **Data processor** (`gnn_physics/data_processors/`): converts batched rod states to `torch_geometric.data.Data` graphs. The abstract base (`abstract_tensegrity_data_processor.py`) handles batching, normalization (via `AccumulatedNormalizer`), and the `pose2node` / `node2pose` transforms. The concrete `BatchTensegrityDataProcessor` fills in `_compute_node_feats`, `_compute_edge_feats`, and `node2pose`.

2. **GNN model** (`gnn_physics/gnn.py`): `EncodeProcessDecode` with separate MLPs per node/edge type in the encoder, stacked `InteractionNetwork` layers (message passing via `BaseInteractionNetwork`) in the processor, and a node-level MLP decoder that outputs `dv` (velocity corrections).

### Robot Model (`robots/tensegrity.py`)

`TensegrityRobot` is constructed from a config JSON specifying rods, cables, and system topology (sites + connectivity). `TensegrityRobotGNN` extends it with mesh geometry needed for graph construction.

### EKF (`ekf.py`)

Implements an Extended Kalman Filter using GTSAM. The linearization (`linearize_gnn`) supports two modes:
- **GNN Jacobian**: uses `torch.func.jacrev` on the GNN forward pass (default).
- **Finite differences**: central differences over `simulator.step` (enabled via `use_finite_diff=True`).

`run_ekf_rollout` is the main entry point; called from `eval.py` when `use_ekf=True`.

### Config Files

- `simulators/configs/*.json`: robot geometry, cable parameters, GNN hyperparameters, contact parameters. The `tensegrity_cfg` key defines rods and cables; top-level keys (`latent_dim`, `nmessage_passing_steps`, etc.) define the GNN.
- `training/configs/*.json`: training data paths, batch size, optimizer params, `num_steps_fwd` (multi-step rollout during training), output path.

## Key Implementation Details

- **Default precision**: `DEFAULT_DTYPE = torch.float32` in `utilities/misc_utils.py`. Switch to `float64` there for higher accuracy/stability.
- **Device handling**: all objects inherit `BaseStateObject` (`state_objects/base_state_object.py`), which tracks `.device` and `.dtype`. Call `.to(device)` on the simulator to move everything.
- **Saved model format**: models are saved as full simulator objects via `torch.save`. Load with `torch.load(path, map_location=device)`.
- **Eval entry point**: the `if __name__ == '__main__'` block in `eval.py` hard-codes `model_path`, `data_dir_path`, and output paths — edit these before running.
- **Training schedule**: `train.py` runs a curriculum over increasing `num_steps_fwd` values with decreasing learning rates across phases.
