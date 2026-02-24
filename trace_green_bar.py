"""
Diagnostic script to trace where the green bar (rod index 1) is updated in the EKF pipeline.
This helps identify if there's a rod ordering mismatch or other issues specific to rod 1.
"""

import numpy as np
import torch
from pathlib import Path
import json

def trace_rod_ordering(simulator, gt_data, extra_gt_data):
    """
    Trace rod ordering through the pipeline to verify consistency.
    """
    print("=" * 80)
    print("TRACING ROD ORDERING")
    print("=" * 80)
    
    # 1. Check robot.rods order
    print("\n1. Robot.rods order (from config):")
    for i, (name, rod) in enumerate(simulator.robot.rods.items()):
        print(f"   Index {i}: {name} (sites: {rod.sites})")
    
    # 2. Check how initial state is built from end_pts
    print("\n2. Initial state from end_pts:")
    end_pts = torch.tensor(gt_data[0]['end_pts'], dtype=simulator.dtype, device=simulator.device)
    print(f"   end_pts shape: {end_pts.shape}")
    print(f"   end_pts (first 6 points): {end_pts[:6].flatten()}")
    
    pos = (end_pts[1::2] + end_pts[::2]) / 2
    prin = end_pts[1::2] - end_pts[::2]
    prin = prin / prin.norm(dim=1, keepdim=True)
    
    print(f"   Computed positions (COM):")
    for i in range(len(pos)):
        print(f"     Rod {i}: {pos[i].flatten().cpu().numpy()}")
    
    # 3. Check gt_data['pos'] and gt_data['quat'] order
    print("\n3. GT data pos/quat order:")
    gt_pos = np.array(gt_data[0]['pos']).reshape(-1, 3)
    gt_quat = np.array(gt_data[0]['quat']).reshape(-1, 4)
    print(f"   gt['pos'] shape: {gt_pos.shape}")
    print(f"   gt['quat'] shape: {gt_quat.shape}")
    
    for i in range(len(gt_pos)):
        print(f"     Rod {i}: pos={gt_pos[i]}, quat={gt_quat[i]}")
    
    # 4. Compare initial state vs gt_data
    print("\n4. Comparing initial state (from end_pts) vs gt_data:")
    for i in range(len(pos)):
        pos_from_endpts = pos[i].flatten().cpu().numpy()
        pos_from_gt = gt_pos[i]
        diff = np.linalg.norm(pos_from_endpts - pos_from_gt)
        print(f"   Rod {i}: ||pos_endpts - pos_gt|| = {diff:.6f}")
        if diff > 0.01:
            print(f"     WARNING: Large difference! pos_endpts={pos_from_endpts}, pos_gt={pos_from_gt}")
    
    # 5. Check measurement construction
    print("\n5. Measurement vector construction (for rod 1/green):")
    if len(gt_data) > 1:
        gt = gt_data[1]
        pos_meas = np.array(gt['pos'], dtype=np.float64).reshape(-1, 3)
        quat_meas = np.array(gt['quat'], dtype=np.float64).reshape(-1, 4)
        z_np = np.hstack([pos_meas, quat_meas]).reshape(-1)
        
        print(f"   Measurement dim: {len(z_np)} (should be 7*n_rods = {7*len(pos_meas)})")
        print(f"   Rod 1 (green) in measurement:")
        print(f"     pos indices [7:10]: {z_np[7:10]}")
        print(f"     quat indices [10:14]: {z_np[10:14]}")
        print(f"     Full rod 1 block [7:14]: {z_np[7:14]}")
    
    # 6. Check state vector structure
    print("\n6. State vector structure (for rod 1/green):")
    linvel = torch.tensor(gt_data[0]['linvel'], dtype=simulator.dtype, device=simulator.device)
    angvel = torch.tensor(gt_data[0]['angvel'], dtype=simulator.dtype, device=simulator.device)
    
    start_state = torch.hstack([
        pos.reshape(-1, 3, 1), 
        torch_quaternion.compute_quat_btwn_z_and_vec(prin.unsqueeze(-1)).reshape(-1, 4, 1),
        linvel.reshape(-1, 3, 1), 
        angvel.reshape(-1, 3, 1),
    ]).reshape(1, -1, 1)
    
    state_flat = start_state.flatten().cpu().numpy()
    print(f"   State dim: {len(state_flat)} (should be 13*n_rods = {13*len(pos)})")
    print(f"   Rod 1 (green) in state vector:")
    print(f"     pos indices [13:16]: {state_flat[13:16]}")
    print(f"     quat indices [16:20]: {state_flat[16:20]}")
    print(f"     linvel indices [20:23]: {state_flat[20:23]}")
    print(f"     angvel indices [23:26]: {state_flat[23:26]}")
    print(f"     Full rod 1 block [13:26]: {state_flat[13:26]}")
    
    # 7. Check observation matrix H
    print("\n7. Observation matrix H (for rod 1/green, pose-only):")
    n_rods = len(pos)
    meas_dim = 7 * n_rods
    state_dim = 13 * n_rods
    H = np.zeros((meas_dim, state_dim), dtype=np.float64)
    for i in range(meas_dim):
        H[i, i] = 1.0
    
    print(f"   H shape: {H.shape}")
    print(f"   Rod 1 measurement row indices [7:14] map to state indices:")
    for i in range(7, 14):
        state_idx = np.where(H[i, :] != 0)[0]
        print(f"     meas[{i}] -> state[{state_idx[0]}]")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("Rod ordering should be consistent:")
    print("  - robot.rods: rod_01 (index 0), rod_23 (index 1), rod_45 (index 2)")
    print("  - State vector: [rod0 (0:13), rod1 (13:26), rod2 (26:39)]")
    print("  - Measurement: [rod0 (0:7), rod1 (7:14), rod2 (14:21)]")
    print("  - sim_data_publisher: [red (0:13), green (13:26), blue (26:39)]")
    print("\nIf green bar is wrong, check:")
    print("  1. Is gt_data['end_pts'] ordered the same as gt_data['pos']/['quat']?")
    print("  2. Is rod index 1 in state vector the same physical rod as rod index 1 in measurements?")
    print("  3. Are there any special handling issues for rod index 1 in quaternion alignment?")


if __name__ == '__main__':
    import sys
    from utilities import torch_quaternion
    
    # Load simulator and data
    model_path = Path("sample_model.pt")
    data_dir_path = Path("../tensegrity/data_sets/mjc_synthetic_5d_0.01/val/R2S2Rrolling_7/")
    
    if not model_path.exists():
        print(f"Error: {model_path} not found")
        sys.exit(1)
    
    simulator = torch.load(model_path, map_location='cpu')
    simulator.eval()
    simulator.to('cpu')
    
    gt_data_json = json.load((data_dir_path / "processed_data.json").open('r'))
    extra_data_json = json.load((data_dir_path / "5d_extra_state_data.json").open('r'))
    
    trace_rod_ordering(simulator, gt_data_json, extra_data_json)
