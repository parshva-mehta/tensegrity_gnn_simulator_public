"""Extended Kalman Filter with linearized dynamics for tensegrity simulation.

Supports linearization via finite differences or GNN-based Jacobians. Uses GTSAM
for predict/update steps. State is per-rod (13 dims: 3 pos, 4 quat, 3 linvel, 3 angvel).
"""

import numpy as np
import torch
import tqdm
from torch.func import jacrev

import gtsam

from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE


def _renormalize_quat_block_in_flat(x_flat: torch.Tensor, rod_index: int) -> None:
    """In-place renormalize one rod's quaternion block in flattened state.

    Assumes 13 dimensions per rod: pos(3), quat(4), linvel(3), angvel(3).
    Only the quat slice [13*rod_index+3 : 13*rod_index+7] is modified.

    Args:
        x_flat: Flattened state tensor, modified in place.
        rod_index: Zero-based rod index (which 13-dim block to renormalize).
    """
    q_start = 13 * rod_index + 3
    q_end = 13 * rod_index + 7
    q = x_flat[q_start:q_end]
    n = q.norm()
    if n > 1e-10:
        x_flat[q_start:q_end] = q / n


def _linearize_step_finite_diff(
    simulator,
    state_0,
    dt,
    control_signals,
    sample_index,
    n_rods,
    eps_pos=1e-5,
    eps_quat=1e-5,
    eps_vel=1e-4,
):
    """Linearize simulator.step via central finite differences.

    Computes Jacobian F such that next_state ≈ next_state_0 + F @ delta_state.
    Uses central differences with per-block epsilons; restores simulator state
    after perturbations so the Jacobian is consistent. Expects batch size 1.

    Args:
        simulator: Simulator with step(state, dt, control_signals=...) returning
            (next_state,) or next_state.
        state_0: Current state tensor, shape (1, state_dim, 1).
        dt: Time step (scalar).
        control_signals: Control inputs for the step.
        sample_index: Batch index to linearize (must be 0 for batch size 1).
        n_rods: Number of rods (state_dim = 13 * n_rods).
        eps_pos: Perturbation for position components.
        eps_quat: Perturbation for quaternion components.
        eps_vel: Perturbation for velocity components.

    Returns:
        next_state_0: Predicted state at sample_index after one step (detached).
        J: Jacobian of shape (state_dim, state_dim), dtype/device match state_0.

    Raises:
        ValueError: If state_0 batch size is not 1.
    """
    device = state_0.device
    dtype = state_0.dtype
    B, state_dim, _ = state_0.shape
    if B != 1:
        raise ValueError("_linearize_step_finite_diff expects state_0 with batch size 1 (EKF single sample)")
    # Finite-diff linearization step 1: assign per-state perturbation scales.
    eps_vec = torch.zeros(state_dim, dtype=dtype, device=device)
    for r in range(n_rods):
        base = 13 * r
        eps_vec[base : base + 3] = eps_pos
        eps_vec[base + 3 : base + 7] = eps_quat
        eps_vec[base + 7 : base + 10] = eps_vel
        eps_vec[base + 10 : base + 13] = eps_vel

    # Finite-diff linearization step 2: evaluate nominal next state at x_k. 
    with torch.no_grad():
        step_out = simulator.step(state_0, dt, control_signals=control_signals)
        full_next = (step_out[0] if isinstance(step_out, tuple) else step_out).detach()
    next_state_0 = full_next[sample_index : sample_index + 1].clone()
    J = torch.zeros(state_dim, state_dim, dtype=dtype, device=device)
    x_flat = state_0[sample_index].reshape(-1).clone()

    # Finite-diff linearization step 3: perturb each state dimension and estimate one Jacobian column.
    for j in range(state_dim):
        eps_j = eps_vec[j].item()
        if eps_j <= 0:
            continue
        e = torch.zeros_like(x_flat)
        e[j] = 1.0
        x_plus = x_flat + eps_j * e
        x_minus = x_flat - eps_j * e
        rod_j = j // 13
        if 13 * rod_j + 3 <= j < 13 * rod_j + 7:
            _renormalize_quat_block_in_flat(x_plus, rod_j)
            _renormalize_quat_block_in_flat(x_minus, rod_j)
        state_plus = state_0.clone()
        state_plus[sample_index] = x_plus.reshape(state_plus[sample_index].shape)
        state_minus = state_0.clone()
        state_minus[sample_index] = x_minus.reshape(state_minus[sample_index].shape)
        simulator.update_state(state_0)
        with torch.no_grad():
            out_plus = simulator.step(state_plus, dt, control_signals=control_signals)
        simulator.update_state(state_0)
        with torch.no_grad():
            out_minus = simulator.step(state_minus, dt, control_signals=control_signals)
        y_plus_full = out_plus[0] if isinstance(out_plus, tuple) else out_plus
        y_minus_full = out_minus[0] if isinstance(out_minus, tuple) else out_minus
        y_plus = y_plus_full[sample_index].reshape(-1).detach()
        y_minus = y_minus_full[sample_index].reshape(-1).detach()
        # Central-difference column: df/dx_j ≈ (f(x+eps e_j) - f(x-eps e_j)) / (2 eps).
        J[:, j] = (y_plus - y_minus) / (2.0 * eps_j)
    with torch.no_grad():
        simulator.update_state(state_0)
    return next_state_0, J


def linearize_gnn(simulator, state, dt, control_signals=None, sample_index=0, use_finite_diff=False):
    """Linearize the dynamics and return nominal next state and Jacobian.

    Produces (next_state_0, J) such that next_state ≈ next_state_0 + J @ delta_state.
    If use_finite_diff is True, linearizes the full simulator.step via finite
    differences; otherwise uses the GNN (and compute_jacobian if available, else
    jacrev on the GNN forward).

    Args:
        simulator: Simulator (GNN or wrapper); may have gnn_sim and compute_jacobian.
        state: Current state, shape (B, state_dim, 1) or (B, state_dim) or (state_dim,).
        dt: Time step.
        control_signals: Optional control inputs (format depends on simulator).
        sample_index: Batch index for which to compute next_state_0 and J.
        use_finite_diff: If True, use finite-diff linearization; else GNN-based.

    Returns:
        next_state_0: Predicted state at sample_index, shape (1, state_dim, 1).
        J: Jacobian (state_dim, state_dim) for the linearized update.

    Raises:
        IndexError: If sample_index is out of range for the state batch size.
    """
    # Linearization step A: canonicalize state shape to (B, D, 1) for a consistent Jacobian interface.
    if state.dim() == 1:
        state = state.unsqueeze(0).unsqueeze(-1)
    elif state.dim() == 2:
        state = state.unsqueeze(-1)
    state_0 = state.detach().clone()
    B, state_dim, _ = state_0.shape
    if not (0 <= sample_index < B):
        raise IndexError(f"sample_index {sample_index} out of range for batch size {B}")

    # Linearization step B (optional): use central finite differences around x_k.
    if use_finite_diff:
        n_rods = state_dim // 13
        return _linearize_step_finite_diff(
            simulator, state_0, dt, control_signals, sample_index, n_rods
        )

    # Linearization step C: get the nominal next state f(x_k, u_k) at the current linearization point.
    gnn_sim = getattr(simulator, 'gnn_sim', simulator)
    with torch.no_grad():
        # Use process_gnn -> node2pose directly so nominal state and Jacobian use the same GNN map.
        # This avoids calling gnn_sim.step(), which may require simulator internals (e.g., rigid_body).
        data_processor = gnn_sim.data_processor
        robot = data_processor.robot
        graph = gnn_sim.process_gnn(state_0)
        body_mask = graph.body_mask.flatten()
        full_next = data_processor.node2pose(
            graph.p_node_pos[body_mask],
            graph.node_pos[body_mask],
            robot.num_nodes_per_rod,
        )
        next_state_0 = full_next[sample_index : sample_index + 1].clone()

    # Linearization step D: prefer simulator-provided Jacobian routine (usually NN/autodiff based).
    if hasattr(simulator, 'compute_jacobian'):
        J = simulator.compute_jacobian(curr_state=state_0, dt=dt, sample_index=sample_index)
        return next_state_0, J

    # Fallback path
    data_processor = gnn_sim.data_processor
    robot = data_processor.robot
    x0 = state_0[sample_index:sample_index + 1].detach().clone().requires_grad_(True)

    def gnn_next_state_flat(x):
        graph = gnn_sim.process_gnn(x)
        body_mask = graph.body_mask.flatten()
        next_state = data_processor.node2pose(
            graph.p_node_pos[body_mask],
            graph.node_pos[body_mask],
            robot.num_nodes_per_rod,
        )
        return next_state.reshape(-1)

    # Linearization step F: evaluate J_x = d f / d x at x_k, then reshape to (state_dim, state_dim).
    J = jacrev(gnn_next_state_flat)(x0)
    J = J.reshape(-1, state_dim)
    return next_state_0, J


def _renormalize_quats_numpy(mean: np.ndarray, n_rods: int) -> None:
    """In-place renormalize quaternion blocks in the state vector.

    State layout: 13 dims per rod; quat occupies indices [13*r+3 : 13*r+7].
    Each quat block is replaced with q / ||q|| when ||q|| > 1e-10.

    Args:
        mean: State mean vector, modified in place.
        n_rods: Number of rods.
    """
    for r in range(n_rods):
        q_start = 13 * r + 3
        q_end = 13 * r + 7
        q = mean[q_start:q_end]
        n = np.linalg.norm(q)
        if n > 1e-10:
            mean[q_start:q_end] = q / n


def _reinit_state_jitter(kf, state, state_dim, jitter=1e-8):
    """Reinitialize Kalman filter state with symmetrized covariance and jitter.

    Replaces P with (P + P.T)/2 + jitter*I and re-inits the filter. Used to avoid
    GTSAM IndeterminantLinearSystemException from numerical asymmetry or singularity.

    Args:
        kf: GTSAM KalmanFilter instance.
        state: Current GTSAM state (GaussianConditional or similar).
        state_dim: State dimension.
        jitter: Diagonal regularization added to covariance (default 1e-8).

    Returns:
        New state from kf.init(mean, P_symmetrized_and_jittered).
    """
    mean_col = np.asarray(state.mean(), dtype=np.float64).reshape(state_dim, 1)
    P = np.asarray(state.covariance(), dtype=np.float64)
    P_sym = 0.5 * (P + P.T)
    P_jittered = P_sym + jitter * np.eye(state_dim, dtype=np.float64)
    return kf.init(mean_col, P_jittered)


def _control_to_numpy_vector(ctrl):
    """Convert control input to a flat float64 numpy vector."""
    if ctrl is None:
        return None
    if isinstance(ctrl, torch.Tensor):
        return ctrl.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if isinstance(ctrl, np.ndarray):
        return ctrl.reshape(-1).astype(np.float64)
    if isinstance(ctrl, (list, tuple)):
        vals = []
        for c in ctrl:
            if isinstance(c, torch.Tensor):
                vals.extend(c.detach().cpu().numpy().reshape(-1).astype(np.float64).tolist())
            else:
                vals.append(float(c))
        return np.asarray(vals, dtype=np.float64)
    return np.asarray([float(ctrl)], dtype=np.float64)


def _get_simulator_control_jacobian(simulator, state_torch, dt, ctrl, sample_index=0):
    """Query simulator (or gnn_sim) for control Jacobian d f / d u."""
    candidates = [simulator, getattr(simulator, "gnn_sim", None)]
    for cand in candidates:
        if cand is None or not hasattr(cand, "compute_control_jacobian"):
            continue
        fn = getattr(cand, "compute_control_jacobian")
        try:
            J_u = fn(
                curr_state=state_torch,
                dt=dt,
                control_signals=ctrl,
                sample_index=sample_index,
            )
        except TypeError:
            try:
                J_u = fn(state_torch, dt, ctrl, sample_index)
            except TypeError:
                J_u = fn(state_torch, dt, ctrl)
        if J_u is None:
            continue
        if torch.is_tensor(J_u):
            J_u = J_u.detach().cpu().numpy()
        J_u = np.asarray(J_u, dtype=np.float64)
        if J_u.ndim != 2:
            raise ValueError(f"compute_control_jacobian must return 2D, got shape {J_u.shape}")
        return J_u
    return None


def _ekf_step_gtsam(kf, state_gtsam, simulator, state_torch, dt, ctrl,
                    H_np, z_np, Q_sigmas, R_sigmas, n_rods, have_measurement,
                    use_finite_diff, innovation_gate_sigma=np.inf,
                    control_jacobian_mode="identity",
                    require_control_jacobian=False):
    """Perform one EKF predict and optionally update step using GTSAM.

    Predicts via linearized dynamics (finite-diff or GNN), then if have_measurement
    updates with measurement z_np and observation matrix H_np. Applies optional
    innovation gating (reject update if innovation norm > innovation_gate_sigma *
    sqrt(meas_dim)). Output state mean is renormalized for quaternions.

    Args:
        kf: GTSAM KalmanFilter instance.
        state_gtsam: Current GTSAM filter state.
        simulator: Simulator for linearize_gnn (step + optional Jacobian).
        state_torch: Current state as torch tensor (1, state_dim, 1).
        dt: Time step.
        ctrl: Control signals for the step.
        H_np: Observation matrix (meas_dim, state_dim).
        z_np: Measurement vector (used only if have_measurement).
        Q_sigmas: Diagonal process noise standard deviations.
        R_sigmas: Diagonal measurement noise standard deviations.
        n_rods: Number of rods.
        have_measurement: If True, perform update with z_np; else predict only.
        use_finite_diff: Passed to linearize_gnn for Jacobian method.
        innovation_gate_sigma: If finite, reject update when innovation too large.

    Returns:
        mean_for_output: State mean after predict/update, with quats renormalized.
        state_gtsam: New GTSAM state (predicted or posterior).
    """
    state_dim = state_torch.numel()
    x_mean = np.array(state_gtsam.mean()).reshape(-1).astype(np.float64)
    # [TENSEGRITY_EKF 1.1] Obtain Jacobians J_x0^f, J_u^f from NN model
    # NOTE: current linearize_gnn returns state Jacobian F (J_x0^f); control Jacobian J_u^f is not explicitly exposed here.
    next_state_0, F = linearize_gnn(
        simulator, state_torch, dt, control_signals=ctrl,
        sample_index=0, use_finite_diff=use_finite_diff
    )
    next_state_0_np = next_state_0.detach().cpu().numpy().reshape(-1).astype(np.float64)
    F_np = F.detach().cpu().numpy().astype(np.float64)

    if not np.all(np.isfinite(F_np)):
        F_np = np.eye(state_dim, dtype=np.float64)

    condF = np.linalg.cond(F_np)
    if not np.isfinite(condF) or condF > 1e8:
        F_np = np.eye(state_dim, dtype=np.float64)
        Q_sigmas_safe = np.maximum(Q_sigmas, 1e-6) * 10.0
    else:
        Q_sigmas_safe = np.maximum(Q_sigmas, 1e-6)

    # [TENSEGRITY_EKF 1.2] Use NN Jacobians to compute J_x, J_o, J_u
    # J_x uses F_np. J_u is simulator-provided when available; otherwise fallback.
    F_cont = np.ascontiguousarray(F_np, dtype=np.float64)
    u_np = _control_to_numpy_vector(ctrl)
    J_u = None
    if control_jacobian_mode == "simulator":
        J_u = _get_simulator_control_jacobian(simulator, state_torch, ctrl=ctrl, dt=dt, sample_index=0)
        if J_u is None and require_control_jacobian:
            raise RuntimeError(
                "control_jacobian_mode='simulator' requested but simulator.compute_control_jacobian is unavailable."
            )
    if J_u is not None and u_np is not None and J_u.shape[1] == u_np.size:
        J_u = np.ascontiguousarray(J_u, dtype=np.float64)
        # [TENSEGRITY_EKF 1.3] Compute prediction error / residual (affine remainder)
        b_aff = np.asarray(next_state_0_np - F_np @ x_mean - J_u @ u_np, dtype=np.float64).reshape(state_dim, 1)
        B_cont = np.hstack([J_u, np.eye(state_dim, dtype=np.float64)])
        u_eff = np.concatenate([u_np.reshape(-1), b_aff.reshape(-1)]).reshape(-1, 1)
    else:
        B_cont = np.ascontiguousarray(np.eye(state_dim, dtype=np.float64))
        b_aff = np.asarray(next_state_0_np - F_np @ x_mean, dtype=np.float64).reshape(state_dim, 1)
        u_eff = b_aff
    model_q = gtsam.noiseModel.Diagonal.Sigmas(Q_sigmas_safe)
    # [TENSEGRITY_EKF 1.4] Build JacobianFactor from Jacobians and b = error
    # [TENSEGRITY_EKF 1.5] Eliminate factor graph into BayesNet / solve linear system
    # These steps are executed internally by GTSAM via KalmanFilter.predict(...).
    try:
        state_pred = kf.predict(state_gtsam, F_cont, B_cont, u_eff, model_q)
    except RuntimeError:
        F_cont = np.ascontiguousarray(np.eye(state_dim, dtype=np.float64))
        B_cont = np.ascontiguousarray(np.eye(state_dim, dtype=np.float64))
        u_eff = np.zeros((state_dim, 1), dtype=np.float64)
        state_pred = kf.predict(state_gtsam, F_cont, B_cont, u_eff, model_q)
    state_pred = _reinit_state_jitter(kf, state_pred, state_dim)
    # [TENSEGRITY_EKF 1.6] Recover posterior estimate x_hat_{k+1}^-
    if not have_measurement:
        mean_np = np.array(state_pred.mean()).reshape(-1)
        return mean_np, state_pred
    mean_pred = np.array(state_pred.mean()).reshape(-1).copy()
    observe_pose_only = (z_np.size == 7 * n_rods)
    mean_pred_col = mean_pred.reshape(state_dim, 1)
    P_pred = np.asarray(state_pred.covariance(), dtype=np.float64)
    P_sym = 0.5 * (P_pred + P_pred.T) + 1e-8 * np.eye(state_dim, dtype=np.float64)
    state_pred = kf.init(mean_pred_col, P_sym)
    # [TENSEGRITY_EKF 2] Optional measurement update: error(x_hat_{k+1}^-) = x_hat_{k+1}^- ⊖ x_{k+1}^{true}
    innovation = z_np.reshape(-1) - (H_np @ mean_pred)
    if np.isfinite(innovation_gate_sigma) and np.linalg.norm(innovation) > innovation_gate_sigma * np.sqrt(innovation.size):
        mean_for_output = mean_pred.copy()
        _renormalize_quats_numpy(mean_for_output, n_rods)
        return mean_for_output, state_pred
    H_np_cont = np.ascontiguousarray(np.asarray(H_np, dtype=np.float64))
    meas_dim = z_np.size
    z_col = np.asarray(z_np, dtype=np.float64).reshape(meas_dim, 1)
    R_sigmas_safe = np.maximum(R_sigmas, 1e-6)
    model_r = gtsam.noiseModel.Diagonal.Sigmas(R_sigmas_safe)
    state_post = kf.update(state_pred, H_np_cont, z_col, model_r)
    state_post = _reinit_state_jitter(kf, state_post, state_dim)
    mean_np = np.array(state_post.mean()).reshape(-1).copy()
    mean_for_output = mean_np.copy()
    _renormalize_quats_numpy(mean_for_output, n_rods)
    return mean_for_output, state_post


def _ensure_ctrl_for_step(ctrl, simulator):
    """Convert control input to simulator-compatible format and device/dtype.

    Accepts list of scalars, list of tensors, numpy array, or torch tensor;
    returns the same in the format and device/dtype expected by the simulator.

    Args:
        ctrl: Control input (list, tuple, np.ndarray, or torch.Tensor); may be None.
        simulator: Simulator object with .device and .dtype (or defaults used).

    Returns:
        Control in simulator format (None, list, or tensor on correct device/dtype).
    """
    if ctrl is None:
        return None
    dtype = getattr(simulator, 'dtype', DEFAULT_DTYPE)
    device = getattr(simulator, 'device', 'cpu')
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if isinstance(ctrl, (list, tuple)) and len(ctrl) > 0:
        if isinstance(ctrl[0], (int, float)):
            return [float(c) for c in ctrl]
        if isinstance(ctrl[0], torch.Tensor):
            return [c.to(device=device, dtype=dtype) for c in ctrl]
    if isinstance(ctrl, np.ndarray):
        return torch.from_numpy(ctrl).to(device=device, dtype=dtype)
    if isinstance(ctrl, torch.Tensor):
        return ctrl.to(device=device, dtype=dtype)
    return ctrl


def _structured_Q_sigmas(state_dim, n_rods, base_sigma, quat_inflation=2.0, vel_inflation=2.0):
    """Build diagonal process noise standard deviations per state block.

    Fills state_dim entries with base_sigma, then multiplies quat (4 dims) and
    velocity (6 dims) blocks per rod by quat_inflation and vel_inflation.

    Args:
        state_dim: Total state dimension (13 * n_rods).
        n_rods: Number of rods.
        base_sigma: Base sigma for position (and default) components.
        quat_inflation: Multiplier for quaternion block sigma (default 2.0).
        vel_inflation: Multiplier for linvel/angvel block sigma (default 2.0).

    Returns:
        sigmas: 1D array of length state_dim, dtype float64.
    """
    sigmas = np.full(state_dim, base_sigma, dtype=np.float64)
    for r in range(n_rods):
        base = 13 * r
        sigmas[base + 3 : base + 7] *= quat_inflation
        sigmas[base + 7 : base + 13] *= vel_inflation
    return sigmas


def _structured_R_sigmas(meas_dim, n_rods, pos_sigma, quat_sigma=None):
    """Build diagonal measurement noise standard deviations.

    For pose-only (meas_dim == 7*n_rods): pos_sigma for position, quat_sigma for
    quat per rod. For full state (meas_dim == 13*n_rods): same for pos/quat and
    pos_sigma for velocity components. quat_sigma defaults to 5*pos_sigma.

    Args:
        meas_dim: Measurement dimension (7*n_rods or 13*n_rods).
        n_rods: Number of rods.
        pos_sigma: Sigma for position (and optionally velocity) components.
        quat_sigma: Sigma for quaternion components; default 5.0 * pos_sigma.

    Returns:
        sigmas: 1D array of length meas_dim, dtype float64.
    """
    if quat_sigma is None:
        quat_sigma = 5.0 * pos_sigma
    sigmas = np.empty(meas_dim, dtype=np.float64)
    if meas_dim == 7 * n_rods:
        for r in range(n_rods):
            sigmas[7 * r : 7 * r + 3] = pos_sigma
            sigmas[7 * r + 3 : 7 * r + 7] = quat_sigma
    else:
        for r in range(n_rods):
            sigmas[13 * r : 13 * r + 3] = pos_sigma
            sigmas[13 * r + 3 : 13 * r + 7] = quat_sigma
            sigmas[13 * r + 7 : 13 * r + 13] = pos_sigma
    return sigmas


class OnlineEKF:
    """Streaming EKF wrapper for step-by-step filtering in a timer loop.

    Wraps the existing batch EKF helpers (_structured_Q_sigmas, _structured_R_sigmas,
    _ekf_step_gtsam) for use in a real-time publisher that receives one observation
    per timer tick.
    """

    def __init__(self, simulator, dt, n_rods,
                 process_noise_scale=1e-4, measurement_noise_scale=1e-3,
                 observe_pose_only=False, use_finite_diff=False,
                 innovation_gate_sigma=np.inf,
                 Q_quat_inflation=2.0, Q_vel_inflation=2.0,
                 control_jacobian_mode="identity",
                 require_control_jacobian=False):
        self.simulator = simulator
        self.dt = dt
        self.n_rods = n_rods
        self.state_dim = 13 * n_rods
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.observe_pose_only = observe_pose_only
        self.use_finite_diff = use_finite_diff
        self.innovation_gate_sigma = innovation_gate_sigma
        self.control_jacobian_mode = control_jacobian_mode
        self.require_control_jacobian = require_control_jacobian

        base_Q_sigma = np.sqrt(float(process_noise_scale))
        self.Q_sigmas = _structured_Q_sigmas(
            self.state_dim, n_rods, base_Q_sigma, Q_quat_inflation, Q_vel_inflation
        )

        pos_sigma = np.sqrt(float(measurement_noise_scale))
        if observe_pose_only:
            meas_dim = 7 * n_rods
            self.H_np = np.zeros((meas_dim, self.state_dim), dtype=np.float64)
            for i in range(meas_dim):
                self.H_np[i, i] = 1.0
        else:
            meas_dim = self.state_dim
            self.H_np = np.eye(self.state_dim, dtype=np.float64)
        self.R_sigmas = _structured_R_sigmas(meas_dim, n_rods, pos_sigma)

        self.kf = gtsam.KalmanFilter(self.state_dim)
        self.state_gtsam = None
        self.state_torch = None

    def initialize(self, start_state: torch.Tensor,
                   rest_lengths=None, motor_speeds=None):
        """Initialize EKF state and optionally configure simulator actuators.

        Args:
            start_state: Initial state tensor, shape (1, state_dim, 1) or compatible.
            rest_lengths: List of cable rest lengths (passed to actuated_cables).
            motor_speeds: List of motor speeds (passed to motor states).
        """
        dtype = getattr(self.simulator, 'dtype', DEFAULT_DTYPE)
        device = getattr(self.simulator, 'device', 'cpu')
        if not isinstance(device, torch.device):
            device = torch.device(device)

        if rest_lengths is not None and motor_speeds is not None:
            cables = list(self.simulator.robot.actuated_cables.values())
            for i, c in enumerate(cables):
                c.actuation_length = c._rest_length - rest_lengths[i]
                c.motor.motor_state.omega_t = torch.tensor(
                    motor_speeds[i], dtype=dtype, device=device
                ).reshape(1, 1, 1)

        start_state = start_state.to(device=device, dtype=dtype)
        if start_state.dim() == 2:
            start_state = start_state.unsqueeze(-1)

        x0_np = start_state.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
        P0_np = float(self.measurement_noise_scale) * np.eye(self.state_dim, dtype=np.float64)
        self.state_gtsam = self.kf.init(x0_np, P0_np)
        self.state_torch = start_state.clone()

    def step(self, z_t: np.ndarray = None, u_t=None, have_measurement=True) -> torch.Tensor:
        """Run one EKF predict+update step.

        Args:
            z_t: Measurement vector (state_dim or pose_dim numpy array).
            u_t: Control input for this step (list, array, or None).

        Returns:
            Filtered state tensor, shape (1, state_dim, 1).
        """
        if have_measurement and z_t is None:
            raise ValueError("z_t must be provided when have_measurement=True")
        dtype = getattr(self.simulator, 'dtype', DEFAULT_DTYPE)
        device = getattr(self.simulator, 'device', 'cpu')
        if not isinstance(device, torch.device):
            device = torch.device(device)

        ctrl_step = _ensure_ctrl_for_step(u_t, self.simulator)
        with torch.no_grad():
            mean_np, self.state_gtsam = _ekf_step_gtsam(
                self.kf, self.state_gtsam, self.simulator,
                self.state_torch, self.dt, ctrl_step, self.H_np, z_t,
                self.Q_sigmas, self.R_sigmas, self.n_rods,
                have_measurement=have_measurement, use_finite_diff=self.use_finite_diff,
                innovation_gate_sigma=self.innovation_gate_sigma,
                control_jacobian_mode=self.control_jacobian_mode,
                require_control_jacobian=self.require_control_jacobian,
            )
        self.state_torch = torch.tensor(
            mean_np, dtype=dtype, device=device
        ).view(1, self.state_dim, 1)
        return self.state_torch


def run_ekf_rollout(simulator,
                    gt_data,
                    extra_gt_data,
                    dt,
                    process_noise_scale=1e-4,
                    measurement_noise_scale=1e-3,
                    observe_pose_only=False,
                    start_state=None,
                    use_finite_diff=False,
                    Q_quat_inflation=2.0,
                    Q_vel_inflation=2.0,
                    innovation_gate_sigma=np.inf,
                    control_jacobian_mode="simulator"):
    """Run an EKF rollout over ground-truth data with predict/update steps.

    Initializes from start_state or from gt_data[0] (endpoints, linvel, angvel).
    For each time step: predicts using linearized dynamics (GNN or finite-diff),
    then updates with the next ground-truth frame as measurement (pose or full
    state). Control signals and cable/motor initialization come from extra_gt_data.

    Args:
        simulator: Tensegrity simulator (GNN or physics) with step() and robot.
        gt_data: List of dicts with 'pos', 'quat', and optionally 'linvel', 'angvel',
            'end_pts'; used as measurements at each step.
        extra_gt_data: List of dicts with 'controls', 'rest_lengths', 'motor_speeds';
            first frame used to set cable rest lengths and motor speeds.
        dt: Time step between frames.
        process_noise_scale: Scale for process noise covariance (sqrt applied to Q).
        measurement_noise_scale: Scale for measurement noise and initial P0.
        observe_pose_only: If True, measurement is 7*n_rods (pos+quat); else full state.
        start_state: Optional initial state tensor; if None, built from gt_data[0].
        use_finite_diff: If True, linearize via finite differences; else GNN Jacobian.
        Q_quat_inflation: Multiplier for quat block in process noise (default 2.0).
        Q_vel_inflation: Multiplier for velocity block in process noise (default 2.0).
        innovation_gate_sigma: If finite, reject update when innovation norm exceeds
            this times sqrt(meas_dim) (default np.inf = no gating).

    Returns:
        frames: List of dicts with keys 'time', 'pose', 'state'. Each 'state' is
            a torch tensor (1, state_dim, 1); 'pose' is flattened (pos, quat) per rod.
    """
    dtype = getattr(simulator, 'dtype', DEFAULT_DTYPE)
    device = getattr(simulator, 'device', 'cpu')
    if not isinstance(device, torch.device):
        device = torch.device(device)

    ctrls = [e['controls'] for e in extra_gt_data]
    init_rest_lengths = extra_gt_data[0]['rest_lengths']
    init_motor_speeds = extra_gt_data[0]['motor_speeds']
    cables = simulator.robot.actuated_cables.values()
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - init_rest_lengths[i]
        c.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i], dtype=dtype, device=device
        ).reshape(1, 1, 1)

    if start_state is None:
        end_pts = torch.tensor(gt_data[0]['end_pts'], dtype=dtype, device=device)
        pos = (end_pts[1::2] + end_pts[::2]) / 2
        prin = end_pts[1::2] - end_pts[::2]
        prin = prin / prin.norm(dim=1, keepdim=True)
        quat = torch_quaternion.compute_quat_btwn_z_and_vec(prin.unsqueeze(-1))
        linvel = torch.tensor(gt_data[0]['linvel'], dtype=dtype, device=device)
        angvel = torch.tensor(gt_data[0]['angvel'], dtype=dtype, device=device)
        start_state = torch.hstack([
            pos.reshape(-1, 3, 1), quat.reshape(-1, 4, 1),
            linvel.reshape(-1, 3, 1), angvel.reshape(-1, 3, 1),
        ]).reshape(1, -1, 1)
    else:
        start_state = start_state.to(device=device, dtype=dtype)
        if start_state.dim() == 2:
            start_state = start_state.unsqueeze(-1)

    state_dim = start_state.numel()
    n_rods = start_state.shape[1] // 13
    pose_dim = 7 * n_rods
    if observe_pose_only:
        meas_dim = pose_dim
    else:
        meas_dim = state_dim

    base_Q_sigma = np.sqrt(float(process_noise_scale))
    Q_sigmas = _structured_Q_sigmas(state_dim, n_rods, base_Q_sigma, Q_quat_inflation, Q_vel_inflation)
    pos_sigma = np.sqrt(float(measurement_noise_scale))
    R_sigmas = _structured_R_sigmas(meas_dim, n_rods, pos_sigma)
    m = state_dim
    x0_np = start_state.detach().cpu().numpy().reshape(-1).astype(np.float64).reshape(m, 1)
    P0_np = float(measurement_noise_scale) * np.eye(m, dtype=np.float64)
    # [TENSEGRITY_EKF 0.1] Set initial state x0 and define prior
    kf = gtsam.KalmanFilter(state_dim)
    state_gtsam = kf.init(x0_np, P0_np)
    if observe_pose_only:
        H_np = np.zeros((meas_dim, state_dim), dtype=np.float64)
        for i in range(meas_dim):
            H_np[i, i] = 1.0
    else:
        H_np = np.eye(state_dim, dtype=np.float64)

    frames = []
    time = 0.0
    state_for_frame = start_state
    pose = state_for_frame.reshape(-1, 13, 1)[:, :7].flatten()
    frames.append({"time": time, "pose": pose, "state": state_for_frame.detach().clone()})

    with torch.no_grad():
        for k, ctrl in enumerate(tqdm.tqdm(ctrls)):
            have_measurement = k + 1 < len(gt_data)
            state_torch = torch.from_numpy(
                np.array(state_gtsam.mean()).reshape(-1)
            ).to(device=device, dtype=dtype).reshape(1, -1, 1)
            z_np = None
            if have_measurement:
                gt = gt_data[k + 1]
                pos = np.array(gt['pos'], dtype=np.float64).reshape(-1, 3)
                quat = np.array(gt['quat'], dtype=np.float64).reshape(-1, 4)
                z_np = np.hstack([pos, quat]).reshape(-1)
                if not observe_pose_only:
                    lv = np.array(gt['linvel'], dtype=np.float64).reshape(-1, 3)
                    av = np.array(gt['angvel'], dtype=np.float64).reshape(-1, 3)
                    z_np = np.hstack([pos, quat, lv, av]).reshape(-1)
            ctrl_step = _ensure_ctrl_for_step(ctrl, simulator)
            mean_np, state_gtsam = _ekf_step_gtsam(
                kf, state_gtsam, simulator, state_torch, dt, ctrl_step,
                H_np, z_np, Q_sigmas, R_sigmas, n_rods, have_measurement,
                use_finite_diff=use_finite_diff,
                innovation_gate_sigma=innovation_gate_sigma,
                control_jacobian_mode=control_jacobian_mode,
            )
            # [TENSEGRITY_EKF 1.7] Set new prior from x_hat_{k+1}^- for next step
            # state_gtsam is carried into the next loop iteration as the EKF prior.
            state_for_frame = torch.from_numpy(mean_np).to(device=device, dtype=dtype).reshape(1, -1, 1)
            time += dt
            pose = state_for_frame.reshape(-1, 13, 1)[:, :7].flatten()
            frames.append({"time": time, "pose": pose, "state": state_for_frame.detach().clone()})

    return frames
