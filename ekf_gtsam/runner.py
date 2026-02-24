import numpy as np

from .adapter import TensegrityAdapter
from .factors import (
    measurement_jacobian,
    motion_transition,
    require_gtsam,
)
from .measurements import compute_cable_lengths, prepare_measurements, validate_edges
from .state import pack_state, unpack_state


def _position_rmse(aligned_est, gt):
    diff = aligned_est - gt
    return np.sqrt(np.mean(diff ** 2))


def _make_gaussian_density(mean, covariance):
    gtsam = require_gtsam()
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    constructors = []
    if hasattr(gtsam.GaussianDensity, "FromMeanAndCovariance"):
        constructors.append(lambda: gtsam.GaussianDensity.FromMeanAndCovariance(mean, covariance))
    if hasattr(gtsam.GaussianDensity, "fromMeanAndCovariance"):
        constructors.append(lambda: gtsam.GaussianDensity.fromMeanAndCovariance(mean, covariance))
    if hasattr(gtsam.GaussianDensity, "FromMeanAndInformation"):
        info = np.linalg.inv(covariance)
        constructors.append(lambda: gtsam.GaussianDensity.FromMeanAndInformation(mean, info))

    constructors.extend(
        [
            lambda: gtsam.GaussianDensity(mean, covariance),
            lambda: gtsam.GaussianDensity(covariance, mean),
        ]
    )

    if hasattr(gtsam.noiseModel, "Gaussian") and hasattr(
        gtsam.noiseModel.Gaussian, "Covariance"
    ):
        model = gtsam.noiseModel.Gaussian.Covariance(covariance)
        constructors.append(lambda: gtsam.GaussianDensity(mean, model))
        constructors.append(lambda: gtsam.GaussianDensity(model, mean))

    for ctor in constructors:
        try:
            return ctor()
        except Exception:
            continue
    raise RuntimeError(
        "Unable to construct gtsam.GaussianDensity from mean/covariance. "
        "Check your GTSAM Python bindings."
    )


def _noise_model_from_sigma(dim, sigma):
    gtsam = require_gtsam()
    sigmas = np.full(int(dim), float(sigma))
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def _density_mean_cov(density):
    for mean_attr in ("mean", "getMean"):
        if hasattr(density, mean_attr):
            mean_val = getattr(density, mean_attr)()
            break
    else:
        raise RuntimeError(
            "Unable to access mean from gtsam.GaussianDensity. "
            "Expected mean() or getMean()."
        )

    for cov_attr in ("covariance", "getCovariance"):
        if hasattr(density, cov_attr):
            cov_val = getattr(density, cov_attr)()
            break
    else:
        raise RuntimeError(
            "Unable to access covariance from gtsam.GaussianDensity. "
            "Expected covariance() or getCovariance()."
        )

    return np.asarray(mean_val).reshape(-1), np.asarray(cov_val)


def run_kf(
    positions,
    velocities,
    edges,
    dt,
    real_measurements=None,
    measurement_noise_sigma=0.01,
    motion_noise_sigma=0.1,
    initial_covariance=1.0,
    num_steps=None,
):
    require_gtsam()

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"positions must be (T, N, 3); got {positions.shape}"
        )
    num_steps = int(num_steps) if num_steps is not None else positions.shape[0] - 1
    num_steps = min(num_steps, positions.shape[0] - 1)
    num_nodes = positions.shape[1]

    if velocities is None:
        velocities = np.zeros_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) / float(dt)
        velocities[0] = velocities[1]
    velocities = np.asarray(velocities, dtype=float)
    if velocities.shape != positions.shape:
        raise ValueError(
            f"velocities must match positions shape; got {velocities.shape}"
        )

    edges = validate_edges(edges, num_nodes)
    state_dim = num_nodes * 6

    if real_measurements is not None:
        real_measurements = np.asarray(real_measurements, dtype=float)
        if real_measurements.ndim == 1:
            real_measurements = np.repeat(
                real_measurements.reshape(1, -1),
                positions.shape[0],
                axis=0,
            )
        if real_measurements.shape[0] < positions.shape[0]:
            raise ValueError(
                "real_measurements must have at least T rows if provided"
            )

    x0 = pack_state(positions[0], velocities[0])
    p0 = np.eye(state_dim) * float(initial_covariance)

    gtsam = require_gtsam()
    kf = gtsam.KalmanFilter(state_dim)
    f_mat = motion_transition(num_nodes, dt)
    q_model = _noise_model_from_sigma(state_dim, motion_noise_sigma)
    r_model = _noise_model_from_sigma(edges.shape[0], measurement_noise_sigma)

    filtered_states = [x0]
    innovations = []
    cable_rmse = []
    pos_rmse = []
    measurement_sources = []

    for step in range(num_steps):
        prior = _make_gaussian_density(x0, p0)
        b_mat = np.zeros_like(f_mat)
        u_vec = np.zeros((state_dim, 1))
        pred_density = kf.predict(prior, f_mat, b_mat, u_vec, q_model)
        pred_state, p_pred = _density_mean_cov(pred_density)

        if real_measurements is not None:
            measurements = real_measurements[step + 1]
            measurements, source = prepare_measurements(
                positions[step + 1], edges, measurements
            )
        else:
            measurements, source = prepare_measurements(
                positions[step + 1], edges, None
            )
        measurement_sources.append(source)

        pred_positions, _ = unpack_state(pred_state)
        innovation = compute_cable_lengths(pred_positions, edges) - measurements
        innovations.append(float(np.linalg.norm(innovation)))

        h_mat = measurement_jacobian(pred_positions, edges, num_nodes)
        h_pred = compute_cable_lengths(pred_positions, edges)
        z_lin = measurements - h_pred + h_mat @ pred_state

        post_density = kf.update(pred_density, h_mat, z_lin, r_model)
        post_state, p0 = _density_mean_cov(post_density)
        x0 = post_state
        filtered_states.append(post_state)

        post_positions, _ = unpack_state(post_state)
        pred_lengths = compute_cable_lengths(post_positions, edges)
        cable_rmse.append(float(np.sqrt(np.mean((pred_lengths - measurements) ** 2))))

        gt_positions = positions[step + 1]
        centroid_offset = gt_positions.mean(axis=0) - post_positions.mean(axis=0)
        aligned_est = post_positions + centroid_offset
        pos_rmse.append(float(_position_rmse(aligned_est, gt_positions)))

    return {
        "filtered_states": np.asarray(filtered_states),
        "innovation_norms": np.asarray(innovations),
        "cable_rmse": np.asarray(cable_rmse),
        "position_rmse": np.asarray(pos_rmse),
        "measurement_sources": measurement_sources,
    }


def run_from_rollout(
    rollout,
    dt,
    real_measurements=None,
    measurement_noise_sigma=0.01,
    motion_noise_sigma=0.1,
    initial_covariance=1.0,
    num_steps=None,
):
    adapter = TensegrityAdapter(rollout)
    positions = adapter.get_positions()
    velocities = adapter.get_velocities()
    edges = adapter.get_edges()
    if positions is None or edges is None:
        raise ValueError(
            "rollout must include positions and edges for KF execution"
        )
    if real_measurements is None:
        real_measurements = adapter.get_measurements(edges)
    return run_kf(
        positions=positions,
        velocities=velocities,
        edges=edges,
        dt=dt,
        real_measurements=real_measurements,
        measurement_noise_sigma=measurement_noise_sigma,
        motion_noise_sigma=motion_noise_sigma,
        initial_covariance=initial_covariance,
        num_steps=num_steps,
    )


def run_ekf(*args, **kwargs):
    raise RuntimeError(
        "EKF mode has been replaced with a linear Kalman filter. "
        "Use run_kf or run_from_rollout instead."
    )
