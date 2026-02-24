import numpy as np


def require_gtsam():
    try:
        import gtsam  # type: ignore
    except Exception as exc:
        raise ImportError(
            "gtsam import failed. Install GTSAM with Python bindings before using "
            "ekf_gtsam (e.g., `pip install gtsam` or your platform-specific build)."
        ) from exc

    missing = []
    if not hasattr(gtsam, "KalmanFilter"):
        missing.append("gtsam.KalmanFilter")
    if not hasattr(gtsam, "GaussianDensity"):
        missing.append("gtsam.GaussianDensity")
    if not hasattr(gtsam, "noiseModel"):
        missing.append("gtsam.noiseModel")
    if missing:
        raise RuntimeError(
            "Missing required GTSAM APIs: "
            + ", ".join(missing)
            + ". Install a GTSAM build with KalmanFilter support."
        )
    return gtsam


def motion_transition(num_nodes, dt):
    state_dim = num_nodes * 6
    pos_dim = num_nodes * 3
    dt = float(dt)
    f = np.eye(state_dim)
    f[:pos_dim, pos_dim:] = dt * np.eye(pos_dim)
    return f


def motion_covariance(num_nodes, sigma):
    state_dim = num_nodes * 6
    sigma = float(sigma)
    return (sigma ** 2) * np.eye(state_dim)


def measurement_covariance(num_measurements, sigma):
    sigma = float(sigma)
    return (sigma ** 2) * np.eye(num_measurements)


def measurement_jacobian(positions, edges, num_nodes, eps=1e-8):
    edges = np.asarray(edges, dtype=int)
    positions = np.asarray(positions, dtype=float)
    eps = float(eps)
    num_edges = edges.shape[0]
    state_dim = num_nodes * 6
    jac = np.zeros((num_edges, state_dim))

    for idx, (i, j) in enumerate(edges):
        delta = positions[i] - positions[j]
        dist = np.linalg.norm(delta)
        denom = dist if dist > eps else eps
        grad = delta / denom
        jac[idx, i * 3:(i + 1) * 3] = grad
        jac[idx, j * 3:(j + 1) * 3] = -grad

    return jac
