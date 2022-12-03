import numpy as np


def wrap_angle_negpi_pi(theta: float) -> float:
    """Returns an equivalent angle (in rad) in the range (-pi, pi]."""
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta <= -np.pi:
        theta += 2 * np.pi

    return theta


def sample_random_vec(m: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Samples a Gaussian random vector.

    args:
        m: Mean value (n x 1)
        R: Noise covariance (n x n, Hermitian symmetric pos-def)

    returns:
        A realization of x~N(m, R)
    """
    n = np.size(m)
    sqrt_R = np.linalg.cholesky(R)

    return m + sqrt_R @ np.random.randn(n)
