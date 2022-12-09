"""Kalman filter update equations."""
from typing import Tuple

import numpy as np


def a_priori_covariance(P: np.ndarray, F: np.ndarray,
                        Q: np.ndarray) -> np.ndarray:
    """Finds the a priori estimation error covariance matrix.

    Args:
        P: A posteriori estimation error covariance at time k-1
        F: Dynamics matrix at time k-1
        Q: Process noise covariance matrix at time k-1
    """
    return F @ P @ F.T + Q


def kalman_gain(Pm: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Computes Kalman gain at this timestep.

    Args:
        Pm: A priori estimation error covariance matrix at time k
        H: Measurement matrix at time k
        R: Measurement noise covariance matrix at time k
    """
    return Pm @ H.T @ np.linalg.inv(H @ Pm @ H.T + R)


def a_priori_state(x: np.ndarray, u: np.ndarray, F: np.ndarray,
                   G: np.ndarray) -> np.ndarray:
    """Finds the a priori state estimate at time k.

    Args:
        x: A posteriori state estiate at time k-1
        u: Control at time k-1
        F: Dynamics matrix at time k-1
        G: Control effect matrix at time k-1
    """
    return F @ x + G @ u


def a_posteriori_state(xm: np.ndarray, y: np.ndarray, H: np.ndarray,
                       K: np.ndarray) -> np.ndarray:
    """Finds the a posteriori state estimate.

    Args:
        xm: A priori state estimate at time k
        y: Measurement at time k
        H: Output matrix at time k
        K: Kalman gain at time k
    """
    return xm + K @ (y - H @ xm)


def a_posteriori_covariance(Pm: np.ndarray, H: np.ndarray, R: np.ndarray,
                            K: np.ndarray) -> np.ndarray:
    """Finds the a posteriori estimation error covariance matrix.

    Args:
        Pm: A priori estimation error covariance matrix at time k
        H: Output matrix at time k
        R: Measurement noise covariance matrix at time k
        K: Kalman gain at time k
    """
    n = np.size(H, 1)
    M = np.eye(n) - K @ H
    return M @ Pm @ M.T + K @ R @ K.T


def innovation_covariance_matrix(Pm: np.ndarray, H: np.ndarray,
                                 R: np.ndarray) -> np.ndarray:
    """Finds the "innovation" covariance matrix needed for the NIS test.

    Args:
        Pm: A priori estimation error covariance matrix at time k
        H: Output matrix at time k
        R: Measurement noise covariance matrix at time k
    """
    return H @ Pm @ H.T + R


def a_priori_measurement(xm: np.ndarray, H: np.ndarray, u: np.ndarray,
                         M: np.ndarray) -> np.ndarray:
    """Finds the estimated measurement needed for the NIS test.

    Args:
        xm: A priori state estimate at time k
        H: Output matrix at time k
        u: Control at time k
        M: Control to output mapping matrix at time k
    """
    print(f'{H=},{xm=},{M=},{u=}')
    return H @ xm + M @ u


def kf_iteration(x: np.ndarray, u: np.ndarray, P: np.ndarray, y: np.ndarray,
                 F: np.ndarray, G: np.ndarray, H: np.ndarray, M: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Runs one iteration of the linear Kalman filter (KF).

    Args:
        x: A posteriori state estimate at time k-1
        u: Control at time k-1
        P: A posteriori estimation error covariance at time k-1
        y: Measurement at time k
        F: Dynamics matrix at time k-1
        G: Control effect matrix at time k-1
        H: Measurement matrix at time k
        Q: Process noise covariance matrix at time k-1
        R: Measurement noise covariance matrix at time k

    Returns:
        A tuple of the a posteriori state estimate at time k and the a
        posteriori estimation error covariance estimate at time k.
    """
    Pm = a_priori_covariance(P, F, Q)
    K = kalman_gain(Pm, H, R)
    xm = a_priori_state(x, u, F, G)
    ym = a_priori_measurement(xm, H, u, M)
    xp = a_posteriori_state(xm, y, H, K)
    Pp = a_posteriori_covariance(Pm, H, R, K)
    S = innovation_covariance_matrix(Pm, H, R)

    return xp, ym, Pp, S
