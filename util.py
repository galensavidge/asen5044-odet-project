import numpy as np


def wrap_angle_negpi_pi(theta: float) -> float:
    """Returns an equivalent angle (in rad) in the range (-pi, pi]."""
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta <= -np.pi:
        theta += 2 * np.pi

    return theta
