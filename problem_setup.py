import dataclasses

import numpy as np

MU_EARTH = 398600.  # [km^3/s^2]
R_EARTH = 6378.  # [km]

OMEGA_EARTH = 2 * np.pi / 86400.  # [rad/sec]


def get_measurements(x: np.ndarray, time: float) -> np.ndarray:
    """Calculates measurements for all ground stations at a specific time.

    Args:
        x: Satellite state vector
        time: Simulation time

    Returns:
        A list of measurements in the form:
            [rho, rho_dot, phi, station_id]
    """
    X, Xdot, Y, Ydot = x
    measurements = []
    for ii in range(12):
        theta = OMEGA_EARTH * time + ii * np.pi / 6
        Xi, Yi = R_EARTH * [np.cos(theta), np.sin(theta)]
        Xdoti, Ydoti = R_EARTH * OMEGA_EARTH * [np.sin(theta), -np.cos(theta)]

        phi = np.arctan2(Y - Yi, X - Xi)

        # Check if the satellite is in view of this station
        if not -np.pi / 2 < phi - theta < np.pi / 2:
            continue

        rho = np.sqrt((X - Xi)**2 + (Y - Yi)**2)
        rhodot = ((X - Xi) * (Xdot - Xdoti) + (Y - Yi) * (Ydot - Ydoti)) / rho

        measurements.append([rho, rhodot, phi, ii + 1])

    return np.array(measurements)


@dataclasses.dataclass
class OdetProblem:
    """Contains parameters for the ODet estimation problem."""
    # Initial position and veocity
    r0: float = R_EARTH + 300  # [km]
    v0: float = np.sqrt(MU_EARTH / r0)  # [km/s]

    # Nominal initial state
    x0: np.ndarray = None

    # Period of nominal orbit
    T0: float = 2 * np.pi * r0 * np.sqrt(r0 / MU_EARTH)

    # Time between measurements
    dt = 10  # [sec]

    def __init__(self) -> None:
        self.x0 = [self.r0, 0, 0, self.v0]
