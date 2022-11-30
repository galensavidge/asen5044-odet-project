import dataclasses

import numpy as np

import util

MU_EARTH = 398600.  # [km^3/s^2]
R_EARTH = 6378.  # [km]

OMEGA_EARTH = 2 * np.pi / 86400.  # [rad/sec]


def ground_station_position(station_id: int, time: float):
    """Returns the Cartesian position of a ground station."""
    theta = OMEGA_EARTH * time + station_id * np.pi / 6
    return R_EARTH * np.array([np.cos(theta), np.sin(theta)])


def ground_station_velocity(station_id: int, time: float):
    """Returns the Cartesian velocity of a ground station."""
    theta = OMEGA_EARTH * time + station_id * np.pi / 6
    return R_EARTH * OMEGA_EARTH * np.array([np.sin(theta), -np.cos(theta)])


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
        Xi, Yi = ground_station_position(ii, time)
        Xdoti, Ydoti = ground_station_velocity(ii, time)

        phi = np.arctan2(Y - Yi, X - Xi)

        # Check if the satellite is in view of this station
        ang_diff = util.wrap_angle_negpi_pi(phi - theta)
        if not np.abs(ang_diff) < np.pi / 2:
            continue

        rho = np.sqrt((X - Xi)**2 + (Y - Yi)**2)
        rhodot = ((X - Xi) * (Xdot - Xdoti) + (Y - Yi) * (Ydot - Ydoti)) / rho

        measurements.append([rho, rhodot, phi, ii + 1])

    return np.array(measurements)


def states_to_meas(x_k: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Converts series of state vectors to measurement vectors.

    Args:
        x_k: 4xT array of satellite state vectors at each time step
        time: array of length T of time at each time step

    Returns:
        y_k: 2d array of outputs, first  dimension length of T, second
        dimension length varies according to number of ground stations in view
    """

    y_k = [[] for i in time]
    for idx in range(np.size(time)):
        y_k[idx] = get_measurements(x_k[:, idx], time[idx])
    return y_k


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
    dt: float = 10  # [sec]

    def __init__(self) -> None:
        self.x0 = [self.r0, 0, 0, self.v0]
