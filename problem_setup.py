import dataclasses

from typing import List

import numpy as np

import util

MU_EARTH = 398600.  # [km^3/s^2]
R_EARTH = 6378.  # [km]

OMEGA_EARTH = 2 * np.pi / 86400.  # [rad/sec]


def ground_station_position(station_id: int, time: float):
    """Returns the Cartesian position of a ground station.

    Args:
        station_id: xero-indexed
    """
    theta = OMEGA_EARTH * time + station_id * np.pi / 6
    return R_EARTH * np.array([np.cos(theta), np.sin(theta)])


def ground_station_velocity(station_id: int, time: float):
    """Returns the Cartesian velocity of a ground station.

    Args:
        station_id: zero-indexed
    """
    theta = OMEGA_EARTH * time + station_id * np.pi / 6
    return R_EARTH * OMEGA_EARTH * np.array([np.sin(theta), -np.cos(theta)])


def check_ground_station_visibility(station_id: int, time: float, X: float,
                                    Y: float) -> bool:
    """Returns if satellite is within range of ground station.

    Args:
        station_id: zero-indexed
    """

    Xi, Yi = ground_station_position(station_id, time)
    theta = OMEGA_EARTH * time + station_id * np.pi / 6
    phi = np.arctan2(Y - Yi, X - Xi)

    # Check if the satellite is in view of this station
    ang_diff = util.wrap_angle_negpi_pi(phi - theta)
    if np.abs(ang_diff) < np.pi / 2:
        return True
    return False


def get_measurements(x: np.ndarray, time: float,
                     station_ids: List) -> np.ndarray:
    """Calculates measurements for ground stations in view at a specific time.

    Args:
        x: Satellite state vector
        time: Simulation time
        station_ids: list of zero-indexed station IDs that are in view

    Returns:
        A list of measurements in the form:
            [rho, rho_dot, phi, station_id]
    """
    X, Xdot, Y, Ydot = x
    measurements = []

    for ii in station_ids:
        Xi, Yi = ground_station_position(ii, time)
        Xdoti, Ydoti = ground_station_velocity(ii, time)
        phi = np.arctan2(Y - Yi, X - Xi)

        rho = np.sqrt((X - Xi)**2 + (Y - Yi)**2)
        rhodot = ((X - Xi) * (Xdot - Xdoti) + (Y - Yi) * (Ydot - Ydoti)) / rho

        measurements.append([rho, rhodot, phi, ii + 1])

    return np.array(measurements)


def find_visible_stations(x: np.ndarray, time: float):
    """Checks what ground stations are visible at a state and time.

    Args:
        x: Satellite state vector
        time: Simulation time

    Returns:
        List of zero-indexed station IDs that are in view
    """
    station_ids = []
    for ii in range(12):
        if check_ground_station_visibility(ii, time, x[0], x[2]):
            station_ids.append(ii)

    return station_ids


def states_to_meas(x_k: np.ndarray, time: np.ndarray,
                   station_ids_list: List) -> List:
    """Converts series of state vectors to measurement vectors.

    Args:
        x_k: 4xT array of satellite state vectors at each time step
        time: array of length T of time at each time step

    Returns:
        y_k: 3d array of outputs, first  dimension is time steps, second
            dimension holds measurement vectors for each ground station in
            view, third dimension are the measurements in form of
            [rho,rhodot,phi,id]
    """

    y_k = [[] for i in time]
    for idx, (t, station_ids) in enumerate(zip(time, station_ids_list)):
        y_k[idx] = get_measurements(x_k[idx, :], t, station_ids)

    return y_k


def form_stacked_meas_vecs(y_k: np.ndarray) -> List:
    """Converts 3d array of measurement vectors to 2d array of stacked vectors.

    Args:
        y_k: 3d array of outputs, first  dimension is time steps, second
            dimension holds measurement vectors for each ground station in
            view, third dimension are the measurements in form of
            [rho,rhodot,phi,id] (output of states_to_meas)

    Returns:
        y_k_stack: 2d list of ouptuts, first dimension is time steps, second
            dimension is a stacked vector of measurements
    """
    y_k_stack = [[] for i in range(np.size(y_k, 0))]

    for t_idx, y in enumerate(y_k):
        y_stack = []
        for meas in y:
            y_stack.extend(meas[0:3])
        y_k_stack[t_idx] = y_stack

    return y_k_stack


def sample_noisy_measurements(x: np.ndarray, time: float,
                              station_ids: np.ndarray,
                              noise_covariance: np.ndarray) -> np.ndarray:
    """Adds noise to measurements at a specific time.

    Args:
        x: Satellite state vector
        time: Simulation time
        station_ids: list of zero-indexed station IDs that are in view
        noise_covariance: The covariance matrix of the AWGN to be added to the
            measurements

    Returns:
        A list of noisy measurements in the form:
            [rho, rho_dot, phi, station_id]
    """
    y_stack = get_measurements(x, time, station_ids)

    for y in y_stack:
        y[1:3] += util.sample_random_vec(np.zeros(3), noise_covariance)

    return y_stack


def states_to_noisy_meas(x_k: np.ndarray, time: np.ndarray,
                         station_ids_list: List,
                         noise_covariance: np.ndarray) -> List:
    y_k = [[] for i in time]
    for idx, (t, station_ids) in enumerate(zip(time, station_ids_list)):
        y_k[idx] = sample_noisy_measurements(x_k[idx, :], t, station_ids,
                                             noise_covariance)

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

    # Measurement noise covariance
    R: np.ndarray = None

    def __init__(self) -> None:
        self.x0 = [self.r0, 0, 0, self.v0]
        self.R = np.ndarray([[0.01, 0, 0], [0, 1, 0], [0, 0, 0.01]])
