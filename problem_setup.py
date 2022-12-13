import dataclasses

from typing import List, Tuple

import numpy as np
from scipy.io import loadmat

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

        measurements.append([rho, rhodot, phi, ii])

    return measurements


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


def form_stacked_meas_vecs(y_k: np.ndarray) -> Tuple[List, List]:
    """Converts 3d array of measurement vectors to 2d array of stacked vectors
    and 2d list of station ids.

    Args:
        y_k: 3d array of outputs, first  dimension is time steps, second
            dimension holds measurement vectors for each ground station in
            view, third dimension are the measurements in form of
            [rho,rhodot,phi,id] (output of states_to_meas)

    Returns:
        y_k_stack: 2d list of ouptuts, first dimension is time steps, second
            dimension is a stacked vector of measurements
        station_ids_k: 2d list of station ids corresponding to the measurements
            in y_k_stack
    """
    y_k_stack = [[] for i in range(len(y_k))]
    station_ids_k = [[] for i in range(len(y_k))]

    for t_idx, y in enumerate(y_k):
        y_stack = []
        s_ids = []
        for meas in y:
            y_stack.extend(meas[0:3])
            s_ids.append(meas[3])
        y_k_stack[t_idx] = y_stack
        station_ids_k[t_idx] = s_ids

    return y_k_stack, station_ids_k


def unstack_meas_vecs(y_k_stack: List, station_ids_k: List) -> List:
    """Converts 2d array of stacked measurement vectors at each time step to 3d
    array of measurements with station ids.

    Args:
        y_k_stack: 2d list of ouptuts, first dimension is time steps, second
            dimension is a stacked vector of measurements
        station_ids_k: 2d list of station ids corresponding to the measurements
            in y_k_stack

    Returns:
        y_k: 3d array of outputs, first  dimension is time steps, second
            dimension holds measurement vectors for each ground station in
            view, third dimension are the measurements in form of
            [rho,rhodot,phi,id] (output of states_to_meas)
    """

    y_k = [[] for i in range(len(y_k_stack))]
    for t_idx, (y_stack,
                station_ids) in enumerate(zip(y_k_stack, station_ids_k)):
        y = []
        p = len(y_stack)
        for meas_idx in range(p // 3):
            # form measurement array in form [rho,rhodot,phi,id]
            meas = y_stack[meas_idx * 3:meas_idx * 3 + 3].tolist()
            meas.append(station_ids[meas_idx])
            y.append(meas)
        y_k[t_idx] = y
    return y_k


def addsubtract_meas_vecs(y_k_plus: List,
                          y_k_minus: List,
                          s: int,
                          m: float = 1) -> List:
    """Add or subtract y_k_minus to/from y_k_plus. If station IDs don't match,
    then the measurements are ignored.

    Args:
        y_k_plus: 3d array of outputs, first  dimension is time steps, second
            dimension holds measurement vectors for each ground station in
            view, third dimension are the measurements in form of
            [rho,rhodot,phi,id] (output of states_to_meas)
        y_k_minus: same
        s: switch to define operation, 1 for addition, -1 for subtraction
        m: optional multiplication scalar


    Returns:
        y_k_ans: same, sum or difference between the measurement values for
            measurements with matching station IDs
    """

    # check if they are the same length
    if len(y_k_plus) != len(y_k_minus):
        raise ValueError(
            'Measurement sets have different number of time steps.')

    # form stacked meas vectors
    y_k_stack_plus, station_ids_k_plus = form_stacked_meas_vecs(y_k_plus)
    y_k_stack_minus, station_ids_k_minus = form_stacked_meas_vecs(y_k_minus)

    y_k_stack_ans = [[] for i in y_k_minus]
    station_ids_k_ans = [[] for i in y_k_minus]
    for t_idx, (y_plus, y_minus, s_id_plus, s_id_minus) in enumerate(
            zip(y_k_stack_plus, y_k_stack_minus, station_ids_k_plus,
                station_ids_k_minus)):

        y_ans = []
        s_id_ans = []

        # if the same stations are in view, subtract like normal
        if len(s_id_plus) == len(s_id_minus) and np.all(
                np.equal(s_id_plus, s_id_minus)):
            y_ans = (np.array(y_plus) + s * np.array(y_minus)) * m
            s_id_ans = s_id_plus

        else:

            # if one of the station id lists is empty, ignore them
            if np.size(s_id_plus) != 0 and np.size(s_id_minus) != 0:

                # find which ids match and what their idxs are in the id lists
                s_id_match = []
                s_id_match_idx = []
                for p_idx, id_p in enumerate(s_id_plus):
                    for m_idx, id_m in enumerate(s_id_minus):
                        if id_p == id_m:
                            s_id_match.append(id_m)
                            s_id_match_idx.append([p_idx, m_idx])

                # sort so that id numbers are in order
                s_id_ans = sorted(s_id_match)
                s_id_ans_idx = [
                    idx for _, idx in sorted(zip(s_id_match, s_id_match_idx))
                ]

                # for each matching id, do the addition/subtraction
                for s_id_idx in s_id_ans_idx:
                    y_p = y_plus[s_id_idx[0] * 3:s_id_idx[0] * 3 + 3]
                    y_m = y_minus[s_id_idx[1] * 3:s_id_idx[1] * 3 + 3]
                    y_ans.extend((np.array(y_p) + s * np.array(y_m)) * m)

        # print(f'{s_id_minus=},{s_id_plus=},{s_id_ans=}')

        y_k_stack_ans[t_idx] = y_ans
        station_ids_k_ans[t_idx] = s_id_ans

    # Put measurments back into unstacked form with station ids
    y_k_ans = unstack_meas_vecs(y_k_stack_ans, station_ids_k_ans)
    return y_k_ans


def retrieve_meas_with_station_id(y: List, station_ids: List) -> List:
    """Return list of measurements from y with the station ids specified.

    Args:
        y: 2d list, 0 dimension holds measurement vectors for each ground
            station in view, 1 dimension are the measurements in form of
            [rho,rhodot,phi,id] (output of states_to_meas)
        station_ids: list of station ids, corresponding measurements will be
            returned

    Returns:
        y_of_stations: 2d list of measurements for the input station ids
    """

    y_of_stations = []
    for id in sorted(station_ids):
        for meas in y:
            if meas[3] == id:
                y_of_stations.append(meas)
    return y_of_stations


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
        y[0:3] += util.sample_random_vec(np.zeros(3), noise_covariance)

    return y_stack


def states_to_noisy_meas(x_k: np.ndarray, time: np.ndarray,
                         station_ids_list: List,
                         noise_covariance: np.ndarray) -> List:
    y_k = [[] for i in time]
    for idx, (t, station_ids) in enumerate(zip(time, station_ids_list)):
        y_k[idx] = sample_noisy_measurements(x_k[idx, :], t, station_ids,
                                             noise_covariance)

    return y_k


def form_process_noise(T: int, cov: np.ndarray):
    """Form Tx2 vector of process noise with covariance matrix cov."""
    w = np.zeros((T, 2))
    for t_idx in range(T):
        w[t_idx, :] = util.sample_random_vec(np.zeros(2), cov)
    return w


def form_zero_process_noise(T: int) -> np.ndarray:
    """Form Tx2 vector of zeros (sized for process noise)."""
    return np.zeros((T, 2))


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

    # Process noise covariance
    W: np.ndarray = None

    def __init__(self) -> None:
        self.x0 = [self.r0, 0, 0, self.v0]
        self.R = np.array([[0.01, 0, 0], [0, 1, 0], [0, 0, 0.01]])
        self.W = np.eye(2) * 1e-10

    def load_canvas_data(self) -> None:
        rel_path = '5044/asen5044-odet-project/orbitdeterm_finalproj_KFdata.mat'
        # TODO for galen: make a rel_path for your computer here if you want
        canvas_data = loadmat(rel_path)
    
        # pull time vector out
        self.time = canvas_data['tvec'][0]

        # pull measurements out and put them in list form
        self.y = [[] for i in self.time]
        for t_idx,ycell in enumerate(canvas_data['ydata'][0]):
            self.y[t_idx] = ycell.transpose().tolist()

        self.y[0][0][-1] = 0

        


