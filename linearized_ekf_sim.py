#!/usr/bin/env python
from typing import List

import numpy as np
import scipy

import kalman_filters
import linear_sim
import nonlinear_sim
import plotting
import problem_setup


def main():
    # Propogate nominal trajectory
    op = problem_setup.OdetProblem()
    prop_time = 1401 * op.dt
    t, x_k_nom = nonlinear_sim.integrate_nl_ct_eom(
        op.x0, np.arange(0, prop_time, op.dt))

    # Prop nonlinear perturbations
    dx0 = np.array([0, 0.075, 0, -0.021])
    _, x_k_pert_nl = nonlinear_sim.integrate_nl_ct_eom(
        op.x0 + dx0, np.arange(0, prop_time, op.dt))
    station_ids_list = [
        problem_setup.find_visible_stations(x, t)
        for x, t in zip(x_k_pert_nl, t)
    ]

    # Generate noisy measurements
    y_k = problem_setup.states_to_noisy_meas(x_k_pert_nl, t, station_ids_list,
                                             op.R)

    u_k = np.zeros((np.size(t), 2))
    P0 = np.eye(4)
    Q = np.eye(4)
    R = op.R
    dx_k_est, P_est_k, S_k = run_linearized_kf(x_k_nom, u_k, y_k, t, dx0, P0,
                                               problem_setup.MU_EARTH, Q, R)


def run_linearized_kf(x_k: np.ndarray, u_k: np.ndarray, y_k: List,
                      t_k: np.ndarray, x0_est: np.ndarray, P0: np.ndarray,
                      mu: float, dt: float, Q: np.ndarray, R: np.ndarray):
    """Runs the linearized Kalman filter for some set of measurements.

    Args:
        x_k: Full state time history on the nominal trajectory
        u_k: Control time history
        y_k: Measurement time history, as output by
            problem_setup.get_measurements()
        t_k: List of time points
        x0_est: Initial state perturbation estimate
        PO: Initial state perturbation error covariance matrix
        mu: Two-body gravitational parameter
        dt: Discrete time step
        Q: Estimated discrete time process noise covariance matrix
        R: Estimated discrete time measurement noise covariance matrix (for a
            single ground station)

    Returns:
        Time histories of the state purturbation estimate, the state
        perturbation error covariance matrix, and the innovation matrix
    """
    x_est = x0_est
    P = P0

    x_ests = [x_est]
    Ps = [P0]
    Ss = [None]

    for x_nom_minus, x_nom_plus, u_minus, ys_plus, t_minus, t_plus in zip(
            x_k[:-1], x_k[1:], u_k[:-1], y_k[1:], t_k[:-1], t_k[1:]):

        # Find Y vector and ground stations in view at k
        y_plus = []
        station_ids = []
        for y in ys_plus:
            y_plus = np.append(y_plus, y[1:3])
            station_ids = np.append(station_ids, y[4])

        # The total measurement noise covariance for the stacked Y vector (at
        # time k)
        R_aug = scipy.linalg.blockdiag(*[R for y in ys_plus])

        # Calculate F, G, and Omega, Jacobians at k-1
        F_minus, G_minus, Oh_minus, _, _ = linear_sim.calc_dt_jacobians(
            x_nom_minus, mu, dt, t_minus, [])

        # Calculate H and M Jacobians at k
        _, _, _, H_plus, M_plus = linear_sim.calc_dt_jacobians(
            x_nom_plus, mu, dt, t_plus, station_ids)

        # Run one iteration of the Kalman filter to find an estimate at time k
        x_est, P, S = kalman_filters.kf_iteration(x_est, u_minus, P, y_plus,
                                                  F_minus, G_minus, H_plus, Q,
                                                  R_aug)

        x_ests.append(x_est)
        Ps.append(Ps)
        Ss.append(S)

    return np.array(x_ests, Ps, Ss)


if __name__ == "__main__":
    main()
