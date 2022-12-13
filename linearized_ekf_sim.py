#!/usr/bin/env python
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy

import kalman_filters
import linear_sim
import nonlinear_sim
import plotting
import problem_setup


def main():
    op = problem_setup.OdetProblem()

    dx_est_0 = np.zeros(4)
    P0 = np.diag([10, 0.2, 10, 0.2])
    Q = 10**-10 * np.eye(2)

    run_lkf_nl_sim(op, dx_est_0, P0, Q)
    #  run_lkf_canvas_data(op, dx_est_0, P0, Q)


def run_lkf_nl_sim(op: problem_setup.OdetProblem, dx_est_0: np.ndarray,
                   P0: np.ndarray, Q: np.ndarray):
    prop_time = 4 * op.T0

    # Propogate nominal trajectory
    w_no_noise = np.zeros((int(np.floor(prop_time / op.dt)), 2))
    t, x_k_nom = nonlinear_sim.integrate_nl_ct_eom(op.x0, op.dt, prop_time,
                                                   w_no_noise)

    # Prop nonlinear perturbations
    dx0 = 0.1 * np.array([0, 0.075, 0, -0.021])
    w = problem_setup.form_process_noise(int(np.floor(prop_time / op.dt)),
                                         op.W)
    _, x_k_pert_nl = nonlinear_sim.integrate_nl_ct_eom(op.x0 + dx0, op.dt,
                                                       prop_time, w)
    station_ids_list = [
        problem_setup.find_visible_stations(x, t)
        for x, t in zip(x_k_pert_nl, t)
    ]

    # Generate noisy measurements
    y_k_pert_nl = problem_setup.states_to_noisy_meas(x_k_pert_nl, t,
                                                     station_ids_list, op.R)

    R = op.R
    dx_k_est, dy_k_est, P_est_k, S_k = run_linearized_kf(
        x_k_nom, y_k_pert_nl, t, dx_est_0, P0, op.dt, Q, R)

    x_k_est = x_k_nom + dx_k_est
    x_k_err = x_k_est - x_k_pert_nl

    fig, axs = plt.subplots(4, 1)
    plotting.states(x_k_est, t, axs, 'Estimated')
    plotting.states(x_k_pert_nl, t, axs, 'True')
    fig.suptitle('Satellite State Estimate')
    fig.tight_layout()

    fig, axs = plt.subplots(4, 1)
    plotting.plot_2sig_err(axs, x_k_err, t, P_est_k)
    fig.suptitle('State Estimate Error')
    fig.tight_layout()

    plt.show()


def run_lkf_canvas_data(op: problem_setup.OdetProblem, dx_est_0: np.ndarray,
                        P0: np.ndarray, Q: np.ndarray):
    op.load_canvas_data()

    prop_time = op.time[-1]

    # Propogate nominal trajectory
    w_no_noise = np.zeros((int(prop_time / op.dt), 2))
    t, x_k_nom = nonlinear_sim.integrate_nl_ct_eom(op.x0, op.dt, prop_time,
                                                   w_no_noise)

    dx_k_est, dy_k_est, P_est_k, S_k = run_linearized_kf(
        x_k_nom, op.y, op.time, dx_est_0, P0, op.dt, Q, op.R)

    x_k_est = x_k_nom + dx_k_est

    fig, axs = plt.subplots(4, 1)
    plotting.states(x_k_est, t, axs, 'Estimated')
    fig.suptitle('Satellite State Estimate')
    fig.tight_layout()

    plt.show()


def run_linearized_kf(x_k: np.ndarray, y_k: List, t_k: np.ndarray,
                      x0_est: np.ndarray, P0: np.ndarray, dt: float,
                      Q: np.ndarray, R: np.ndarray):
    """Runs the linearized Kalman filter for some set of measurements.

    Args:
        x_k: Full state time history on the nominal trajectory
        y_k: Measurement time history, as output by
            problem_setup.get_measurements()
        t_k: List of time points
        x0_est: Initial state perturbation estimate
        PO: Initial state perturbation error covariance matrix
        dt: Discrete time step
        Q: Estimated discrete time process noise covariance matrix
        R: Estimated discrete time measurement noise covariance matrix (for a
            single ground station)

    Returns:
        Time histories of the state purturbation estimate, the output
        perturbation estimate, the state perturbation error covariance matrix,
        and the innovation matrix
    """
    u_k = np.zeros((np.size(t_k), 2))

    x_est = x0_est
    P = P0

    x_ests = [x_est]
    y_ests = [[]]
    Ps = [P0]
    Ss = [None]

    for x_nom_minus, x_nom_plus, u_minus, ys_plus, t_minus, t_plus in zip(
            x_k[:-1], x_k[1:], u_k[:-1], y_k[1:], t_k[:-1], t_k[1:]):

        # Find ground stations in view at k
        station_ids = [y[3] for y in ys_plus]

        # Find nominal measurements
        ys_nom = problem_setup.get_measurements(x_nom_plus, t_plus,
                                                station_ids)

        # Current perturbation measumrent
        dy_plus = problem_setup.addsubtract_meas_vecs([ys_plus], [ys_nom],
                                                      -1)[0]
        dy_plus_stack = problem_setup.form_stacked_meas_vecs([dy_plus])[0][0]

        # The total measurement noise covariance for the stacked Y vector (at
        # time k)
        R_aug = scipy.linalg.block_diag(*[R for y in ys_plus])

        # Calculate F, G, and Omega, Jacobians at k-1
        F_minus, G_minus, Oh_minus, _, _ = linear_sim.calc_dt_jacobians(
            x_nom_minus, problem_setup.MU_EARTH, dt, t_minus, [])

        # Calculate H and M Jacobians at k
        _, _, _, H_plus, M_plus = linear_sim.calc_dt_jacobians(
            x_nom_plus, problem_setup.MU_EARTH, dt, t_plus, station_ids)

        Qk = Oh_minus @ Q @ Oh_minus.T

        # Run one iteration of the Kalman filter to find an estimate at time k
        x_est, y_est, P, S = kalman_filters.kf_iteration(
            x_est, u_minus, P, dy_plus_stack, F_minus, G_minus, H_plus, M_plus,
            Qk, R_aug)

        x_ests.append(x_est)
        Ps.append(P)
        y_ests.append(
            problem_setup.unstack_meas_vecs([y_est], [station_ids])[0])
        Ss.append(S)

    return np.array(x_ests), y_ests, np.array(Ps), Ss


if __name__ == "__main__":
    main()
