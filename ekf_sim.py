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

    x_est_0 = op.x0
    P0 = np.diag([200, 2, 200, 2])
    Q = 10**-10 * np.diag([1, 1])
    #  run_ekf_nl_sim(op, x_est_0, P0, Q)

    Q_canvas = 10**-8 * np.diag([1, 1])
    run_ekf_canvas_data(op, x_est_0, P0, Q_canvas)


def run_ekf_nl_sim(op: problem_setup.OdetProblem, x_est_0: np.ndarray,
                   P0: np.ndarray, Q: np.ndarray):
    # Prop nonlinear perturbations
    prop_time = 4 * op.T0
    dx0 = 0.1 * np.array([0, 0.075, 0, -0.021])
    w = problem_setup.form_process_noise(int(np.floor(prop_time / op.dt)),
                                         op.W)
    t, x_k_pert_nl = nonlinear_sim.integrate_nl_ct_eom(op.x0 + dx0, op.dt,
                                                       prop_time, w)
    station_ids_list = [
        problem_setup.find_visible_stations(x, t)
        for x, t in zip(x_k_pert_nl, t)
    ]

    # Generate noisy measurements
    y_k_pert_nl = problem_setup.states_to_noisy_meas(x_k_pert_nl, t,
                                                     station_ids_list, op.R)

    x_k_est, y_k_est, P_est_k, S_k = run_ekf(y_k_pert_nl, t, x_est_0, P0,
                                             op.dt, Q, op.R)

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


def run_ekf_canvas_data(op: problem_setup.OdetProblem, x_est_0: np.ndarray,
                        P0: np.ndarray, Q: np.ndarray):
    op.load_canvas_data()

    x_k_est, y_k_est, P_est_k, S_k = run_ekf(op.y, op.time, x_est_0, P0, op.dt,
                                             Q, op.R)

    fig, axs = plt.subplots(4, 1)
    plotting.plot_2sig_err(axs,
                           x_k_est,
                           op.time,
                           P_est_k,
                           'Estimated',
                           bounds_relative_to_state=True)
    fig.suptitle('Satellite State Estimate')
    fig.tight_layout()

    plt.show()


def run_ekf(y_k: List, t_k: np.ndarray, x0_est: np.ndarray, P0: np.ndarray,
            dt: float, Q: np.ndarray, R: np.ndarray):
    x_est = x0_est
    P = P0

    x_ests = [x_est]
    y_ests = [[]]
    Ps = [P0]
    Ss = [None]

    for ys_plus, t_minus, t_plus in zip(y_k[1:], t_k[:-1], t_k[1:]):
        # Calculate F and Omega at k-1
        Fm, _, Ohm, _, _ = linear_sim.calc_dt_jacobians(
            x_est, problem_setup.MU_EARTH, dt, t_minus, [])

        # A priori covariance update
        Qk = Ohm @ Q @ Ohm.T
        Pm = kalman_filters.a_priori_covariance(P, Fm, Qk)

        # Propagate nonlinear dynamics to get a priori state estimate
        w_no_noise = problem_setup.form_zero_process_noise(2)
        _, xms = nonlinear_sim.integrate_nl_ct_eom(x_est, dt, 1.1 * dt,
                                                   w_no_noise)
        xm = xms[-1]

        # Find measurement vector and ground stations in view at time k
        y_plus = []
        for y in ys_plus:
            y_plus.extend(y[0:3])
        station_ids = [y[3] for y in ys_plus]
        y_plus = np.array(y_plus)

        # Calculate H and M at time k
        _, _, _, H, M = linear_sim.calc_dt_jacobians(xm,
                                                     problem_setup.MU_EARTH,
                                                     dt, t_plus, station_ids)

        # The total measurement noise covariance for the stacked Y vector (at
        # time k)
        R_aug = scipy.linalg.block_diag(*[R for y in ys_plus])

        # Find the Kalman gain at time k
        K = kalman_filters.kalman_gain(Pm, H, R_aug)

        # Get the estimated measurements at time k using the nonlinear output
        # equation
        ys_est = problem_setup.get_measurements(xm, t_plus, station_ids)
        y_est = []
        for y in ys_est:
            y_est.extend(y[0:3])
        y_est = np.array(y_est)

        # A posteriori state estimate at time k
        if len(y_plus) == 0:
            x_est = xm
        else:
            x_est = kalman_filters.a_posteriori_state_ekf(xm, y_plus, y_est, K)

        P = kalman_filters.a_posteriori_covariance_ekf(Pm, H, K)

        # Compute the innovation covariance matrix
        S = kalman_filters.innovation_covariance_matrix(Pm, H, R_aug)

        x_ests.append(x_est)
        Ps.append(P)
        y_ests.append(
            problem_setup.unstack_meas_vecs([y_est], [station_ids])[0])
        Ss.append(S)

    return np.array(x_ests), y_ests, np.array(Ps), Ss


if __name__ == "__main__":
    main()
