#!/usr/bin/env python
"""Propagates the linearized dynamical system."""

from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import problem_setup
import plotting
import nonlinear_sim


def main():

    # propogate nominal trajectory
    op = problem_setup.OdetProblem()
    prop_time = 1401 * op.dt
    t, x_k = nonlinear_sim.integrate_nl_ct_eom(op.x0,
                                               np.arange(0, prop_time, op.dt))

    # prop nonlinear perturbations
    # dx0 = np.array([5,0.1,5,0.1])
    # dx0 = np.array([3,0,0,0])
    dx0 = np.array([0, 0.075, 0, -0.021])
    _, x_k_pert_nl = nonlinear_sim.integrate_nl_ct_eom(
        op.x0 + dx0, np.arange(0, prop_time, op.dt))
    station_ids_list = [
        problem_setup.find_visible_stations(x, t)
        for x, t in zip(x_k_pert_nl, t)
    ]
    y_k = problem_setup.states_to_meas(x_k, t, station_ids_list)
    y_k_pert_nl = problem_setup.states_to_meas(x_k_pert_nl, t,
                                               station_ids_list)

    # prop linearized perturbations
    u_k = np.zeros((np.size(t), 2))

    dx_k, dy_k = prop_pert(dx0, x_k, station_ids_list, u_k, op.dt,
                           problem_setup.MU_EARTH)

    dx_k_nl = x_k_pert_nl - x_k

    # full perturbed solution
    x_k_pert, y_k_pert = pert_sol(x_k, dx_k, station_ids_list, dy_k, op.dt)


    # fig,axs= plt.subplots(4,1)
    # plotting.states(dx_k,t,axs,'Linearized')
    # plotting.states(dx_k_nl,t,axs,'Nonlinear')
    # fig.suptitle('Linearized Sim State Perturbations')
    # fig.tight_layout()

    # fig2,axs2= plt.subplots(4,1)
    # plotting.states(x_k_pert,t,axs2,'Linearized w/ pert')
    # plotting.states(x_k,t,axs2,'Nominal')
    # plotting.states(x_k_pert_nl,t,axs2,'Nonlinear w/ pert')
    # fig2.suptitle('Linearized Sim State')
    # fig2.tight_layout()

    fig3, axs3 = plt.subplots(4, 1)
    # plotting.measurements_withids(y_k, t, axs3,'Nonimal',color='k')
    plotting.measurements_withids(y_k_pert_nl,
                                  t,
                                  axs3,
                                  'Nonlinear w/ pert',
                                  color='k')
    plotting.measurements_withids(y_k_pert, t, axs3, 'Linearized w/ Pert')
    fig3.suptitle('Linearized Sim Measurements')
    fig3.tight_layout()

    # fig4,axs4 = plt.subplots(4,1)
    # plotting.measurements_withids(dy_k,t,axs4)
    # fig4.suptitle('Linearized Sim Measurement Perturbations')
    # fig4.tight_layout()

    plt.show()


def pert_sol(x_k_nom: np.ndarray, dx_k: np.ndarray, station_ids_list: List,
             dy_k: List, dt: float) -> Tuple[np.ndarray, List]:
    """Add the linearized perturbations to the nominal state to get the full
    perturbed solution.

    For measurements, it uses the NL perturbed solution to assess which ground
    stations are in view.
    """
    x_k = x_k_nom + dx_k

    y_k = [[] for i in x_k_nom]
    for t_idx, (x_nom, station_ids) in enumerate(zip(x_k_nom,
                                                     station_ids_list)):
        t = dt * t_idx

        # get nominal measurements for each of the ground stations in view of
        # the perturbed state
        y_nom = problem_setup.get_measurements(x_nom, t, station_ids)

        # add perturbation to get perturbed measurements
        dy = dy_k[t_idx]
        y = []
        for idx, meas in enumerate(y_nom):
            pert = dy[idx]
            full = np.array(meas[0:3]) + np.array(pert[0:3])
            if meas[3] != pert[3]:
                raise ValueError('Mismatched station IDs!')
            full = np.append(full, pert[3])
            y.append(full)
        y_k[t_idx] = y

    return x_k, y_k


def prop_pert(dx0: np.ndarray, x_nom: np.ndarray, station_ids_list: List,
              du_k: np.ndarray, dt: float,
              mu: float) -> Tuple[np.ndarray, List[List]]:
    """Propogate perturbations through linearized dynamics.

    Args:
        dx0: 4x1 array, initial perturbation state
        x_nom: 4xT array, nominal trajectory
        x_pert: 4xT array, nonlinear trajectory with perturbations
        du_k: 2xT array, nominal input perturbation
        dt: time step
        mu: gravitational parameter

    Returns:
        dx_k: 4xT array, state perturbations from the nominal trajectory
        dy_k: 3d array of measurement perturbations, first  dimension is time
            steps, second dimension holds measurement vectors for each ground
            station in view, third dimension are the measurements in form of
            [rho,rhodot,phi,id]
    """

    dx_k = np.zeros(np.shape(x_nom))
    dx_k[0, :] = dx0
    dy_k = [[] for i in range(np.size(x_nom, 0))]

    for t_idx, (x, station_ids) in enumerate(zip(x_nom, station_ids_list)):
        dx = dx_k[t_idx, :]
        du = du_k[t_idx, :]

        # Current time
        t = dt * t_idx

        # calc time step using nominal trajectory
        F, G, Oh, H, M = calc_dt_jacobians(x, mu, dt, t, station_ids)

        # save state perturbation propogation to the next time step
        if t_idx != np.size(x_nom, 0) - 1:
            dx_k[t_idx + 1, :] = np.matmul(F, dx) + np.matmul(G, du)

        # save measurement pert prop and id only if a ground station is in view
        dy_id = []
        if H is not None:
            dy = np.matmul(H, dx) + np.matmul(M, du)
            idx = 0
            for id_idx in station_ids:
                dy_id.append([
                    dy[idx * 3], dy[idx * 3 + 1], dy[idx * 3 + 2], id_idx + 1
                ])
                idx += 1
        dy_k[t_idx] = dy_id

    return dx_k, dy_k


def calc_dt_jacobians(
    x: np.ndarray, mu: float, dt: float, t: float, station_ids: List[bool]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the discrete-time Jacobians at a time step.

    Args:
        x: state vector [X,Xdot,Y,Ydot]
        mu: graviational paramter
        dt: time step
        t: time
        station_ids: list of 12 booleans specifying which ground stations are
            in view

    Returns:
        F: 4x4 array, dynamics Jacobian
        G: 4x2 array, input Jacobian
        Oh: 4X2 array, process noise Jacobian
        H: 3x4 array, output Jacobian
        M: 3x2 array, feedthrough Jacobian
    """

    A, B, Gam, C, D = calc_ct_jacobians(x, t, mu, station_ids)
    F = np.eye(4) + A * dt
    G = dt * B
    Oh = dt * Gam
    H = C
    M = D
    return F, G, Oh, H, M


def calc_ct_jacobians(
    x: np.ndarray, t: float, mu: float, station_ids: List[bool]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the continuous-time Jacobians at a time step.

    Args:
        x: state vector [X,Xdot,Y,Ydot]
        t: time
        mu: graviational paramter
        station_ids: list of 12 booleans specifying which ground stations are
            in view

    Returns:
        A: 4x4 array, dynamics Jacobian
        B: 4x2 array, input Jacobian
        Gam: 4X2 array, process noise Jacobian
        C: 3*GSx4 array, output Jacobian (GS is number of ground stations in
            view), = None if none are in view
        D: 3*GSx2 array, feedthrough Jacobian (GS is number of ground stations
            in view), = None if none are in view
    """
    X, Xdot, Y, Ydot = x
    r = np.linalg.norm([X, Y])
    A = np.array([[0, 1, 0, 0],
                  [mu / r**5 * (2 * X**2 - Y**2), 0, mu / r**5 * 3 * X * Y, 0],
                  [0, 0, 0, 1],
                  [mu / r**5 * 3 * X * Y, 0, mu / r**5 * (2 * Y**2 - X**2),
                   0]])
    B = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
    Gam = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])

    C = None
    D = None
    for idx in station_ids:
        # ground station state
        Xi, Yi = problem_setup.ground_station_position(idx, t)
        Xdoti, Ydoti = problem_setup.ground_station_velocity(idx, t)

        # create C for this ground station and stack in total C
        rho = np.sqrt((X - Xi)**2 + (Y - Yi)**2)
        rhodot = ((X - Xi) * (Xdot - Xdoti) + (Y - Yi) * (Ydot - Ydoti)) / rho
        Cii = np.array([[(X - Xi) / rho, 0, (Y - Yi) / rho, 0],
                        [(Xdot - Xdoti) / rho - (X - Xi) * rhodot / rho**2,
                         (X - Xi) / rho,
                         (Ydot - Ydoti) / rho - (Y - Yi) * rhodot / rho**2,
                         (Y - Yi) / rho],
                        [-(Y - Yi) / rho**2, 0, (X - Xi) / rho**2, 0]])

        if C is None:
            C = Cii
        else:
            C = np.vstack((C, Cii))

    if C is not None:
        D = np.zeros((np.size(C, 0), 2))

    return A, B, Gam, C, D


if __name__ == "__main__":
    main()

# TODO: y pert doesn't look like his
# current, this looks at the perturbed state to check if the ground station is
# in view and then calcs the measurement vectors for those ground stations. It
# cuts off too early (compared to Dr. Ahmeds results) for some of the states,
# specifically the ones with a lot of state perturbation.
