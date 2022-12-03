#!/usr/bin/env python
"""Numerically integrates the nonlinear dynamical system."""

from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate

import problem_setup
import plotting


def main():
    example_sim_nonoise()
    example_sim_wnoise()

    plt.show()


def example_sim_nonoise():

    # nonlinear states
    op = problem_setup.OdetProblem()
    w = np.zeros((int(np.floor(op.T0 / op.dt)), 2))
    t, x_k = integrate_nl_ct_eom(op.x0, op.dt, op.T0, w)

    fig, axs = plt.subplots(4, 1)
    plotting.states(x_k, t, axs)
    fig.suptitle('Nonlinear Sim States')
    fig.tight_layout()

    # measurements
    station_ids_list = [
        problem_setup.find_visible_stations(x, t) for x, t in zip(x_k, t)
    ]
    y_k = problem_setup.states_to_meas(x_k, t, station_ids_list)

    fig2, axs2 = plt.subplots(4, 1)
    plotting.measurements_withids(y_k, t, axs2)
    fig2.suptitle('Nonlinear Sim Measurements')
    fig2.tight_layout()


def example_sim_wnoise():
    # nonlinear states
    op = problem_setup.OdetProblem()
    w = np.random.randn(int(np.floor(op.T0 / op.dt)), 2) * 1e-5
    t, x_k = integrate_nl_ct_eom(op.x0, op.dt, op.T0, w)

    fig, axs = plt.subplots(4, 1)
    plotting.states(x_k, t, axs)
    fig.suptitle('Nonlinear Sim States, w/ Process Noise')
    fig.tight_layout()

    # measurements
    station_ids_list = [
        problem_setup.find_visible_stations(x, t) for x, t in zip(x_k, t)
    ]
    y_k = problem_setup.states_to_meas(x_k, t, station_ids_list)

    fig2, axs2 = plt.subplots(4, 1)
    plotting.measurements_withids(y_k, t, axs2)
    fig2.suptitle('Nonlinear Sim Measurements w/ Process Noise')
    fig2.tight_layout()


def nonlinear_ct_eom(t: float, x: np.ndarray, w: np.ndarray,
                     dt: float) -> np.ndarray:
    """Equations of motion for the nonlinear continuous-time system."""
    t_idx = int(np.floor(t / dt)) - 1
    X, Xdot, Y, Ydot = x
    mu = problem_setup.MU_EARTH
    r3 = np.sqrt(X**2 + Y**2)**3
    return [Xdot, -mu * X / r3 + w[t_idx, 0], Ydot, -mu * Y / r3 + w[t_idx, 1]]


def integrate_nl_ct_eom(x0: np.ndarray, dt: float, t_final: float,
                        w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Integrates the NL EOMs and returns the state at a set of time points."""

    # construct t evals
    t_points = np.arange(0, t_final, dt)

    # integrate
    solution = integrate.solve_ivp(fun=nonlinear_ct_eom,
                                   t_span=(0, t_final),
                                   t_eval=t_points,
                                   y0=x0,
                                   args=(w, dt),
                                   rtol=1e-12,
                                   atol=1e-12)
    return solution.t, np.transpose(solution.y)


if __name__ == "__main__":
    main()
