#!/usr/bin/env python
"""Numerically integrates the nonlinear dynamical system."""

from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate

import problem_setup
import plotting


def main():

    # nonlinear states
    op = problem_setup.OdetProblem()
    t, x_k = integrate_nl_ct_eom(op.x0, np.arange(0, op.T0, op.dt))
    X = x_k[0, :]
    Y = x_k[2, :]

    fig,axs= plt.subplots(4,1)
    plotting.states(x_k,t,axs)
    fig.suptitle('Nonlinear Sim States')
    fig.tight_layout()

    # measurements
    y_k = problem_setup.states_to_meas(x_k,t)

    fig2,axs2 = plt.subplots(4,1)
    plotting.measurements(y_k,t,axs2)
    fig2.suptitle('Nonlinear Sim Measurements')
    fig2.tight_layout()
        
    plt.show()




def nonlinear_ct_eom(x: np.ndarray) -> np.ndarray:
    """Equations of motion for the nonlinear continuous-time system."""
    X, Xdot, Y, Ydot = x
    mu = problem_setup.MU_EARTH
    r3 = np.sqrt(X**2 + Y**2)**3
    return [Xdot, -mu * X / r3, Ydot, -mu * Y / r3]


def integrate_nl_ct_eom(x0: np.ndarray,
                        t_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Integrates the NL EOMs and returns the state at a set of time points."""
    solution = integrate.solve_ivp(fun=lambda t, x: nonlinear_ct_eom(x),
                                   t_span=(t_points[0], t_points[-1]),
                                   t_eval=t_points,
                                   y0=x0,
                                   rtol=1e-9,
                                   atol=1e-9)

    return solution.t, solution.y


if __name__ == "__main__":
    main()

