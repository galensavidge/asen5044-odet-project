#!/usr/bin/env python
"""Propagates the linearized dynamical system."""

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import problem_setup
import plotting
import nonlinear_sim

def main():

    
    # propogate nominal trajectory
    op = problem_setup.OdetProblem()
    t, x_k = nonlinear_sim.integrate_nl_ct_eom(op.x0, np.arange(0, op.T0, op.dt))

    # prop linearized perturbations
    # dx0 = np.array([5,0.1,5,0.1])
    dx0 = np.array([3,0,0,0])
    u_k = np.zeros((np.size(t),2))
    
    dx_k,dy_k = prop_pert(dx0,x_k,u_k,op.dt,problem_setup.MU_EARTH)

    # prop nonlinear perturbations
    _,x_k_pert = nonlinear_sim.integrate_nl_ct_eom(op.x0 + dx0, np.arange(0, op.T0, op.dt))
    dx_k_nl = x_k_pert - x_k

    fig,axs= plt.subplots(4,1)
    plotting.states(dx_k,t,axs,'Linearized')
    plotting.states(dx_k_nl,t,axs,'Nonliear')
    fig.suptitle('Linearized Sim State Perturbations')
    fig.tight_layout()

    fig2,axs2= plt.subplots(4,1)
    plotting.states(dx_k+x_k,t,axs2,'Linearized w/ pert')
    plotting.states(x_k,t,axs2,'Nominal')
    plotting.states(x_k_pert,t,axs2,'Nonlinear w/ pert')

    fig2.suptitle('Linearized Sim State')
    fig2.tight_layout()

    plt.show()


def prop_pert(dx0:np.ndarray,x_nom:np.ndarray,du_k:np.ndarray,dt:float,mu:float) -> np.ndarray:
    """Propogate perturbations through linearized dynamics.
    
    Args:
        dx0: 4x1 array, initial perturbation state
        x_nom: 4xT array, nominal trajectory
        du_k: 2xT array, nominal input perturbation
        dt: time step
        mu: gravitational parameter

    Returns:
        dx_k: 4xT array, state perturbations from the nominal trajectory
        dy_k: 4xT array, measurement perturbations from the nominal trajectory
    """

    dx_k = np.zeros(np.shape(x_nom))
    dx_k[0,:] = dx0
    dy_k = np.zeros((np.size(x_nom,0),3))

    for t_idx,x in enumerate(x_nom):
        dx = dx_k[t_idx,:]
        du = du_k[t_idx,:]

        # calc time step using nominal trajectory
        F,G,Oh,H,M = calc_dt_jacobians(x,mu,dt)

        # save perturbation propogation
        if t_idx != np.size(x_nom,0) -1:
            dx_k[t_idx+1,:] = np.matmul(F,dx) + np.matmul(G,du)
        dy_k[t_idx] = np.matmul(H,dx) + np.matmul(M,du)

    return dx_k,dy_k


def calc_dt_jacobians(x: np.ndarray, mu:float, dt: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Calculate the discrete-time Jacobians at a time step.
    
    Args:
        x: state vector [X,Xdot,Y,Ydot]
        mu: graviational paramter
        dt: time step 
    
    Returns:
        F: 4x4 array, dynamics Jacobian
        G: 4x2 array, input Jacobian
        Oh: 4X2 array, process noise Jacobian
        H: 3x4 array, output Jacobian
        M: 3x2 array, feedthrough Jacobian
    """

    A,B,Gam,C,D = calc_ct_jacobians(x,mu)
    F = np.eye(4) + A*dt
    G = dt*B
    Oh = dt*Gam
    H = C
    M = D
    return F,G,Oh,H,M


    
def calc_ct_jacobians(x: np.ndarray, mu:float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Calculate the continuous-time Jacobians at a time step.
    
    Args:
        x: state vector [X,Xdot,Y,Ydot]
        mu: graviational paramter
    
    Returns:
        A: 4x4 array, dynamics Jacobian
        B: 4x2 array, input Jacobian
        Gam: 4X2 array, process noise Jacobian
        C: 3x4 array, output Jacobian
        D: 3x2 array, feedthrough Jacobian
    """

    X, Xdot, Y, Ydot = x
    r = np.linalg.norm([X,Y])
    A = np.array([[0,1,0,0],[mu/r**5*(2*X**2-Y**2),0,mu/r**5*3*X*Y,0],[0,0,0,1],[mu/r**5*3*X*Y,0,mu/r**5*(2*Y**2-X**2),0]])
    B = np.array([[0,0],[1,0],[0,0],[0,1]])
    Gam = np.array([[0,0],[1,0],[0,0],[0,1]])
    
    # TODO: fill in C
    C = np.zeros((3,4))
    D = np.zeros((3,2))

    return A,B,Gam,C,D


if __name__ == "__main__":
    main()



# TODO: C/H matrices, need access to which station is in sight and its location