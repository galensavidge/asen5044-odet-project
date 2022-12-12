"""Monte Carlo simulations."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

import consistency_tests
import linearized_ekf_sim
import nonlinear_sim
import plotting
import problem_setup
import util


class Sim:

    def __init__(self, tfinal: float,x0: np.ndarray):
        # common parameters
        self.op = problem_setup.OdetProblem()

        # Time vector
        self.tfinal = tfinal
        self.time = np.arange(0, tfinal, self.op.dt)
        self.T = np.size(self.time)

        # Integrate nominal trajectory without noise
        self.x0 = x0
        w_no_noise = problem_setup.form_zero_process_noise(self.T)
        _, self.x_nom = nonlinear_sim.integrate_nl_ct_eom(
            x0, self.op.dt, self.tfinal, w_no_noise)
        self.u = np.zeros((self.T, 2))

class KF_Sim:

    def __init__(self, nom_sim: Sim, x_est0: np.ndarray,
                 P0: np.ndarray, Qkf: np.ndarray, Rkf: np.ndarray,kf_type:str):
        """Run truth sim with noise and perturbation, perform KF, store results.
        
        Args:
            nom_sim: Sim object with nominal trajectory
            x_est0: intitial estimate for KF
            P0: initial error covariance matrix for KF
            Qkf: process noise covariance matrix for KF to estimate with
            Rkf: measurement noise covariance matrix for KF to estimate with
            kf_type: 'LKF' will run the linearized KF (above inputs are then related to perturbations from the nominal trajectory). 'EKF' will run the extended KF (above inputs are related to the full state)
        """

        if kf_type != 'LKF' and kf_type != 'EKF':
            raise ValueError(f'KF type {kf_type} is not valid. Use LKF or EKF.')

        self.nom = nom_sim

        # integrate true trajectory
        x_true0 = util.sample_random_vec(x_est0,P0)
        if kf_type == 'LKF':
            x_true0 += self.nom.x0
        self.integrate_nom_traj(x_true0)

        # run kF
        if kf_type == 'LKF':
            self.linearized_kalman_filter(x_est0,P0,Qkf,Rkf)
        if kf_type == 'EKF':
            self.extended_kalman_filter(x_est0,P0,Qkf,Rkf)

    def integrate_nom_traj(self,x_true0: np.ndarray):

        # Integrate perturbed state with noise
        proccess_noise = problem_setup.form_process_noise(self.nom.T, self.nom.op.W)
        _, self.x_true = nonlinear_sim.integrate_nl_ct_eom(
            x_true0, self.nom.op.dt, self.nom.tfinal, proccess_noise)

        # create measurements
        self.station_ids_list = [
            problem_setup.find_visible_stations(x, t)
            for x, t in zip(self.nom.x_nom, self.nom.time)
        ]
        self.y_nom = problem_setup.states_to_meas(
            self.nom.x_nom, self.nom.time, self.station_ids_list)
        self.y_true = problem_setup.states_to_noisy_meas(
            self.x_true, self.nom.time, self.station_ids_list, self.nom.op.R)


    def linearized_kalman_filter(self, dx_est0: np.ndarray,
                 P0: np.ndarray, Qkf: np.ndarray, Rkf: np.ndarray):

         # find nonlinear perturbations
        self.dx = self.x_true - self.nom.x_nom
        self.dy = problem_setup.addsubtract_meas_vecs(self.y_true, self.y_nom, -1)

        # Estimated filter state and measurements
        self.dx_est, self.dy_est, self.err_cov, self.inn_cov = linearized_ekf_sim.run_linearized_kf(self.nom.x_nom,self.y_true,self.nom.time, dx_est0,P0, self.nom.op.dt, Qkf,Rkf)
        self.x_est = self.dx_est + self.nom.x_nom
        self.y_est = problem_setup.addsubtract_meas_vecs(self.dy_est,self.y_nom,1)

        # Find error
        self.dx_err = self.dx - self.dx_est
        self.dy_err = problem_setup.addsubtract_meas_vecs(self.dy,self.dy_est,-1)
        self.x_err = self.x_true - self.x_est
        self.y_err = problem_setup.addsubtract_meas_vecs(self.y_true,self.y_est,-1)

        # set variables for NEES and NIS
        self.state_err = self.dx_err
        self.meas_err = self.dy_err
    
    def extended_kalman_filter(self, x_est0: np.ndarray, P0: np.ndarray, Qkf: np.ndarray, Rkf: np.ndarray):
        # TODO: fill in with extended kalman filter code
        self.state_err = self.x_true
        self.meas_err = self.y_true
    

def run_monte_carlo(tfinal: float, x0_nom: np.ndarray, x0_est: np.ndarray,P0: np.ndarray,Q:np.ndarray,R:np.ndarray,num_sims: int,kf_type: str) -> List[KF_Sim]:
    """Run Monte Carlo simulations."""

    if kf_type != 'LKF' and kf_type != 'EKF':
            raise ValueError(f'KF type {kf_type} is not valid. Use LKF or EKF.')

    print('Modelling nominal trajectory...')
    nom = Sim(tfinal,x0_nom)
    sims = [[] for i in range(num_sims)]
    for idx in range(num_sims):
        print(f'KF Estimation #{idx+1} out of {num_sims}')
        sims[idx] = KF_Sim(nom,x0_est,P0,Q,R,kf_type)

    return sims

def run_lkf_mc(tfinal: float, x0_nom: np.ndarray,Q:np.ndarray,R:np.ndarray,num_sims: int,alpha: float,plot_states:bool = False,plot_perts:bool = False, plot_err: bool = False,plot_meas: bool = False):
    """Run Monte Carlo with the LKF."""

    dx0 = np.array([0,0,0,0])
    P0 = np.diag([10, 0.2, 10, 0.2])

    sims = run_monte_carlo(tfinal, x0_nom,dx0,P0,Q,R,num_sims,'LKF')

    # plot
    for idx, sim in enumerate(sims):

        if plot_states:
            fig1, axs1 = plt.subplots(4, 1)
            plotting.states(sim.x_true, sim.nom.time, axs1, 'Truth')
            plotting.states(sim.x_est, sim.nom.time, axs1, 'Est')
            fig1.suptitle(f'LKF States, Est #{idx+1}')
            fig1.tight_layout()
        
        if plot_perts:
            fig3, axs3 = plt.subplots(4, 1)
            plotting.states(sim.dx, sim.nom.time, axs3, 'Truth')
            plotting.states(sim.dx_est, sim.nom.time, axs3, 'Est')
            fig1.suptitle(f'LKF State Perturbations, Est #{idx+1}')
            fig1.tight_layout()
        
        if plot_err:
            fig2, axs2 = plt.subplots(4, 1)
            plotting.plot_2sig_err(axs2, sim.dx_err, sim.nom.time,sim.err_cov)
            fig2.suptitle(f'LKF, 2 Sigma Error, Est #{idx+1}')
            fig2.tight_layout()       

        if plot_meas:
            fig4, axs4 = plt.subplots(4, 1)
            plotting.measurements_withids(sim.y_true,sim.nom.time,axs4)
            fig4.suptitle(f'LKF, Measurements, Est #{idx+1}')
            fig4.tight_layout()


    # NEES and NIS
    consistency_tests.nees_and_nis_test(sims, alpha)

def run_ekf_mc(tfinal: float, x0_nom: np.ndarray,Q:np.ndarray,R:np.ndarray,num_sims: int,alpha: float,plot_states:bool = False, plot_err: bool = False,plot_meas: bool = False):
    """Run Monte Carlo with the EKF."""

    x0 = np.array([10, 0.1, -10, -0.1])*400
    P0 = np.diag([10, 0.2, 10, 0.2])

    sims = run_monte_carlo(tfinal, x0_nom,x0,P0,Q,R,num_sims,'EKF')

    # plot
    for idx, sim in enumerate(sims):

        if plot_states:
            fig1, axs1 = plt.subplots(4, 1)
            plotting.states(sim.x_true, sim.nom.time, axs1, 'Truth')
            plotting.states(sim.x_est, sim.nom.time, axs1, 'Est')
            fig1.suptitle(f'EKF, States, Est #{idx+1}')
            fig1.tight_layout()

        if plot_err:
            fig2, axs2 = plt.subplots(4, 1)
            plotting.plot_2sig_err(axs2, sim.x_err, sim.nom.time,sim.err_cov)
            fig2.suptitle(f'EKF, 2 Sigma Error, Est #{idx+1}')
            fig2.tight_layout()

        if plot_meas:
            fig4, axs4 = plt.subplots(4, 1)
            plotting.measurements_withids(sim.y_true,sim.nom.time,axs4)
            fig4.suptitle(f'EKF, Measurements, Est #{idx+1}')
            fig4.tight_layout()

    # NEES and NIS
    consistency_tests.nees_and_nis_test(sims, alpha)



def main():

    # params shared between KFs
    op = problem_setup.OdetProblem()
    tfinal = op.T0 * 0.5
    x0_nom = op.x0
    Q = 10**-10 * np.diag([1, 1])
    R = op.R

    # MC params
    num_sims = 1
    alpha = 0.05

    # run whichever or both
    run_lkf_mc(tfinal,x0_nom,Q,R,num_sims,alpha,True,True,True,True)
    # run_ekf_mc(tfinal,x0_nom,Q,R,num_sims,alpha,False,False,False)

    plt.show()


if __name__ == "__main__":
    main()

# TODO:
# update initialization for true traj