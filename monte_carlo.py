"""Monte Carlo simulations."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

import consistency_tests
import linearized_ekf_sim
import nonlinear_sim
import plotting
import problem_setup


class Sim:

    def __init__(self, tfinal: float):
        # common parameters
        self.op = problem_setup.OdetProblem()

        # Time vector
        self.tfinal = tfinal
        self.time = np.arange(0, tfinal, self.op.dt)
        self.T = np.size(self.time)

    def sim_truth(self, x0: np.ndarray,dx0:np.ndarray):
        """Integrate nonlinear system with noise."""

        self.x0 = x0
        self.dx0 = dx0

        # Integrate nominal trajectory without noise
        w_no_noise = problem_setup.form_zero_process_noise(self.T)
        _, self.x_nom = nonlinear_sim.integrate_nl_ct_eom(
            x0, self.op.dt, self.tfinal, w_no_noise)
        self.u = np.zeros((self.T, 2))

class KF_Sim:

    def __init__(self, truth_sim: Sim, x_est0: np.ndarray,
                 P0: np.ndarray, Qkf: np.ndarray, Rkf: np.ndarray,kf_type:str):
        """Run truth sim with noise and perturbation, perform KF, store results.
        
        Args:
            truth_sim: Sim object with nominal trajectory
            x_est0: intitial estimate for KF
            P0: initial error covariance matrix for KF
            Qkf: process noise covariance matrix for KF to estimate with
            Rkf: measurement noise covariance matrix for KF to estimate with
            kf_type: 'LKF' will run the linearized KF (above inputs are then related to perturbations from the nominal trajectory). 'EKF' will run the extended KF (above inputs are related to the full state)
        """

        if kf_type != 'LKF' and kf_type != 'EKF':
            raise ValueError(f'KF type {kf_type} is not valid. Use LKF or EKF.')

        self.truth = truth_sim
        self.integrate_nom_traj()
        if kf_type == 'LKF':
            self.linearized_kalman_filter(x_est0,P0,Qkf,Rkf)
        if kf_type == 'EKF':
            self.extended_kalman_filter(x_est0,P0,Qkf,Rkf)

    def integrate_nom_traj(self):

        # Integrate perturbed state with noise
        proccess_noise = problem_setup.form_process_noise(self.truth.T, self.truth.op.W)
        _, self.x_true = nonlinear_sim.integrate_nl_ct_eom(
            self.truth.x0+self.truth.dx0, self.truth.op.dt, self.truth.tfinal, proccess_noise)

        # create measurements
        self.station_ids_list = [
            problem_setup.find_visible_stations(x, t)
            for x, t in zip(self.truth.x_nom, self.truth.time)
        ]
        self.y_nom = problem_setup.states_to_meas(
            self.truth.x_nom, self.truth.time, self.station_ids_list)
        self.y_true = problem_setup.states_to_noisy_meas(
            self.x_true, self.truth.time, self.station_ids_list, self.truth.op.R)


    def linearized_kalman_filter(self, dx_est0: np.ndarray,
                 P0: np.ndarray, Qkf: np.ndarray, Rkf: np.ndarray):

         # find nonlinear perturbations
        self.dx = self.x_true - self.truth.x_nom
        self.dy = problem_setup.addsubtract_meas_vecs(self.y_true, self.y_nom, -1)

        # Estimated filter state and measurements
        self.dx_est, self.dy_est, self.err_cov, self.inn_cov = linearized_ekf_sim.run_linearized_kf(self.truth.x_nom,self.y_true,self.truth.time, dx_est0,P0, self.truth.op.dt, Qkf,Rkf)
        self.x_est = self.dx_est + self.truth.x_nom
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
    

def run_monte_carlo(tfinal: float, x0_true: np.ndarray, dx0: np.ndarray,P0: np.ndarray,Q:np.ndarray,R:np.ndarray,num_sims: int) -> List[KF_Sim]:
    """Run Monte Carlo simulations."""

    print('Modelling truth...')
    truth = Sim(tfinal)
    truth.sim_truth(x0_true,dx0)
    sims = [[] for i in range(num_sims)]
    for idx in range(num_sims):
        print(f'KF Estimation #{idx+1} out of {num_sims}')
        sims[idx] = KF_Sim(truth,dx0,P0,Q,R,'LKF')

    return sims


def main():

    op = problem_setup.OdetProblem()
    tfinal = op.T0 * 3
    x0_true = op.x0
    dx0 = np.array([10, 0.1, -10, -0.1])
    P0 = np.diag([200, 2, 200, 2])
    Q = 10**-10 * np.diag([1, 1])
    R = op.R

    sims = run_monte_carlo(tfinal, x0_true,dx0,P0,Q,R, 5)

    

    for idx, sim in enumerate(sims):

        # plot states
        fig1, axs1 = plt.subplots(4, 1)
        plotting.states(sim.x_true, sim.truth.time, axs1, 'Truth')
        plotting.states(sim.x_est, sim.truth.time, axs1, 'Est')
        fig1.suptitle(f'States, Est #{idx+1}')
        fig1.tight_layout()

        # plot error
        fig2, axs2 = plt.subplots(4, 1)
        plotting.plot_2sig_err(axs2, sim.dx_err, sim.truth.time,sim.err_cov)
        fig2.suptitle(f'2 Sigma Error, Est #{idx+1}')
        fig2.tight_layout()

    

    # NEES and NIS
    consistency_tests.nees_and_nis_test(sims, 0.05)

    plt.show()


if __name__ == "__main__":
    main()

# TODO:
# y true is going to have different station ids from y nom on every run