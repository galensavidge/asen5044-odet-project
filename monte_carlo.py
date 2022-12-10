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

    def __init__(self, truth_sim: Sim, dx_est0: np.ndarray,
                 P0: np.ndarray, Qkf: np.ndarray, Rkf: np.ndarray):

        self.truth = truth_sim
        self.integrate_nom_traj()
        self.kalman_filter(dx_est0,P0,Qkf,Rkf)

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

        # find nonlinear perturbations
        self.dx = self.x_true - self.truth.x_nom
        self.y_true_stacked = problem_setup.form_stacked_meas_vecs(self.y_true)
        y_nom_stacked = problem_setup.form_stacked_meas_vecs(self.y_nom)
        self.dy = [[] for i in self.y_true]
        for t_idx,(y_tr,y_no) in enumerate(zip(self.y_true_stacked,y_nom_stacked)):
            self.dy[t_idx] = np.array(y_tr) - np.array(y_no)
    
    def kalman_filter(self, dx_est0: np.ndarray,
                 P0: np.ndarray, Qkf: np.ndarray, Rkf: np.ndarray):

        # Estimated filter state and measurements
        self.dx_est, self.dy_est, self.err_cov, self.inn_cov = linearized_ekf_sim.run_linearized_kf(self.truth.x_nom,self.y_true,self.truth.time, dx_est0,P0, self.truth.op.dt, Qkf,Rkf)
        self.x_est = self.dx_est + self.truth.x_nom

        y_nom_stacked = problem_setup.form_stacked_meas_vecs(self.y_true)
        self.y_est = self.dy_est + y_nom_stacked

        # Find error
        self.state_err = self.dx - self.dx_est
        self.meas_res = [np.array([]) for i in self.y_true]
        for t_idx,(y_tr,dy_es) in enumerate(zip(self.y_true_stacked,self.dy_est)):
            if t_idx != 0:
                self.meas_res[t_idx] = np.array(y_tr) - np.array(dy_es)

        # TODO: calcing measurement residuals - go into each time an state and
        # subtract, but keep station id


def run_monte_carlo(tfinal: float, x0_true: np.ndarray, dx0: np.ndarray,P0: np.ndarray,Q:np.ndarray,R:np.ndarray,num_sims: int) -> List[KF_Sim]:
    """Run Monte Carlo simulations."""

    print('Modelling truth...')
    truth = Sim(tfinal)
    truth.sim_truth(x0_true,dx0)
    sims = [[] for i in range(num_sims)]
    for idx in range(num_sims):
        print(f'KF Estimation #{idx+1} out of {num_sims}')
        sims[idx] = KF_Sim(truth,dx0,P0,Q,R)

    return sims


def main():

    op = problem_setup.OdetProblem()
    tfinal = op.T0 * 0.5
    x0_true = op.x0
    dx0 = np.array([10, 0.1, -10, -0.1])
    P0 = np.diag([200, 2, 200, 2])
    Q = 10**-10 * np.diag([1, 1])
    R = op.R

    sims = run_monte_carlo(tfinal, x0_true,dx0,P0,Q,R, 2)

    

    for idx, sim in enumerate(sims):

        # plot states
        fig1, axs1 = plt.subplots(4, 1)
        plotting.states(sim.x_true, sim.truth.time, axs1, 'Truth')
        plotting.states(sim.x_est, sim.truth.time, axs1, 'Est')
        fig1.suptitle(f'States, Est #{idx+1}')
        fig1.tight_layout()

        # plot error
        fig2, axs2 = plt.subplots(4, 1)
        plotting.plot_2sig_err(axs2, sim.state_err, sim.truth.time,sim.err_cov)
        fig2.suptitle(f'2 Sigma Error, Est #{idx+1}')
        fig2.tight_layout()

    

    # NEES and NIS
    consistency_tests.nees_and_nis_test(sims, 0.05)

    plt.show()


if __name__ == "__main__":
    main()

# TODO:
# y true is going to have different station ids from y nom on every run