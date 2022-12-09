"""Monte Carlo simulations."""

from typing import List

import numpy as np
import matplotlib.pyplot as plt

import problem_setup
import nonlinear_sim 
import plotting
import consistency_tests

class Sim:
    def __init__(self,tfinal: float):
        # common parameters
        self.op = problem_setup.OdetProblem()

        # Time vector
        self.tfinal = tfinal
        self.time = np.arange(0, tfinal, self.op.dt)
        self.T = np.size(self.time)

    def sim_truth(self,x0: np.ndarray):
        """Integrate nonlinear system with noise."""

        # Integrate state
        proccess_noise = problem_setup.form_process_noise(self.T,self.op.W)
        _, self.x_true = nonlinear_sim.integrate_nl_ct_eom(x0,self.op.dt,self.tfinal,proccess_noise)

        # create measurements
        self.station_ids_list = [problem_setup.find_visible_stations(x, t) for x, t in zip(self.x_true, self.time)]
        self.y_true = problem_setup.states_to_noisy_meas(self.x_true, self.time,self.station_ids_list,self.op.R)

class KF_Sim():
    def __init__(self, truth_sim: Sim):
        # TODO fill in

        self.truth = truth_sim

        # Estimated filter state and measurements
        self.x_est = np.zeros_like(self.truth.x_true)

        # Filter error covariances
        # err_cov is P+ at every time step
        # inn_cov is H*P-*H^T+R at every time step
        self.err_cov = np.zeros((self.truth.T,np.size(self.truth.x_true,1),np.size(self.truth.x_true,1)))

        y_stacked = problem_setup.form_stacked_meas_vecs(self.truth.y_true)

        self.y_est = [[] for y in self.truth.y_true]
        self.inn_cov =  [[] for y in self.truth.y_true]
        for idx,y in enumerate(self.truth.y_true):
            self.y_est[idx] = [np.zeros(np.size(meas)) for meas in y]
            self.inn_cov[idx] = np.eye(len(y_stacked[idx]))

        # Find error
        self.state_err = self.truth.x_true - self.x_est
        self.meas_res = y_stacked

        # TODO: calcing measurement residuals - go into each time an state and subtract, but keep station id

def run_monte_carlo(tfinal:float,x0_true: np.ndarray,num_sims:int)-> List[KF_Sim]:
    """Run Monte Carlo simulations."""

    truth = Sim(tfinal)
    truth.sim_truth(x0_true)
    sims = [[] for i in range(num_sims)]
    for idx in range(num_sims):
        sims[idx] = KF_Sim(truth)
   
    return sims

def main():

    op = problem_setup.OdetProblem()
    tfinal = op.T0 * 3
    x0_true = op.x0

    sims = run_monte_carlo(tfinal,x0_true,2)

    # plot states
    fig1,axs1 = plt.subplots(4,1)

    for idx,sim in enumerate(sims):
        if idx == 0:
            plotting.states(sim.truth.x_true,sim.truth.time,axs1,'Truth')
        plotting.states(sim.x_est,sim.truth.time,axs1,f'Est #{idx}')
        
        # plot error
        fig2,axs2 = plt.subplots(4,1)
        plotting.plot_2sig_err(axs2,sim.state_err,sim.truth.time,sim.err_cov)
        fig2.suptitle('2 Sigma Error')

    fig1.suptitle('States')
    fig1.tight_layout()

    # NEES and NIS
    consistency_tests.nees_and_nis_test(sims,0.05)

    plt.show()

if __name__ == "__main__":
    main()
        

# TODO:
# perturbation info for linearized KF
