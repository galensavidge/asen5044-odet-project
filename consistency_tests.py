"""NEES and NIS tests to check filter consistency.

Goal from tests is to confirm the three characteristics of dynamic KF filter consistency:
    1. unbiased: expected value of the state error is 0 at every time step
    2. efficiency: the state error covariance matches the filter covariance at every time step
    3. measurement residuals are a white Gaussian sequence distributed by the "innovation" covariance matrix (S = H*P-*H^T + R) at every time step

The following tests are to be run on state and output data from N Monte Carlo simulations.

NEES: Normalized Estimation Error Squared
    The NEES test examines the state error at every time step to determine if consistency conditions #1 and #2 hold. The test follows the steps:
        0. Decide significance level alpha (probability of a false positive).
        1. Calc a sample average of the normalized squared state error at each time step.
        2. Calc the chi-squared test bounds r1 and r2 using alpha and N. (scipy.chi2.ppf is inverse cdf)
        3. Plot the sample averages at each time step with r1 and r2. If most of the sample averages lie in the bounds, then the filter is consistent.

NIS: Normalized Innovation Squared
    The NIS test examines the measurement residual at every time step to determine if the consistency condition #3 holds. The test follows the same steps of the NEES test, except using a normalized squared measurement residual instead of state error.
"""

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2

import plotting
from monte_carlo import KF_Sim


def nees_and_nis_test(sim_objs: List[KF_Sim],alpha: float):
    """Perform NEES and NIS tests on MC run outputs and plot results."""   

    # tests
    nees,r1_nees,r2_nees = nees_test(sim_objs,alpha)
    # nis,r1_nis,r2_nis = nis_test(sim_objs,alpha)

    # assume all sims have the same time vector
    time = sim_objs[0].truth.time

    # plot
    fig1,ax1 = plt.subplots(1,1)
    plotting.plot_nees_test(ax1,nees,time,r1_nees,r2_nees)
    # fig2,ax2 = plt.subplots(1,1)
    # plotting.plot_nis_test(ax2,nis,time,r1_nis,r2_nis)
    fig1.tight_layout()
    # fig2.tight_layout()

def nees_test(sim_objs: List[KF_Sim],alpha: float) -> Tuple[np.ndarray,float,float]:
    """Perform the NEES test on MC simulation results.
    
    Args:
        sim_objs: list of objects storing results from MC simulations
        alpha: significance level of test (alpha = 0.05 for a 95% confidence interval)

    Returns:
        avg_err_sq_norm: array of normalized squared state error values averaged over all the simulations
        r1: upper error bound according to alpha
        r2: lower error bound according to alpha
    """

    # number of sims
    N = len(sim_objs)

    # number of states
    n = np.size(sim_objs[0].x_true,1)

    # get normed error squared for each time step avgd over the N sims
    avg_err_sq_norm = 0 
    for idx,sim in enumerate(sim_objs):

        # get normed error squared for current sim
        err_sq_norm = sq_weight(sim.state_err,sim.err_cov)

        # avg into total
        if idx == 0:
            avg_err_sq_norm = err_sq_norm
        else:
            avg_err_sq_norm = (avg_err_sq_norm + err_sq_norm)/2
    
    # calc bounds
    time = sim_objs[0].truth.time
    r1 = chi2.ppf(alpha/2,N*n)/N * np.ones(np.size(time))
    r2 = chi2.ppf(1-alpha/2,N*n)/N *  np.ones(np.size(time))

    return avg_err_sq_norm, r1, r2

def nis_test(sim_objs: List[KF_Sim],alpha: float) -> Tuple[np.ndarray,float,float]:
    """Perform the NIS test on MC simulation results.
    
    Args:
        sim_objs: list of objects storing results from MC simulations
        alpha: significance level of test (alpha = 0.05 for a 95% confidence interval)

    Returns:
        avg_res_sq_norm: array of normalized squared measurement residual values averaged over all the simulations
        r1: upper error bound according to alpha
        r2: lower error bound according to alpha
    """

    # number of sims
    N = len(sim_objs)

    # assume all sims have the same time vector
    time = sim_objs[0].truth.time

    avg_res_sq_norm = np.full(np.size(time),np.nan)
    r1 = np.zeros(np.size(time))
    r2 = np.zeros(np.size(time))
    for t_idx in range(np.size(time)):

        # number of measurements
        num_meas = np.size(sim_objs[0].meas_res[t_idx])
        p = num_meas * 3
        if num_meas == 0:
            continue

        # get normed residual squared for THIS time step avgd over the N sims
        avg_res_sq_norm_tidx = 0
    
        for idx,sim in enumerate(sim_objs):

            # get normed residual squared for current sim
            res = sim.meas_res[t_idx]
            res_sq_norm = np.transpose(res) @ np.linalg.inv(sim.inn_cov[t_idx]) @ res

            # avg into total
            if idx == 0:
                avg_res_sq_norm_tidx = res_sq_norm
            else:
                avg_res_sq_norm_tidx = (avg_res_sq_norm_tidx + res_sq_norm)/2
        
        avg_res_sq_norm[t_idx] = avg_res_sq_norm_tidx
    
        # calc bounds
        r1[t_idx] = chi2.ppf(alpha/2,N*p)/N
        r2[t_idx] = chi2.ppf(1-alpha/2,N*p)/N

    return avg_res_sq_norm, r1, r2
        

def sq_weight(vec_k: np.ndarray,inv_weight_k:np.ndarray) -> np.ndarray:
    """Find the weighted squared vector.
    
    Args:
        vec_k: Txn array of vectors
        inv_weight_k: Txnxn array of the inverse of the weight matrices

    Returns:
        vec_sq_weight_k: Tx1 array of weighted squares
    """

    vec_sq_weight_k = np.zeros(np.size(vec_k,0))
    for t_idx,(vec,weight) in enumerate(zip(vec_k,inv_weight_k)):
        vec_sq_weight_k[t_idx] = np.transpose(vec) @ np.linalg.inv(weight) @ vec
    return vec_sq_weight_k


# TODO:
# NIS when number of measurements is different