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
from scipy.stats.distributions import chi2


# TODO: MOVE THIS OBJECT TO THE MONTE CARLO PYTHON FILE WHEN WE MAKE IT.
# could also do a different organization. just need a way to handle N mc runs x n states x T time steps (add a dimension for the err_covs!)

class MC_Sim_Results:
    def __init__(self,x_true,y_true,x_est,y_est,err_cov,inn_cov) -> None:
        """Save MC results to an object."""

        # Ground truth
        self.x_true = x_true
        self.y_true = y_true

        # Estimated filter state and measurements
        self.x_est = x_est
        self.y_est = y_est

        # Filter error covariances
        # err_cov is P+ at every time step
        # inn_cov is H*P-*H^T+R at every time step
        self.err_cov = err_cov
        self.inn_cov = inn_cov
        

def nees_test(sim_objs: List[MC_Sim_Results],alpha: float) -> Tuple[np.ndarray,float,float]:
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
    avg_err_sq_norm = np.zeros(n)
    for idx,sim in enumerate(sim_objs):

        # get normed error squared for current sim
        err = sim.x_true - sim.x_est
        err_sq_norm = sq_weight(err,sim.err_cov)

        # avg into total
        if idx == 0:
            avg_err_sq_norm = err_sq_norm
        else:
            avg_err_sq_norm = (avg_err_sq_norm + err_sq_norm)/2
    
    # calc bounds
    r1 = chi2.ppf(alpha/2,N*n)/N
    r2 = chi2.ppf(1-alpha/2,N*n)/N

    return avg_err_sq_norm, r1, r2

def nis_test(sim_objs: List[MC_Sim_Results],alpha: float) -> Tuple[np.ndarray,float,float]:
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

    # number of measurements
    p = np.size(sim_objs[0].y_true,1)

    # get normed residual squared for each time step avgd over the N sims
    avg_res_sq_norm = np.zeros(p)
    for idx,sim in enumerate(sim_objs):

        # get normed residual squared for current sim
        res = sim.y_true - sim.y_est
        res_sq_norm = sq_weight(res,sim.inn_cov)

        # avg into total
        if idx == 0:
            avg_res_sq_norm = res_sq_norm
        else:
            avg_res_sq_norm = (avg_res_sq_norm + res_sq_norm)/2
    
    # calc bounds
    r1 = chi2.ppf(alpha/2,N*p)/N
    r2 = chi2.ppf(1-alpha/2,N*p)/N

    return avg_res_sq_norm, r1, r2
        

def sq_weight(vec_k: np.ndarray,weight_k:np.ndarray) -> np.ndarray:
    """Find the weighted squared vector.
    
    Args:
        vec_k: Txn array of vectors
        weight_k: Txnxn array of weight matrices

    Returns:
        vec_sq_weight_k: Tx1 array of weighted squares
    """

    vec_sq_weight_k = np.zeros(np.size(vec_k,1))
    for t_idx,(vec,weight) in enumerate(zip(vec_k,weight_k)):
        vec_sq_weight_k[t_idx] = np.transpose(vec) @ weight @ vec
    return vec_sq_weight_k


# TODO:
# better MC sim results saver
# TEST!!!!!!!!!!!!!!!!