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

from typing import List

import numpy as np


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
        

def nees_test(sim_objs = List[MC_Sim_Results]):
    """Perform the NEES test on MC simulation results."""

    # number of sims
    N = len(sim_objs)

    avg_err_sq_norm = np.zeros()
    for idx,sim in enumerate(sim_objs):
        err = sim.x_true - sim.x_est
        err_sq_norm = sq_weight(err,sim.err_cov)
    
    # TODO:
    # average err_sq_norm for all sims
    # get r1 and r2
    # return relevant info
        

def sq_weight(vec_k: np.ndarray,weight_k:np.ndarray) -> np.ndarray:
    """Find the weighted squared vector.
    
    Args:
        vec_k: Txn array of vectors
        weight_k: Txnxn array of weight matrices

    Returns:
        vec_sq_weight_k: Txn array of weighted squares
    """

    #TODO: test this

    vec_sq_weight_k = np.zeros(np.size(vec_k))
    for t_idx,(vec,weight) in enumerate(zip(vec_k,weight_k)):
        vec_sq_weight_k[t_idx,:] = np.transpose(vec) @ weight @ vec
    return vec_sq_weight_k

# TODO:
# NIS test
# plotting functions
# better MC sim results saver