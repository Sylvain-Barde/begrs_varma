# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:24:25 2021

This files contains the functions required to simulate the VAR(1) and VARMA(1,1)
specifications.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
from scipy.stats import norm
import sobol

def get_sobol_samples(num_samples, parameter_support, skips):
    """
    Draw multi-dimensional sobol samples

    Arguments:
        num_samples (int):
            Number of samples to draw.
        parameter_support (ndarray):
            A 2D set of bounds for each parameter. Structure is:
                2 x numParams
            with row 0 containing lower bounds, row 1 upper bounds

        skips (int):
            Number of draws to skip from the start of the sobol sequence

    Returns:
        sobol_samples (ndarray):
            A 2D set of Sobol draws. Structure is:
                num_samples x num_param

    """
    params = np.transpose(parameter_support)
    sobol_samples = params[0,:] + sobol.sample(
                        dimension = parameter_support.shape[0],
                        n_points = num_samples,
                        skip = skips
                        )*(params[1,:]-params[0,:])

    return sobol_samples


def varmaSim(C,A,M,S,N,burn):
    """
    Simulate a VARMA(1,1) process

    Note, the dimensions of vector C and matrices A, M and S must match the
    required number of variables V

    Arguments:
        C (ndarray):
            Vector of constants.
            Structure is V x 1
        A (ndarray):
            Matrix of first order auto-regressive parameters.
            Structure is V x V
        M (ndarray):
            Matrix of first-order moving average parameters.
            Structure is V x V
        S (ndarray):
            Variance-covariance matrix for additive noise.
            Structure is V x V
        N (int)
            Number of simulation periods to run
        burn (int)
            Number of burn-in periods to discard.

    Returns
    -------
        x_full (ndarray):
            A 2D array of simulated data. Structure is:
                (N-burn) x V
    """

    num_vars = len(C)
    L,U = np.linalg.eig(S)
    Sroot = np.asmatrix(U)*np.diag(L)**0.5

    e = Sroot*norm.ppf(np.random.rand(num_vars,N+burn))   # Correlated shocks

    x_full = np.zeros([num_vars,N+burn])
    for i in range(1,N+burn):
        x_full[:,i:i+1] = C + A*x_full[:,i-1,None] + e[:,i] + M*e[:,i-1]

    return x_full[:,burn:N+burn]


def varmaPredict(C,A,M,S,y):
    """
    Provide the one-step-ahead prediction of a VARMA(1,1) process given its
    parameters and realisations. This is used as the discepancy function in the
    context of ABC-SMC estimation.

    Note, the dimensions of vector C and matrices A, M, S and y must match the
    required number of variables V

    Arguments:
        C (ndarray):
            Vector of constants.
            Structure is V x 1
        A (ndarray):
            Matrix of first order auto-regressive parameters.
            Structure is V x V
        M (ndarray):
            Matrix of first-order moving average parameters.
            Structure is V x V
        S (ndarray):
            Variance-covariance matrix for additive noise.
            Structure is V x V
        y (ndarray):
            A 2d array of data forming the information set from which to
            generate predictions. Structure is T x V


    Returns:
        y_predict (ndarray):
            A 2D array of predicted values.
            Structure is (T-1) x V
    """

    num_vars = len(C)
    N = y.shape[1]
    L,U = np.linalg.eig(S)
    Sroot = np.asmatrix(U)*np.diag(L)**0.5

    y_predict = np.zeros_like(y)
    e = np.zeros_like(y)
    y_predict[:,0] = y[:,0]
    e[:,0:1] = Sroot*norm.ppf(np.random.rand(num_vars,1))   # Correlated shocks

    for i in range(1,N):
        y_predict[:,i:i+1] = C + A*y[:,i-1,None] + M*e[:,i-1,None]
        e[:,i] = y[:,i] - y_predict[:,i]

    return y_predict
