# -*- coding: utf-8 -*-

# estimating the mean of a Gaussian from independent data given a Gaussian 
# prior on the mean
import sys
sys.path.append( ".." )

import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal as mvnorm
from scipy.linalg import block_diag
from scipy.special import gamma
from pyEPABC import run_EPABC

if __name__ == "__main__":
    D = 4
    
    prior_mean = np.zeros(D)
    prior_cov = np.diag(np.full(D, 4.5 ** 2))
    prior_prec = inv(prior_cov)
    
#    lik_cov = np.diag(np.random.rand(D) * 5)
    lik_cov = np.diag(np.ones(D) * (4.0 ** 2))
    lik_prec = inv(lik_cov)
    
    # sample mean from prior
    lik_mean = np.random.multivariate_normal(prior_mean, prior_cov)
    
    # sample some data
    N = 100
    data = np.random.multivariate_normal(lik_mean, lik_cov, N)
    
    # compute analytic posterior
    pos_cov = inv( prior_prec + N * lik_prec )
    pos_mean = np.dot(pos_cov, np.dot(prior_prec, prior_mean) + 
                               np.dot(lik_prec, np.sum(data, axis=0)))
                        
    # compute analytic log-marginal likelihood
    A = np.kron(np.ones((N, 1)), np.eye(D))
    ml_cov = block_diag(*([lik_cov] * N)) + np.dot(A, np.dot(prior_cov, A.T))
    ml_mean = np.dot(A, prior_mean)
    logml = mvnorm.logpdf(data.flatten(), ml_mean, ml_cov)
    
    # sample from multivariate normal with different means
    simfun = lambda mean, ind: np.random.multivariate_normal(
        np.zeros(mean.shape[1]), lik_cov, mean.shape[0]) + mean
    
    # Euclidean distance to use within EPABC
    distfun = lambda dat, sims: np.linalg.norm(dat - sims, axis=1)
    
    # allowable distance
    epsilon = 2
    
    # this is a correction term used for the computation of the marginal
    # likelihood; it is the normalising constant of a uniform distribution
    # defined by the distance and epsilon, for the Euclidean distance above it
    # is (according to Barthelme2014):
    veps = np.pi ** (D/2) / gamma(D/2 + 1) * epsilon ** D;
    
    ep_mean, ep_cov, ep_logml, nacc, ntotal = run_EPABC(data, simfun, distfun, 
        prior_mean, prior_cov, epsilon=epsilon, minacc=500, samplestep=100000, 
        samplemax=20000000, npass=4, alpha=0.3, veps=veps)