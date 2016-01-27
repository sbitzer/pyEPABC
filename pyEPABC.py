# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:51:54 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

import math
import numpy as np
from numpy.linalg import slogdet, solve, cholesky, inv, LinAlgError
from scipy.stats import norm
from warnings import warn

def run_EPABC(data, simfun, distfun, prior_mean, prior_cov, epsilon=0.1, 
              npass=2, minacc=300, samplemax=1000000, samplestep=5000, 
              alpha=0.5, veps=1.0, doQMC=True, verbose=1, diagnostics=1):

    # number of sites (data points) x dimensionality of data points
    N, D = data.shape
    # number of parameters
    P = prior_mean.shape[0]

    if np.isscalar(veps):
        veps = np.full(N, veps)

    # initialise approximation with prior
    r, Q = Gauss2exp(prior_mean, prior_cov)
    
    # log normalising constant of the prior, for computing marginal likelihood
    logZprior = logZexp(r, Q)
    
    # site contributions to marginal likelihood, will be filled later
    logC = np.full(N, np.nan)
    
    # site factors of the posterior (one for each data point)
    # Note that I store the factors in C-contiguous order such that slicing 
    # for a factor automatically leads to contiguous locations in memory.
    # It's unclear, though, whether that makes a difference.
    rl = np.zeros((N, P))
    Ql = np.zeros((N, P, P))
    
    # initialise diagnostic variables:
    # number of accepted samples
    nacc = np.zeros((npass, N))
    # total number of samples
    ntotal = np.zeros((npass, N))
    
    # initialise Halton sequence, if desired
    if doQMC:
        # get original Halton sequence in unit-cuboid (uniform in [0,1])
        Halton_seq = create_Halton_sequence(samplestep, P)
        
        # transform to samples from standard normal
        Halton_seq = norm.ppf(Halton_seq)
    else:
        Halton_seq = None
    
    # outer loop over passes
    for p in range(npass):
        # print status information
        if verbose:
            print('pass %d:' % (p,))
        
        # loop over sites (data points)
        for dind in range(N):
            # compute cavity distribution (that without site dind)
            r_c = r - rl[dind, :]
            Q_c = Q - Ql[dind, :, :]
            
            # get Gauss-parameters of cavity distribution
            mean_c, cov_c = exp2Gauss(r_c, Q_c)
            
            # only go on if the covariance of the cavity distribution is 
            # actually positive definite
            try:
                cholesky(cov_c)
            except LinAlgError:
                raise LinAlgError('Covariance of the cavity distribution' + 
                                  'for site %d\n' % (dind,) + 
                                  'is probably not positive definite' +
                                  'in pass %d!' % (p,))
            else:
                # allocate memory to store accepted simulations, you only need
                # minacc + samplestep elements, because the loop breaks once 
                # minacc is reached
                samples = np.full((minacc + samplestep, P), np.nan)
                
                # loop for simulations
                for s in range(math.ceil(samplemax / samplestep)):
                    # determine how many samples you need to get
                    S = np.min(samplemax - s * samplestep, samplestep)
                    
                    # sample from cavity distribution
                    parsamples = mvnrand(mean_c, cov_c, S, Halton_seq)
                    
                    # simulate from model with sampled parameters
                    sims = simfun(parsamples, dind)
                    
                    # get distances between simulated and real data
                    dists = distfun(data[dind, :], sims)
                    
                    # find accepted samples
                    accind = dists < epsilon
                    naccnew = nacc[p, dind] + np.sum(accind)
                    if nacc[p, dind] < naccnew:
                        samples[nacc[p, dind]:naccnew, :] = \
                            parsamples[accind, :]
                        nacc[p, dind] = naccnew
                    
                    # break if enough accepted samples
                    if nacc[p, dind] >= minacc:
                        break
                
                samples = samples[:nacc[p, dind], :]
                ntotal[p, dind] = np.min(samplemax, (s+1) * samplestep)
                
                # get mean and covariance of accepted samples
                mean_new = np.mean(samples, axis=0)
                cov_new = np.cov(samples, rowvar=0)
                
                if nacc < minacc:
                    warn('The minimum number of accepted samples was not ' + 
                         'reached for:\nsite %d in pass %d\n' % (dind, p) + 
                         'Continuing anyway, but checking for positive ' + 
                         'definiteness\nof estimated covariance. Error ' + 
                         'may follow.', RuntimeWarning)
                    # raises LinAlgError if cov_new is not positive definite
                    cholesky(cov_new)
                         
                # get new natural parameters
                r_new, Q_new = Gauss2exp(mean_new, cov_new)
                
                # partially update hybrid distribution parameters
                Q = alpha * Q_new + (1-alpha) * Q;
                r = alpha * r_new + (1-alpha) * r;
                
                # update log normalisation constant for site
                # THERE MIGHT BE SOME ISSUE WITH TRANSPOSES HERE: validate that r is 1D!
                accratio = nacc[p, dind] / ntotal[p, dind]
                logC[dind] = ( math.log(accratio) - logZexp(r, Q) + 
                               logZexp(r_c, Q_c) );
                
                # update current site
                rl[dind, :] = r - r_c;
                Ql[dind, :, :] = Q - Q_c;
                
            if verbose and ( math.floor((dind-1) / N * 100) < 
                             math.floor(dind / N * 100) ):
                print('%3d%% completed' % math.floor(dind / N * 100));

    mean_pos, cov_pos = exp2Gauss(r, Q)
    
    logml = np.sum(logC) + logZexp(r, Q) - logZprior - np.sum(np.log(veps));
    
    return mean_pos, cov_pos, logml

def Gauss2exp(mean, cov):
    Q = inv(cov)
    
    return np.dot(Q, mean), Q

def exp2Gauss(r, Q):
    cov = inv(Q)
    
    return np.dot(cov, r), cov

def logZexp(r, Q):
    sd, logd = slogdet(Q)
    if sd != 1:
        raise LinAlgError('Q is not positive definite!')
    
    D = r.shape[0]
    
    return D / 2 * math.log(2 * math.pi) - logd / 2 + np.dot(r, solve(Q, r)) / 2
    
    
def mvnrand(mu, cov, S, Halton_seq=None):
    if Halton_seq is None:
        # use Numpy built-in
        return np.random.multivariate_normal(mu, cov, S)
    else:
        # sample manually
        samples = Halton_seq[:S, :len(mu)]
        return np.dot(samples, cholesky(cov).T) + mu
    
def cholinv(A):
    """
    Supposedly this a faster alternative to 'inv' for positive definite 
    matrices, but at least here, with Numpy, this is not true. So this function
    is unnecessary here, but I keep it, because it may perhaps become useful
    some time.
    """
    
    L = cholesky(A)
    
    return solve(L.T, solve(L, np.eye(L.shape[0])))


def create_Halton_sequence(N, D):
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                       53, 59, 61, 67, 71, 73, 79, 83, 89, 97]);
                       
    if len(primes) < D:
        raise Exception('Halton sequence creation failed, because the ' + 
                        'requested\ndimensionality exceeds the available ' +
                        'number of primes.')
    else:
        Halton_seq = np.zeros((N, D))
        for d in range(D):
            Halton_seq[:, d] = create_vdCorput_sequence(N, primes[d], 
                                                          zero=False)
                                                          
    return Halton_seq
                                                          
def create_vdCorput_sequence(N, base, zero=True):
    """
    Creates a low-discrepancy van der Corput sequence (over the unit interval).
    
    This is a Python re-implementation of a corresponding Matlab function by
    Dimitri Shvorob, dimitri.shvorob@vanderbilt.edu, 6/20/07
    """
    if N % 1 > 0 or N < 0:
        raise ValueError('N must be a non-negative integer')
    if base % 1 > 0 or base < 2:
        raise ValueError('base must be a non-negative integer greater than 1')
        
    seq = np.zeros(N)
    for n in range(N):
        a = basexpflip(n+1, base)
        g = base ** np.arange(1, len(a)+1)
        seq[n] = np.sum(a / g)
        
    return seq
        
def basexpflip(k, base):
    """
    reversed expansion of positive integer k with given base
    
    part of create_vdCorput_sequence
    """
    j = int(np.fix( math.log(k) / math.log(base) ) + 1)
    a = np.zeros(j)
    q = base ** (j - 1)
    
    for i in range(j):
        a[i] = math.floor(k / q)
        k = k - q * a[i]
        q = q / base
        
    return a[::-1]