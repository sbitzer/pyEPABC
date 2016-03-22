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
from datetime import datetime

def run_EPABC(data, simfun, distfun, prior_mean, prior_cov, epsilon, 
              npass=2, minacc=300, samplemax=1000000, samplestep=5000, 
              alpha=1.0, veps=1.0, doQMC=True, verbose=1):
    """
    runs EP-ABC 
    (likelihood-free, probabilistic inference based on expectation propagation)
    
    Parameters
    ----------
    data : 2D numpy array; N, D = shape
        observed data where N is the number of data points and D is their 
        dimensionality
    simfun : function, call signature: `sims` = simfun(`parsamples`, `dind`)
        returns simulations from model
        
        `parsamples` is a 2D numpy array of parameter samples (NP, P = shape 
        where NP is the number of samples and P is the number of parameters)
        
        `dind` is an int scalar which indexes data and allows data 
        point-specific simulations from the model
        
        `sims` is a 2D numpy array containing simulated data, there will be 
        one simulated data point per parameter sample (NP, D = shape)
    distfun : function, call signature: `dists` = distfun(`datapoint`, `sims`)
        computes distances in data space
        
        `datapoint` is a 1D numpy array sliced from `data` (D = shape)
        
        `sims` is a 2D numpy array as returned from `simfun` (NP, D = shape)
        
        `dists` is a 1D numpy array containing computed distances between
        `datapoint` and `sims` (NP = shape)
    prior_mean : 1D numpy array; P = shape
        mean of Gaussian prior over parameters
    prior_cov : 2D numpy array; P, P = shape
        covariance matrix of Gaussian prior over parameters
    epsilon : float
        maximum distance for which parameter samples will be accepted
    npass : int, optional
        number of passes through data before returning posterior
    minacc : int, optional
        number of accepted samples that needs to be reached before updating 
        the posterior in each step
    samplemax : int, optional
        maximum number of samples to be generated in each step before updating 
        the posterior even though `minacc` was not reached
    
    Returns
    -------
    mean_pos : 1D numpy array; P = shape
        mean of Gaussian posterior over parameters
    cov_pos : 2D numpy array; P, P = shape
        covariance matrix of Gaussian posterior over parameters
    logml : float
        log marginal likelihood (aka. log model evidence)
    nacc : 2D numpy array; `npass`, N = shape
        number of accepted samples for each step
    ntotal : 2D numpy array; `npass`, N = shape
        total number of samples for each step
    runtime : timedelta
        total runtime of function as timedelta object
        
    Other Parameters
    ----------------
    samplestep : int, optional
        size of sample batches, internally NP = `samplestep`
    alpha : float, 0 < `alpha` <= 1, optional
        partial update parameter for greater numerical stability, for 
        `alpha` = 1.0 you do standard, full EP updates
    veps : float or 1D numpy array with N = shape
        correction term(s) for marginal likelihood
        
        this is the normalising constant of the uniform distribution within 
        the region of data space in which simulated data points are accepted,
        further info in Notes
    doQMC : bool, optional
        whether to use quasi-monte carlo sampling of parameters based on a 
        Halton sequence, improves numerical stability
    verbose : bool, optional
        whether to print progress counter
        
    Raises
    ------
    LinAlgError
        if covariance matrices estimated from accepted parameter samples are 
        not positive definite
        
    Notes
    -----
    EP-ABC is defined and explained in [1]_. This is a simplified 
    implementation that steps through each data point individually, i.e., 
    the algorithm in [1]_ is not implemented here in its full generality. 
    
    The basic idea is to assume that the posterior distribution over parameters
    factorises into one Gaussian factor per data point (this is the expectation 
    propagation part) and that you can estimate these factors by simulating 
    data from your model (this is the likelihood-free, or ABC part) with 
    samples from an intermediate parameter distribution. The crucial point is 
    to only accept samples for estimation of the factor for which the simulated
    data was close to the corresponding observed data point. Closeness is 
    determined by the given `distfun` and `epsilon`.
    
    So EP-ABC makes two approximations:
    
    1. factorised Gaussian posterior and 
    2. sampling estimates of the likelihood. 
    
    With the parameters of this function you can tune 2. The theory states that 
    for vanishing `epsilon` you get close to the true posterior which here 
    means that you come close to the analytic, factorised Gaussian posterior 
    that you could get, if you knew the likelihood of the model. However, you 
    will also decrease the acceptance ratio, when decreasing `epsilon`. This 
    is bad for two reasons: 
    
    - The algorithm will become very slow, because you will need to generate
      millions of samples for each data point.
    - Experience has shown that for few accepted samples EP-ABC tends to 
      underestimate the variance of the posterior. This may become so severe 
      that the posterior becomes very narrow around unrealistic parameter 
      values.
    
    To prevent the latter from happening you should try to increase the number
    of accepted samples (for each data point) by increasing `minacc` and 
    `samplemax`, as necessary. The appropriate `minacc` is problem-specific. 
    Another useful option is to use partial updates with, e.g., 
    ``alpha = 0.5``. This prevents premature convergence, but may also just 
    delay the unreasonable narrowing of the posterior into later passes through 
    the data. You can also increase `epsilon` again, if the acceptance ratio is
    very low, but you risk worse approximations of the posterior.
    
    Notice that distances can become very large in high-dimensional spaces. 
    This is the curse of dimensionality and means that the acceptance ratio 
    will heavily drop when the dimensionality increases. You can test this in 
    the Gaussian example mentioned below. You may increase `epsilon` to 
    increase the acceptance ratio, but beware again that the quality of the 
    approximation may deteriorate.
    
    The likelihood-free setting means that the estimate of the marginal 
    likelihood provided by the expectation propagation part needs to be 
    corrected by `veps` (see [1]_ for details). `veps` is the normalising 
    constant of the uniform distribution within the region of data space in 
    which simulated data points are accepted. For Euclidean distances this is 
    the volume of a Euclidean ball with radius `epsilon` which is
    
    .. math:: v_\epsilon = \epsilon^D \pi^{D/2} / \Gamma(D/2 + 1).
    
    It is essential for efficient sampling that the parameters are 
    appropriately constrained. Yet, run_EPABC expects relatively unconstrained 
    Gaussian priors for the parameters. You can, however, still implement 
    parameter constraints inside `simfun` by appropriate transformation of 
    parameter values. For example, you can constrain parameters to be positive 
    by exponentiating, or you can restrict the range of parameter values by 
    transforming through a Gaussian cumulative density function through which 
    you can also define uniform priors over parameters. This is demonstrated in 
    the drift diffusion model example.
    
    Examples
    --------
    There are two examples in the corresponding subdirectory demonstrating the 
    usage of pyEPABC with a trivial Gaussian example and a more involved 
    example using a drift diffusion model.
    
    References
    ----------
    .. [1] Barthelmé, S. & Chopin, N. Expectation-Propagation for 
       Likelihood-Free Inference. Journal of the American Statistical 
       Association, 2014, 109, 315-333, 
       https://doi.org/10.1080/01621459.2013.864178
    """


    # record start time
    starttime = datetime.now()

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
        if verbose:
            print('Creating Gaussian Halton sequence ... ', end='', flush=True)
        
        # get original Halton sequence in unit-cuboid (uniform in [0,1])
        Halton_seq = create_Halton_sequence(samplestep, P)
        
        # transform to samples from standard normal
        Halton_seq = norm.ppf(Halton_seq)
        
        if verbose:
            print('done.')
    else:
        Halton_seq = None
    
    # outer loop over passes
    for p in range(npass):
        
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
                raise LinAlgError('Covariance of the cavity distribution ' + 
                                  'for site %d ' % (dind,) + 
                                  'is probably not positive definite ' +
                                  'in pass %d!' % (p,))
            else:
                # allocate memory to store accepted simulations, you only need
                # minacc + samplestep elements, because the loop breaks once 
                # minacc is reached
                samples = np.full((minacc + samplestep, P), np.nan)
                
                # loop for simulations
                for s in range(math.ceil(samplemax / samplestep)):
                    # determine how many samples you need to get
                    S = np.min([samplemax - s * samplestep, samplestep])
                    
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
                ntotal[p, dind] = np.min([samplemax, (s+1) * samplestep])
                
                if nacc[p, dind] < P:
                    warn('Skipping site %d in pass %d, ' % (dind, p) + 
                         'because the number of accepted samples was ' + 
                         'smaller than the number of parameters.')
                else:
                    # get mean and covariance of accepted samples
                    mean_new = np.mean(samples, axis=0)
                    cov_new = np.cov(samples, rowvar=0, ddof=1.5)
                    
                    if nacc[p, dind] < minacc:
                        warn('The minimum number of accepted samples was not ' + 
                             'reached for site %d in pass %d (%d accepted). ' % (dind, p+1, nacc[p, dind]) + 
                             'Continuing anyway, but checking for positive ' + 
                             'definiteness of estimated covariance. Error ' + 
                             'may follow.', RuntimeWarning)
                        # raises LinAlgError if cov_new is not positive definite
                        cholesky(cov_new)
                             
                    # get new natural parameters
                    r_new, Q_new = Gauss2exp(mean_new, cov_new)
                    
                    # partially update hybrid distribution parameters
                    Q = alpha * Q_new + (1-alpha) * Q;
                    r = alpha * r_new + (1-alpha) * r;
                    
                    # update log normalisation constant for site
                    accratio = nacc[p, dind] / ntotal[p, dind]
                    logC[dind] = ( math.log(accratio) - logZexp(r, Q) + 
                                   logZexp(r_c, Q_c) );
                    
                    # update current site
                    rl[dind, :] = r - r_c;
                    Ql[dind, :, :] = Q - Q_c;
                
            # print status information
            if verbose and ( math.floor(dind / N * 100) < 
                             math.floor((dind+1) / N * 100) ):
                print('\rpass %d/%d: %3d%% completed' % (p+1, npass, 
                    math.floor((dind+1) / N * 100)), end='');
                
        if verbose:
            # finish line by printing \n
            print('')

    mean_pos, cov_pos = exp2Gauss(r, Q)
    
    logml = np.sum(logC) + logZexp(r, Q) - logZprior - np.sum(np.log(veps));
    
    runtime = datetime.now() - starttime
    if verbose:
        print('elapsed time: ' + runtime.__str__())
    
    return mean_pos, cov_pos, logml, nacc, ntotal, runtime

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