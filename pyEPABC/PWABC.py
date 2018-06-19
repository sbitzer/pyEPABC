# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:57:42 2016
"""
from __future__ import print_function, division

import math
import numpy as np
from datetime import datetime
from scipy.stats import norm
from numpy.linalg import cholesky, inv, slogdet
from warnings import warn
from .EPABC import create_Halton_sequence

def run_PWABC(data, simfun, distfun, prior_mean, prior_cov, epsilon,
              minacc=300, samplemax=1000000, samplestep=5000,
              veps=1.0, doQMC=True, verbose=1):
    """
    runs PW-ABC - piecewise approximate Bayesian computation, [1]_
    (likelihood-free, probabilistic inference based on a Gaussian factorisation
     of the posterior)

    References
    ----------
    .. [1] White, S. R.; Kypraios, T. & Preston, S. P. Piecewise Approximate
       Bayesian Computation: fast inference for discretely observed Markov
       models using a factorised posterior distribution.
       Statistics and Computing, 2013, 25, 289-301,
       https://doi.org/10.1007/s11222-013-9432-2
    """

    # record start time
    starttime = datetime.now()

    # number of sites (data points) x dimensionality of data points
    N, D = data.shape
    # number of parameters
    P = prior_mean.shape[0]

    if np.isscalar(veps):
        veps = np.full(N, veps)

    # initialise diagnostic variables:
    # number of accepted samples
    nacc = np.zeros(N)
    # total number of samples
    ntotal = np.zeros(N)

    # get upper triangular form of prior covariance matrix
    prior_cov_R = cholesky(prior_cov).T

    # initialise Halton sequence, if desired
    if doQMC:
        if verbose:
            print('Creating Gaussian Halton sequence ... ', end='', flush=True)

        # get original Halton sequence in unit-cuboid (uniform in [0,1])
        # we will reuse the same Halton sequence in each sampling step
        Halton_seq = create_Halton_sequence(samplestep, P)

        # transform to samples from standard normal
        Halton_seq = norm.ppf(Halton_seq)

        if verbose:
            print('done.')

        # use Halton sequence to get quasi Monte Carlo samples from the prior
        Halton_samples = np.dot(Halton_seq, prior_cov_R) + prior_mean
    else:
        Halton_samples = None
        # create sample store (parameter samples can be reused across data points)
        parsamples = np.full((samplemax, N), np.nan)

    # number of already stored batches of parameter samples (in sample steps)
    NS = 0

    # mean and covariance of posterior data factors
    means = np.full((N, P), np.nan)
    covs = np.full((N, P, P), np.nan)
    precs = np.full((N, P, P), np.nan)

    # intermediate sum of estimated precisions
    Binv = np.zeros((P, P))
    # intermediate sum of precision-scaled means
    atmp = np.zeros(P)

    # loop over data points
    for dind in range(N):
        # allocate memory to store accepted simulations, you only need
        # minacc + samplestep elements, because the loop breaks once
        # minacc is reached
        samples = np.full((minacc + samplestep, P), np.nan)

        # loop over simulations
        for s in range(math.ceil(samplemax / samplestep)):
            # determine how many samples you need to get
            S = np.min([samplemax - s * samplestep, samplestep])

            if doQMC:
                # just use samples from Halton sequence
                pars = Halton_samples[:S, :]
            elif s < NS:
                # reuse parameter samples from previous data points
                pars = parsamples[s * samplestep : s * samplestep + S]
            else:
                # get fresh samples from prior
                pars = np.random.normal(size=(S, P))
                pars = np.dot(pars, prior_cov_R) + prior_mean

                # store them for later reuse
                parsamples[s * samplestep : s * samplestep + S] = pars

            # simulate from model with sampled parameters
            sims = simfun(pars, dind)

            # get distances between simulated and real data
            dists = distfun(data[dind, :], sims)

            # find accepted samples
            accind = dists < epsilon
            naccnew = nacc[dind] + np.sum(accind)
            if nacc[dind] < naccnew:
                samples[nacc[dind]:naccnew, :] = pars[accind, :]
                nacc[dind] = naccnew

            # break if enough accepted samples
            if nacc[dind] >= minacc:
                break

        samples = samples[:nacc[dind], :]
        ntotal[dind] = np.min([samplemax, (s+1) * samplestep])

        if nacc[dind] < P:
            warn('Skipping site %d, ' % dind +
                 'because the number of accepted samples was ' +
                 'smaller than the number of parameters.')
        else:
            # get mean and covariance of accepted samples
            means[dind, :] = np.mean(samples, axis=0)
            covs[dind, :, :] = np.cov(samples, rowvar=0)

            if nacc[dind] < minacc:
                warn('The minimum number of accepted samples was not ' +
                     'reached for site %d (%d accepted). ' % (dind, nacc[dind]) +
                     'Continuing anyway, but checking for positive ' +
                     'definiteness of estimated covariance. Error ' +
                     'may follow.', RuntimeWarning)
                # raises LinAlgError if cov_new is not positive definite
                cholesky(covs[dind, :, :])

            if dind >= 1:
                precs[dind, :, :] = inv(covs[dind, :, :])
                Binv += precs[dind, :, :]
                atmp += np.dot(precs[dind, :, :], means[dind, :])

        # print status information
        if verbose and ( math.floor(dind / N * 100) <
                         math.floor((dind+1) / N * 100) ):
            print('\r%3d%% completed' % math.floor((dind+1) / N * 100), end='');

    if verbose:
        # finish line by printing \n
        print('')

    # compute posterior covariance (Eq. 20 in White et al.)
    prior_prec = inv(prior_cov)
    post_cov = inv((2-N) * prior_prec + Binv)
    # compute posterior mean (Eq. 21 in White et al.)
    post_mean = np.dot(post_cov, (2-N) * np.dot(prior_prec, prior_mean) + atmp)
    
    # compute log marginal likelihood
    B = inv(Binv)
    a = np.dot(B, atmp)
    logml = np.sum(np.log(nacc[1:] / ntotal[1:])) - np.sum(np.log(veps))
    for dind in range(1, N):
        sd, logd = slogdet(covs[dind, :, :])
        logml -= 0.5 * logd
        for dind2 in range(dind+1, N):
            mdiff = means[dind, :] - means[dind2, :]
            logml -= 0.5 * np.dot(np.dot(np.dot(np.dot(mdiff, 
                precs[dind, :, :]), B), precs[dind2, :, :]), mdiff)
    sd, logd = slogdet(post_cov)
    logml += 0.5 * logd
    sd, logd = slogdet(prior_cov)
    logml += (N - 2) / 2 * logd
    mdiff = a - prior_mean
    logml -= 0.5 * np.dot(np.dot(mdiff, inv(prior_cov / (2 - N) + B)), mdiff)

    runtime = datetime.now() - starttime
    if verbose:
        print('elapsed time: ' + runtime.__str__())

    return post_mean, post_cov, logml, nacc, ntotal, runtime