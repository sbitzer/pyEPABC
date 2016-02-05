# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 17:45:29 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

import sys
sys.path.append( ".." )

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from math import log
import matplotlib.pyplot as plt
from pyEPABC import run_EPABC
import testDDM_C

# turn this off, if you're not interested
compare_to_hddm = True

if compare_to_hddm:
    import hddm


def get_hddm_prior_samples(model, N, parnames=['v', 'a', 'z', 't', 'st']):
    # initialise pyMC, if not done already
    if model.mc is None:
        model.mcmc()
    
    samples = pd.DataFrame([], columns=(parnames+['distribution']))
    
    for n in range(N):
        samples.loc[n, 'distribution'] = 'hddm_prior'
        
        sample = model.draw_from_prior(update=True)
        for name in parnames:
            samples.loc[n, name] = sample[name]
            
    return samples


def get_hddm_posterior_samples(model, parnames=['v', 'a', 'z', 't', 'st']):
    samples = pd.DataFrame([], columns=(parnames+['distribution']))
    
    for name in parnames:
        samples.loc[:, name] = model.nodes_db.loc[name]['node'].trace()
        
    samples['distribution'] = 'hddm_pos'
        
    return samples
    
    
def plot_param_dist(samples, parnames, axlim_q=1):
    pg = sns.PairGrid(samples, hue='distribution', diag_sharey=False)
    pg.map_diag(sns.kdeplot, shade=True)
    pg.map_offdiag(plt.scatter, alpha=0.3)
    
    # adjust axis limits distorted by kdeplot
    for p, name in enumerate(parnames):
        m = samples[name].quantile(axlim_q)
        lims = [samples[name].min() - 0.1 * m, 1.1 * m]
        pg.axes[0, p].set_xlim(lims)
        if p == 0:
            pg.axes[p, 1].set_ylim(lims)
        else:
            pg.axes[p, 0].set_ylim(lims)
            
    pg.add_legend(frameon=True)
    
    return pg
    

def response_dist(data, sims):
    """
    Simple distance between responses.
    
    Is infinite, if choices don't match, else it's the absolute difference 
    in RTs.
    
    data: vector with data[0]=response, data[1]=RT
    sims: array with N responses in rows, i.e., sims.shape = [N, 2]
    """
    matchind = sims[:, 0] == data[0]
    
    dists = np.full(sims.shape[0], np.inf)
    
    dists[matchind] = np.abs(sims[matchind, 1] - data[1])
    
    return dists


if __name__ == "__main__":
    # load data
    data = pd.read_csv('responsedata.csv')
    
    parnames = ['v', 'a', 'z', 't', 'st']

    # number of samples to plot later
    N = 500
    
    # initialise samples
    samples = pd.DataFrame([], columns=(parnames+['distribution']))
    
    # set to True if you want to generate HDDM results
    if compare_to_hddm:
        # transforming into hddm format
        hddmdata = data.values
        hddmdata[:, :2] = hddmdata[:, :2] - 1
        hddmdata = pd.DataFrame(hddmdata, columns=['stim', 'response', 'rt'])
        
        # specify HDDM model and parameters to test
        model = hddm.HDDMStimCoding(hddmdata, stim_col='stim', split_param='v', 
                                    include=('z', 'st'))
        
        samples = samples.append(get_hddm_prior_samples(model, N, parnames), 
                       ignore_index=True)
        
        #model.find_starting_values()
        S = 2000
        B = 20
        model.sample(S, burn=B)
        
        hddmsamples = get_hddm_posterior_samples(model, parnames)
        samples = samples.append(hddmsamples.iloc[np.linspace(0, S-B-1, N, 
            dtype=int)], ignore_index=True)
        
    # fit only v, a, z, t, st
    # implement constraints through transformations
    # v, a, ndt, sndt are positive, z is in [0, 1]
    paramtransform = lambda params: np.c_[np.exp(params[:, :2]), 
                                          norm.cdf(params[:, 2]), 
                                          np.exp(params[:, 3:])]
    
    # these define the priors, values are chosen such that the means of the 
    # priors roughly correspond to those in HDDM, I also tried to approximately 
    # match the spreads, but the underlying distributions differ as can be seen
    # in the generated plot, I also deliberately chose wider priors for a and t
    # as especially the HDDM prior for t appeared to me very narrow
    # NOTE: the HDDM prior shows a correlation between t and st, I should check
    # where this comes from
    prior_mean = np.array([log(2), log(1.5), 0, log(0.4), log(0.15)])
    prior_cov = np.diag(np.array([1, 1, 0.3, 1, 1.3]) ** 2)
    
    # sample from EPABC prior
    samples_pr = np.random.multivariate_normal(prior_mean, prior_cov, N)
    samples_pr = pd.DataFrame(paramtransform(samples_pr), columns=parnames)
    samples_pr['distribution'] = 'epabc_prior'
    samples = samples.append(samples_pr, ignore_index=True)
    
    # plot the prior(s)
    pg_prior = plot_param_dist(samples[samples['distribution'].map(
        lambda x: x.endswith('prior'))], parnames, axlim_q=0.98)
    
    # the C-function expects all 8 parameters, so fill up with default values
    fillpar = lambda parsamples: np.c_[parsamples[:, 0], 
                                       np.zeros((parsamples.shape[0], 1)),
                                       parsamples[:, 1:3],
                                       np.zeros((parsamples.shape[0], 1)),
                                       parsamples[:, 3:],
                                       np.ones((parsamples.shape[0], 1))]

    # wrapper to C-function for sampling from DDM
    simfun = lambda parsamples, dind: testDDM_C.sample_from_DDM(
        fillpar(paramtransform(parsamples)), dind, 
        data.values[:, 0].astype(dtype=int))
    
    # run EPABC
    ep_mean, ep_cov, ep_logml, nacc, ntotal = run_EPABC(data.values[:, 1:], 
        simfun, response_dist, prior_mean, prior_cov, epsilon=0.05, 
        minacc=500, samplestep=10000, samplemax=2000000, npass=3, alpha=0.3, 
        veps=1)
    
    # sample from EPABC posterior
    samples_pos = np.random.multivariate_normal(ep_mean, ep_cov, N)
    samples_pos = pd.DataFrame(paramtransform(samples_pos), columns=parnames)
    samples_pos['distribution'] = 'epabc_pos'
    samples = samples.append(samples_pos, ignore_index=True)
        
    # plot the posterior(s)
    pg_pos = plot_param_dist(samples[samples['distribution'].map(
        lambda x: x.endswith('pos'))], parnames)
        
    plt.show()