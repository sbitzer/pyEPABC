# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal as mvnorm
from scipy.linalg import block_diag
from scipy.special import gamma
from pyEPABC import run_EPABC
import pandas as pd
import matplotlib.pyplot as plt

def plot_cov_ellipse(cov, pos, volume=.95, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
            
    This function was originally written by Noah H. Silbert, see
    http://www.nhsilbert.net/source/2014/06/bivariate-normal-ellipse-plotting-in-python/
    """

    from scipy.stats import chi2
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)


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
    epsilon = 3
    
    # this is a correction term used for the computation of the marginal
    # likelihood; it is the normalising constant of a uniform distribution
    # defined by the distance and epsilon, for the Euclidean distance above it
    # is the volume of a Euclidean ball with radius epsilon:
    veps = np.pi ** (D/2) / gamma(D/2 + 1) * epsilon ** D;
    
    ep_mean, ep_cov, ep_logml, nacc, ntotal, runtime = run_EPABC(data, simfun, 
        distfun, prior_mean, prior_cov, epsilon=epsilon, minacc=500, 
        samplestep=100000, samplemax=20000000, npass=2, alpha=0.9, veps=veps, 
        doQMC=True)
        
    # plot the first two dimensions of the results
    if D >= 2:
        cols = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        ax = plt.axes(aspect='equal')
        lh = []
        
        # data and likelihood
        lh.append(ax.plot(data[:, 0], data[:, 1], '.', c='0.4', label='data')[0])
        plot_cov_ellipse(lik_cov[:2, :2], lik_mean[:2], ax=ax, ec='0.4')
        lh.append(ax.plot(lik_mean[0], lik_mean[1], '+', c='0.4', label='true')[0])
        
        # analytic posterior
        plot_cov_ellipse(pos_cov[:2, :2], pos_mean[:2], ax=ax, ec=cols[0])
        lh.append(ax.plot(pos_mean[0], pos_mean[1], '+', c=cols[0], label='analytic')[0])
        
        # EP-ABC posterior
        plot_cov_ellipse(ep_cov[:2, :2], ep_mean[:2], ax=ax, ec=cols[1])
        lh.append(ax.plot(ep_mean[0], ep_mean[1], '+', c=cols[1], label='EP-ABC')[0])
        
        # legend
        ax.legend(handles=lh, loc='best', numpoints=1)
        
        plt.show()
    
    # print means and standard deviations
    print('\nana-pos: analytic posterior; EP-ABC: EP-ABC posterior')
    df = pd.DataFrame(np.c_[lik_mean, pos_mean, ep_mean], columns=['true mu', 
                      'ana-pos mu', 'EP-ABC mu'])
    df['ana-pos std'] = np.sqrt(np.diag(pos_cov))
    df['EP-ABC std'] = np.sqrt(np.diag(ep_cov))
    print(df)
    print('')
        
    # print log marginal likelihoods
    print('log marginal likelihoods:')
    print('analytic = %8.2f' % logml)
    print('EP-ABC   = %8.2f' % ep_logml)