# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:08:59 2016

@author: bitzer
"""

from __future__ import print_function, division

import math
import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from warnings import warn
from abc import ABCMeta, abstractmethod


#%% define transformations
class transform(object):
    """A parameter transformation."""
    
    __metaclass__ = ABCMeta
    
    def __init__(self, **params):
        self.transformed_range = None
    
    @abstractmethod
    def transform(self, x):
        return None
    
    @abstractmethod
    def transformed_pdf(self, y, mu, sigma2):
        return None
    
    @abstractmethod
    def transformed_mode(self, mu, sigma2):
        return None
        
    def find_transformed_mode(self, mu, sigma2, **kwargs):
        fun = lambda x: -self.transformed_pdf(x, mu, sigma2)
        
        res = scipy.optimize.minimize_scalar(fun, **kwargs)
        
        return res.x
        
    @abstractmethod
    def transformed_ppf(self, q, mu, sigma2):
        return None
        
    def approximate_transformed_ppf(self, q, mu, sigma2):
        # converts to numpy array, if necessary
        q = np.array(q)
        
        x = np.random.normal(mu, math.sqrt(sigma2), 1000)
            
        return np.percentile(self.transform(x), q * 100)
        

class identity(transform):
    def __init__(self):
        self.transformed_range = np.r_[-np.inf, np.inf]
    
    def transform(self, x):
        return x
        
    def transformed_pdf(self, y, mu, sigma2):
        return scipy.stats.norm.pdf(y, loc=mu, scale=math.sqrt(sigma2))
        
    def transformed_mode(self, mu, sigma2):
        return mu
        
    def transformed_ppf(self, q, mu, sigma2):
        return scipy.stats.norm.ppf(q, loc=mu, scale=math.sqrt(sigma2))


class absolute(transform):
    """Takes absolute value of Gaussian, creates folden normal.
    
    scipy's standardised formulation uses c = mu / sigma, scale = sigma
    """
    
    def __init__(self):
        self.transformed_range = np.r_[0, np.inf]
        
        warn('The absolute transform is not suited for inference in pyEPABC,'
             'because it most likely results in misleading posteriors!')
    
    def transform(self, x):
        return np.abs(x)
        
    def transformed_pdf(self, y, mu, sigma2):
        sigma = math.sqrt(sigma2)
        return scipy.stats.foldnorm.pdf(y, mu/sigma, scale=sigma)
        
    def transformed_mode(self, mu, sigma2):
        sigma = math.sqrt(sigma2)
        return scipy.optimize.minimize_scalar(
                lambda x: -scipy.stats.foldnorm.pdf(
                        x, mu / sigma, scale=sigma)).x
        
    def transformed_ppf(self, q, mu, sigma2):
        sigma = math.sqrt(sigma2)
        return scipy.stats.foldnorm.ppf(q, mu/sigma, scale=sigma)


class zero(transform):
    """Maps negative values to 0.
    
    This creates a complicated transformed distribution for which actual
    probability mass is collected at 0. I don't currently know whether this
    corresponds to a named distribution.
    
    The probability density function is simply that of the underlying Gaussian
    distribution with negative values set to 0, but it's unclear what a 
    pdf-value at 0 should be, if I want this to reflect the probability mass
    collected at 0, as this is not differentiable. To be at least a bit 
    informative the actual probability mass (not the density) is returned for 
    0.
    
    The transformed mode is defined to be 0 for mu <= 0 and equal to mu else.
    """
    
    def __init__(self):
        self.transformed_range = np.r_[0, np.inf]
        
    def transform(self, x):
        return np.fmax(x, 0)
    
    def transformed_pdf(self, y, mu, sigma2):
        y = np.atleast_1d(y)
        ind0 = np.flatnonzero(y == 0)
        
        pdf = scipy.stats.norm.pdf(y, loc=mu, scale=math.sqrt(sigma2))
        
        pdf[ind0] = scipy.stats.norm.cdf(0, loc=mu, scale=math.sqrt(sigma2))
            
        return pdf
    
    def transformed_mode(self, mu, sigma2):
        return max(0, mu)
        
    def transformed_ppf(self, q, mu, sigma2):
        return np.fmax(scipy.stats.norm.ppf(
                q, loc=mu, scale=math.sqrt(sigma2)), 0)
    

class exponential(transform):
    def __init__(self):
        self.transformed_range = np.r_[0, np.inf]
    
    def transform(self, x):
        return np.exp(x)
        
    def transformed_pdf(self, y, mu, sigma2):
        return scipy.stats.lognorm.pdf(y, math.sqrt(sigma2), scale=math.exp(mu))
        
    def transformed_mode(self, mu, sigma2):
        return math.exp(mu - sigma2)
        
    def transformed_ppf(self, q, mu, sigma2):
        return scipy.stats.lognorm.ppf(q, math.sqrt(sigma2), scale=math.exp(mu))
        
        
class gaussprob(transform):
    
    @property
    def width(self):
        return self._width
        
    @property
    def shift(self):
        return self._shift
    
    def __init__(self, width=1.0, shift=0.0):
        self._width = width
        self._shift = shift
        
        self.transformed_range = np.r_[self.shift, self.width + self.shift]
        
    def transform(self, x):
        return gaussprob_trans(x, self.width, self.shift)
        
    def transformed_pdf(self, y, mu, sigma2):
        if sigma2 == 1.0:
            sigma2 -= 1e-15
            warn('subtracted 1e-15 from sigma2=1.0 to avoid division by 0')
            
        return np.exp((sigma2 - 1) / 2 / sigma2 * (
                      scipy.stats.norm.ppf((y - self.shift) / self.width) - 
                      mu / (1 - sigma2)) ** 2 + 
                      mu**2 / (1 - sigma2) / 2 
                     ) / np.sqrt(sigma2) / self.width

    def transformed_mode(self, mu, sigma2):
        return self.find_transformed_mode(mu, sigma2, method='Bounded', 
                                          bounds=self.transformed_range)
    
    def transformed_ppf(self, q, mu, sigma2):
        if np.isscalar(q):
            q = np.array(q)
            
        warn('gaussprob transform: No analytic ppf implemented. '
             'Using numeric approximation.')
            
        return self.approximate_transformed_ppf(q, mu, sigma2)


try:
    from numba import vectorize, float64
    @vectorize([float64(float64, float64, float64)], nopython=True)
    def gaussprob_trans(x, width, shift):
        cdf = (1 + math.erf(x / math.sqrt(2))) / 2
    
        return cdf * width + shift
except ImportError:
    def gaussprob_trans(x, width, shift):
        return scipy.stats.norm.cdf(x) * width + shift

        
#%% define parameter container
class parameter_container:
    
    @property
    def names(self):
        return self.params.name
    
    def __init__(self):
        self.params = pd.DataFrame(columns=('name', 'transform'))
        self.P = 0
        self.mu = np.array([])
        self.cov = np.array([[]])

        
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['transformfun']
        return state
        
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.generate_transformfun()
        
        
    def add_param(self, name, mu, sigma, transform=identity(), multiples=1):
        if multiples > 1:
            name += '_{:d}'
        
        for i in range(multiples):
            self.params.loc[self.P] = [name.format(i), transform]
            self.P += 1
            self.generate_transformfun()
            
            self.mu = np.r_[self.mu, mu]
            cov = np.zeros((self.P, self.P))
            cov[:self.P-1, :self.P-1] = self.cov
            self.cov = cov
            self.cov[-1, -1] = sigma ** 2
        
    def drop_params(self, names):
        for name in names:
            if name in self.names.values:
                ind = self.names[self.names == name].index[0]
                self.params.drop(ind, inplace=True)
                self.mu = np.delete(self.mu, ind)
                self.cov = np.delete(np.delete(self.cov, 1, axis=0), 1, axis=1)
                self.P -= 1
        
                self.params.index = np.arange(self.P)
        
        self.generate_transformfun()

    def generate_transformfun(self):
        trstr = ""
        for i in range(self.P):
            trstr += ("self.params.loc[%s, 'transform']." % i +
                      "transform(values[:, %s])," % i)
        trstr = trstr[:-1]
        
        self.transformfun = eval("lambda self, values: np.c_[%s]" % trstr)
        
        
    def transform(self, values):
        return self.transformfun(self, values)
        
    
    def sample(self, S, mu=None, cov=None):
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov
            
        return np.random.multivariate_normal(mu, cov, S)
        
    
    def sample_transformed(self, S, mu=None, cov=None):
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov
            
        return self.transform(self.sample(S, mu, cov))
        
        
    def get_transformed_mode(self, mu=None, cov=None):
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov
        
        mode = np.full(self.P, np.nan)
        for par in self.params.itertuples():
            i = par.Index
            
            mode[i] = par.transform.transformed_mode(mu[i], cov[i, i])
            
        return mode
    
    
    def compare_pdfs(self, mu, cov, q_lower=0.025, q_upper=0.975, 
                     label_self='prior', label_arg='posterior', **subplots_kw):
        """Compare pdfs of the interal and an external parameter distribution.
        
        Generates a figure with one subplot per parameter showing the 
        (marginal) pdf of the parameter distribution defined by self.mu and 
        self.cov together with another parameter distribution defined by the 
        mu and cov arguments. Especially useful for comparing the change in 
        parameter distribution from prior to posterior.
        """
        P = self.params.shape[0]
        
        fig, axes = plt.subplots((P - 1) // 4 + 1, min(P, 4), squeeze=False,
                                 **subplots_kw);
        
        for par, ax in zip(self.params.itertuples(), axes.flatten()[:P]):
            ind = par.Index
            name = par.name
            par = par.transform
            
            lim1 = par.transformed_ppf(np.r_[q_lower, q_upper], self.mu[ind], 
                                       self.cov[ind, ind])
            lim2 = par.transformed_ppf(np.r_[q_lower, q_upper], mu[ind], 
                                       cov[ind, ind])
            xx = np.linspace(min(lim1[0], lim2[0]), max(lim1[1], lim2[1]), 500)
            
            
            ax.plot(xx, par.transformed_pdf(
                    xx, self.mu[ind], self.cov[ind, ind]), label=label_self); 
            ax.plot(xx, par.transformed_pdf(
                    xx, mu[ind], cov[ind, ind]), label=label_arg); 
            
            ax.set_xlabel(name)
        
        for row in axes:
            row[0].set_ylabel('density value')
            
        axes[0, 0].legend()
        
        for ax in axes.flatten()[P:]:
            ax.set_visible(False)
            
        fig.tight_layout()
        
        return fig, axes
        
    
    def plot_param_hist(self, samples, transformed=True, only_marginals=False, 
                        **distplot_kws):
        
        if transformed:
            samples = self.transform(samples)
        samples = pd.DataFrame(samples, columns=self.params.name)
        
        if only_marginals:
            fig, axes = plt.subplots(1, self.P)
            for par in self.params.itertuples():
                i = par.Index
                
                sns.distplot(samples[par.name], ax=axes[i], **distplot_kws)
                
                axes[i].set_xlabel(par.name)
                if i == 0:
                    axes[i].set_ylabel('pdf')
                    
            return fig, axes
        else:
            pg = sns.PairGrid(samples, diag_sharey=False)
            
            # scatter plot in upper diagonal
            pg = pg.map_upper(plt.scatter, alpha=0.3)
            
            # correlation of sampels in lower diagonal
            pg = pg.map_lower(plot_corrcoef)
            
            # fill diagonal with empty axes
            pg = pg.map_diag(lambda x, **kwargs: None)
            # plot marginal histograms in diagonal
            for par in self.params.itertuples():
                i = par.Index
                
                sns.distplot(samples[par.name], ax=pg.diag_axes[i],
                             **distplot_kws)
        
            return pg
    
    
    def plot_param_dist(self, mu=None, cov=None, S=500, q_lower=0.005, 
                        q_upper=0.995, only_marginals=False, dist_names=['']):
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov
        
        if only_marginals:
            fig, axes = plt.subplots(1, self.P)
            for par in self.params.itertuples():
                i = par.Index
                
                xlim = par.transform.transformed_ppf(np.r_[q_lower, q_upper], 
                                                 mu[i], cov[i, i])
                x = np.linspace(xlim[0], xlim[1], 1000)
                axes[i].plot(x, par.transform.transformed_pdf(x, mu[i], cov[i, i]))
                axes[i].set_xlim(xlim)
                axes[i].set_xlabel(par.name)
                if i == 0:
                    axes[i].set_ylabel('pdf')
                    
            return fig, axes
        else:
            samples = np.random.multivariate_normal(mu, cov, S)
            samples = self.transform(samples)
            samples = pd.DataFrame(samples, columns=self.params.name)
            samples['distribution'] = dist_names[0]
        
            pg = sns.PairGrid(samples, hue='distribution', diag_sharey=False)
            
            # scatter plot in upper diagonal
            pg = pg.map_upper(plt.scatter, alpha=0.3)
            
            # correlation of sampels in lower diagonal
            pg = pg.map_lower(plot_corrcoef)
            
            # fill diagonal with empty axes
            pg = pg.map_diag(lambda x, **kwargs: None)
            # plot analytical pdfs in diagonal
            for par in self.params.itertuples():
                i = par.Index
                
                xlim = par.transform.transformed_ppf(np.r_[q_lower, q_upper], 
                                                 mu[i], cov[i, i])
                    
                x = np.linspace(xlim[0], xlim[1], 1000)
                pg.diag_axes[i].plot(x, par.transform.transformed_pdf(x, mu[i], cov[i, i]))
                pg.diag_axes[i].set_xlim(xlim)
                
                # also set y-limits of off-diagonal
                if self.P > 1:
                    if i==0:
                        pg.axes[0, 1].set_ylim(xlim)
                    else:
                        pg.axes[i, 0].set_ylim(xlim)
            
#            pg.add_legend(frameon=True)
        
            return pg

            
def plot_corrcoef(x, y, **kwargs):
    corrcoefs = np.corrcoef(x, y)
    
    ax = plt.gca()
    ax.text(0.5, 0.5, 'R = %4.2f' % corrcoefs[0, 1], 
            horizontalalignment='center', verticalalignment='center', 
            transform=ax.transAxes, **kwargs)
    
#    ax.set_axis_off()
            
    
#%% some tests
if __name__ == "__main__":
    # test parameter container
    pars = parameter_container()
    pars.add_param('noisestd', 0, 1, transform=exponential())
    pars.add_param('prior', 0.8, 0.5, transform=gaussprob())
    pars.add_param('ndtmean', -5, 2)
    
    pg = pars.plot_param_dist()
    
    for i, par in pars.params.iterrows():
        mu = pars.mu[i]
        sigma2 = pars.cov[i, i]
        x = par.transform.transformed_mode(mu, sigma2)
        pg.diag_axes[i].plot(x, par.transform.transformed_pdf(x, mu, sigma2), '*r')
    
    # function for checking the implemented gaussprobpdf
    def check_gaussprobpdf(mu=0.0, sigma=1.0):
        g_samples = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=10000)
        p_samples = scipy.stats.norm.cdf(g_samples)
        
        gtr = gaussprob()
        
        plt.figure()
        ax = sns.distplot(p_samples)
        lower, upper = ax.get_xlim()
        yy = np.linspace(lower, upper, 1000)
        ax.plot(yy, gtr.transformed_pdf(yy, mu, sigma**2))