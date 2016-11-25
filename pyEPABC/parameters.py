# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:08:59 2016

@author: bitzer
"""

import math
import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from warnings import warn
from abc import ABCMeta, abstractmethod


#%% define transformations
class transform(metaclass=ABCMeta):
    """A parameter transformation."""
    
    def __init__(self, **params):
        self.transformed_range = None
    
    @abstractmethod
    def transform(self, x):
        return None
    
    @abstractmethod
    def transformed_pdf(self, y, mu, sigma2):
        return None
    
    @abstractmethod
    def transformed_ppf(self, q, mu, sigma2):
        return None

class identity(transform):
    def __init__(self):
        self.transformed_range = np.r_[-np.inf, np.inf]
    
    def transform(self, x):
        return x
        
    def transformed_pdf(self, y, mu, sigma2):
        return scipy.stats.norm.pdf(y, loc=mu, scale=math.sqrt(sigma2))
        
    def transformed_ppf(self, q, mu, sigma2):
        return scipy.stats.norm.ppf(q, loc=mu, scale=math.sqrt(sigma2))


class exponential(transform):
    def __init__(self):
        self.transformed_range = np.r_[0, np.inf]
    
    def transform(self, x):
        return np.exp(x)
        
    def transformed_pdf(self, y, mu, sigma2):
        return scipy.stats.lognorm.pdf(y, math.sqrt(sigma2), scale=math.exp(mu))
        
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
    
    def transformed_ppf(self, q, mu, sigma2):
        if np.isscalar(q):
            q = np.array(q)
            
        return np.full(q.shape, np.nan)


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
    
    def __init__(self):
        self.params = pd.DataFrame(columns=('name', 'transform'))
        self.P = 0
        self.mu = np.array([])
        self.cov = np.array([[]])

        
    def add_param(self, name, mu, sigma2, transform=identity()):
        self.params.loc[self.P] = [name, transform]
        self.P += 1
        self.generate_transformfun()
        
        self.mu = np.r_[self.mu, mu]
        cov = np.zeros((self.P, self.P))
        cov[:self.P-1, :self.P-1] = self.cov
        self.cov = cov
        self.cov[-1, -1] = sigma2


    def generate_transformfun(self):
        trstr = ""
        for i in range(self.P):
            trstr += ("self.params.loc[%s, 'transform']." % i +
                      "transform(values[:, %s])," % i)
        trstr = trstr[:-1]
        
        self.transformfun = eval("lambda self, values: np.c_[%s]" % trstr)
        
        
    def transform(self, values):
        return self.transformfun(self, values)
        
        
    def plot_param_dist(self, mu=None, cov=None, S=500, q_lower=0.005, 
                        q_upper=0.995, only_marginals=False, dist_names=['']):
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov
        
        if only_marginals:
            fig, axes = plt.subplots(1, self.P)
            for i, par in self.params.iterrows():
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
            for i, par in self.params.iterrows():
                xlim = par.transform.transformed_ppf(np.r_[q_lower, q_upper], 
                                                 mu[i], cov[i, i])
                if np.any(np.isnan(xlim)):
                    xlim = par.transform.transformed_range
                    
                x = np.linspace(xlim[0], xlim[1], 1000)
                pg.diag_axes[i].plot(x, par.transform.transformed_pdf(x, mu[i], cov[i, i]))
                pg.diag_axes[i].set_xlim(xlim)
            
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
    pars.add_param('prior', 0, 1, transform=gaussprob())
    pars.add_param('ndtmean', -5, 2)
    
    pg = pars.plot_param_dist()
    
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