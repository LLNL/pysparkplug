from typing import Union, Sequence, Optional
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import pysp.utils.vector as vec
import numpy as np


class DiagonalGaussianDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, mu: Union[Sequence[float], np.ndarray], covar: Union[Sequence[float], np.ndarray], name: Optional[str] = None):

        self.dim      = len(mu)
        self.mu       = np.asarray(mu, dtype=float)
        self.covar    = np.asarray(covar, dtype=float)
        self.name     = name
        self.log_c    = -0.5*(np.log(2.0*np.pi)*self.dim + np.log(self.covar).sum())

        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c

    def __str__(self):
        s1 = repr(list(self.mu.flatten()))
        s2 = repr(list(self.covar.flatten()))
        s3 = repr(self.name)
        return 'DiagonalGaussianDistribution(%s, %s, name=%s)'%(s1, s2, s3)

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
        return rv

    def seq_log_density(self, x):
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
        return rv

    def seq_encode(self, x):
        xv = np.reshape(x, (-1, self.dim))
        return xv

    def sampler(self, seed=None):
        return DiagonalGaussianSampler(self, seed)

    def estimator(self, pseudo_count=None):
        if pseudo_count is None:
            return DiagonalGaussianEstimator(name=self.name)
        else:
            return DiagonalGaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), name=self.name)


class DiagonalGaussianSampler(object):

    def __init__(self, dist: DiagonalGaussianDistribution, seed: Optional[int] = None):
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None):
        if size is None:
            rv  = self.rng.randn(self.dist.dim)
            rv *= np.sqrt(self.dist.covar)
            rv += self.dist.mu
            return rv
        else:
            return [self.sample() for i in range(size)]



class DiagonalGaussianAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, dim=None):

        self.dim     = dim
        self.count   = 0.0

        if dim is not None:
            self.sum  = vec.zeros(dim)
            self.sum2 = vec.zeros(dim)
        else:
            self.sum = None
            self.sum2 = None

    def update(self, x, weight, estimate):

        if self.dim is None:
            self.dim  = len(x)
            self.sum  = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        xWeight    = x*weight
        self.count += weight
        self.sum  += xWeight
        xWeight *= x
        self.sum2 += xWeight

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_update(self, x, weights, estimate):

        if self.dim is None:
            self.dim  = x.shape[1]
            self.sum  = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        xWeight    = np.multiply(x.T, weights)
        self.count += weights.sum()
        self.sum   += xWeight.sum(axis=1)
        xWeight *= x.T
        self.sum2  += xWeight.sum(axis=1)

    def combine(self, suff_stat):

        if suff_stat[0] is not None and self.sum is not None:
            self.sum  += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]

        elif suff_stat[0] is not None and self.sum is None:
            self.sum  = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]

        return self

    def value(self):
        return self.sum, self.sum2, self.count

    def from_value(self, x):
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]


class DiagonalGaussianEstimator(object):

    def __init__(self, dim=None, pseudo_count=(None, None), suff_stat = (None, None), name=None):

        dim_loc = dim if dim is not None else ((None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1])))) if suff_stat[0] is None else len(suff_stat[0]))

        self.name           = name
        self.dim            = dim_loc
        self.is_diag        = False
        self.pseudo_count   = pseudo_count
        self.priorMu        = None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
        self.priorCovar     = None if suff_stat[1] is None else np.reshape(suff_stat[1], dim_loc)

    def accumulatorFactory(self):
        dim = self.dim
        obj = type('', (object,), {'make': lambda o: DiagonalGaussianAccumulator(dim=dim)})()
        return(obj)

    def estimate(self, nobs, suff_stat):

        nobs = suff_stat[2]
        pc1, pc2 = self.pseudo_count

        if pc1 is not None and self.priorMu is not None:
            mu = (suff_stat[0] + pc1*self.priorMu)/(nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.priorCovar is not None:
            covar = (suff_stat[1] + (pc2 * self.priorCovar) - (mu*mu*nobs))/(nobs + pc2)
        else:
            covar = (suff_stat[1]/nobs) - (mu*mu)

        return DiagonalGaussianDistribution(mu, covar, name=self.name)
