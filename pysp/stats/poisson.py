from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
from scipy.special import gammaln


class PoissonDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, lam, name=None):
        self.lam       = lam
        self.logLambda = log(lam)
        self.name      = name

    def __str__(self):
        return 'PoissonDistribution(%s, name=%s)'%(repr(self.lam), repr(self.name))

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):
        return x*self.logLambda - gammaln(x + 1.0) - self.lam

    def seq_log_density(self, x):
        rv = x[0]*self.logLambda
        rv -= x[1]
        rv -= self.lam
        return rv

    def seq_encode(self, x):
        rv1 = np.asarray(x, dtype=float)
        rv2 = gammaln(rv1 + 1.0)
        return rv1, rv2

    def sampler(self, seed=None):
        return PoissonSampler(self, seed)

    def estimator(self, pseudo_count=None):

        if pseudo_count is None:
            return PoissonEstimator(name=self.name)
        else:
            return PoissonEstimator(pseudo_count=pseudo_count, suff_stat=self.lam, name=self.name)

class PoissonSampler(object):

    def __init__(self, dist, seed=None):
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        return self.rng.poisson(lam=self.dist.lam, size=size)


class PoissonAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, keys):
        self.sum   = 0.0
        self.count = 0.0
        self.key   = keys

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def update(self, x, weight, estimate):
        self.sum  += x*weight
        self.count += weight

    def seq_update(self, x, weights, estimate):
        self.sum   += np.dot(x[0], weights)
        self.count += weights.sum()

    def combine(self, suff_stat):
        self.sum  += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self):
        return self.count, self.sum

    def from_value(self, x):
        self.count = x[0]
        self.sum = x[1]

    def key_merge(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class PoissonAccumulatorFactory(object):

    def __init__(self, keys):
        self.keys = keys

    def make(self):
        return PoissonAccumulator(keys=self.keys)


class PoissonEstimator(ParameterEstimator):

    def __init__(self, pseudo_count=None, suff_stat=None, name=None, keys=None):

        self.pseudo_count  = pseudo_count
        self.suff_stat     = suff_stat
        self.name = name
        self.keys = keys

    def accumulatorFactory(self):
        return PoissonAccumulatorFactory(self.keys)

    def estimate(self, nobs, suff_stat):

        nobs, psum = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:
            return PoissonDistribution((psum + self.suff_stat*self.pseudo_count)/(nobs + self.pseudo_count), name=self.name)
        else:
            return PoissonDistribution(psum/nobs, name=self.name)
