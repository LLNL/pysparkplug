from typing import Optional, Tuple
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator
from numpy.random import RandomState
import numpy as np


class ExponentialDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, beta: float, name: Optional[str] = None):
        self.beta = beta
        self.logBeta = np.log(beta)
        self.name = name

    def __str__(self):
        return 'ExponentialDistribution(%s, name=%s)' % (repr(self.beta), repr(self.name))

    def density(self, x):
        return np.exp(self.log_density(x))

    def log_density(self, x):
        if x < 0:
            return -inf
        else:
            return -x / self.beta - self.logBeta

    def seq_log_density(self, x):
        rv = x * (-1.0 / self.beta)
        rv -= self.logBeta
        return rv

    def seq_encode(self, x):
        rv = np.asarray(x)
        return rv

    def sampler(self, seed=None):
        return ExponentialSampler(self, seed)

    def estimator(self, pseudo_count=None):

        if pseudo_count is None:
            return ExponentialEstimator(name=self.name)
        else:
            return ExponentialEstimator(pseudo_count=pseudo_count, suff_stat=self.beta, name=self.name)


class ExponentialSampler(object):

    def __init__(self, dist: ExponentialDistribution, seed: Optional[int] = None):
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        return self.rng.exponential(scale=self.dist.beta, size=size)


class ExponentialAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, keys):
        self.sum = 0.0
        self.count = 0.0
        self.key = keys

    def update(self, x, weight, estimate):
        if x >= 0:
            self.sum += x * weight
            self.count += weight

    def seq_update(self, x, weights, estimate):
        self.sum += np.dot(x, weights)
        self.count += np.sum(weights)

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def combine(self, suff_stat):
        self.sum += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self):
        return self.count, self.sum

    def from_value(self, x):
        self.count = x[0]
        self.sum = x[1]
        return self

    def key_merge(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                vals = stats_dict[self.key]
                stats_dict[self.key] = (vals[0] + self.count, vals[1] + self.sum)
            else:
                stats_dict[self.key] = (self.count, self.sum)

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                vals = stats_dict[self.key]
                self.count = vals[0]
                self.sum = vals[1]


class ExponentialAccumulatorFactory(object):

    def __init__(self, keys):
        self.keys = keys

    def make(self):
        return ExponentialAccumulator(keys=self.keys)


class ExponentialEstimator(ParameterEstimator):

    def __init__(self, pseudo_count: Optional[float] = None, suff_stat: Optional[float] = None, name: Optional[str] = None, keys: Optional[str] = None):

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulatorFactory(self):
        return ExponentialAccumulatorFactory(self.keys)

    def estimate(self, nobs, suff_stat):

        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[1] + self.suff_stat * self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[1] + self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        else:
            if suff_stat[0] > 0:
                p = suff_stat[1] / suff_stat[0]
            else:
                p = 1.0

        return ExponentialDistribution(p, name=self.name)
