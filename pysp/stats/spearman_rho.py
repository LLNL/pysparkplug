from typing import Optional, Sequence, Union
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
import itertools


class SpearmanRankingDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, sigma: Union[Sequence[float], np.ndarray], rho: float = 1.0, name: Optional[str] = None, keys: Optional[str] = None):

        self.sigma = np.asarray(sigma)
        self.rho   = rho
        self.name  = name
        self.dim   = len(sigma)
        self.keys  = keys

        perms = map(np.asarray,map(list, itertools.permutations(range(self.dim))))
        self.log_const = np.log(sum(map(lambda u: np.exp(-rho * np.dot(self.sigma-u, self.sigma-u)), perms)))

    def __str__(self):
        return 'SpearmanRankingDistribution(sigma=%s, rho=%s, name=%s, keys=%s)'%(repr(self.sigma), repr(self.rho), repr(self.name), repr(self.keys))

    def density(self, x):
        return np.exp(self.log_density(x))

    def log_density(self, x):
        temp = np.subtract(x, self.sigma)
        return -self.rho * np.dot(temp,temp) - self.log_const

    def seq_log_density(self, x):
        temp = x - self.sigma
        temp *= temp
        rv = np.sum(temp, axis=1) * -self.rho
        rv -= self.log_const
        return rv

    def seq_encode(self, x):
        rv = np.asarray(x)
        return rv

    def sampler(self, seed=None):
        return SpearmanRankingSampler(self, seed)

    def estimator(self, pseudo_count=None):
        return SpearmanRankingEstimator(self.dim, pseudo_count=pseudo_count, name=self.name, keys=self.keys)


class SpearmanRankingSampler(object):

    def __init__(self, dist, seed=None):
        self.rng  = np.random.RandomState(seed)
        self.dist = dist

        self.perms = list(map(list, itertools.permutations(range(dist.dim))))
        self.probs = np.exp(dist.seq_log_density(dist.seq_encode(self.perms)))

    def sample(self, size=None):
        idx = self.rng.choice(len(self.perms), p=self.probs, replace=True, size=size)

        if size is None:
            return self.perms[idx]
        else:
            return [self.perms[u] for u in idx]


class SpearmanRankingAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, dim, name=None, keys=None):
        self.sum  = np.zeros(dim, dtype=np.float64)
        self.count = 0.0
        self.key = keys
        self.name = name

    def update(self, x, weight, estimate):
        self.sum += np.multiply(x, weight)
        self.count += weight

    def initialize(self, x, weight, rng):
        if weight != 0:
            self.sum += np.multiply(x, weight)
            self.count += 0

    def seq_update(self, x, weights, estimate):
        self.sum += np.dot(x.T, weights)
        self.count += weights.sum()

    def combine(self, suff_stat):
        self.sum  += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self):
        return self.count, self.sum

    def from_value(self, x):
        self.sum = x[1]
        self.count = x[0]
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


class SpearmanRankingAccumulatorFactory():

    def __init__(self, dim, name, keys):
        self.keys = keys
        self.name = name
        self.dim  = dim

    def make(self):
        return SpearmanRankingAccumulator(dim=self.dim, name=self.name, keys=self.keys)


class SpearmanRankingEstimator(ParameterEstimator):

    def __init__(self, dim, pseudo_count=None, suff_stat=None, name=None, keys=None):

        self.pseudo_count  = pseudo_count
        self.suff_stat     = suff_stat
        self.keys          = keys
        self.name          = name
        self.dim           = dim

    def accumulatorFactory(self):
        return SpearmanRankingAccumulatorFactory(self.dim, self.name, self.keys)

    def estimate(self, nobs, suff_stat):

        count, vsum = suff_stat

        if count > 0:
            sigma = np.argsort(vsum)
            rho   = 1.0
        else:
            sigma = vsum
            rho   = 0.0

        return SpearmanRankingDistribution(sigma, rho, name=self.name, keys=self.keys)



