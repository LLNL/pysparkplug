from typing import Sequence, Optional, Tuple, Union
from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np


class IntegerBernoulliSetDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, log_pvec: Union[Sequence[float], np.ndarray], log_nvec: Optional[Union[Sequence[float], np.ndarray]] = None, name: Optional[str] = None):

        num_vals = len(log_pvec)
        self.name     = name
        self.num_vals = num_vals
        self.log_pvec = np.asarray(log_pvec, dtype=np.float64).copy()

        if log_nvec is None:
            '''
            is_one   = log_pvec == 0
            is_zero  = log_pvec == -np.inf
            is_good  = np.bitwise_and(~is_one, ~is_zero)

            log_nvec = np.zeros(len(log_pvec), dtype=np.float64)
            log_dvec = np.zeros(len(log_pvec), dtype=np.float64)
            log_nvec[is_good] = np.log1p(-np.exp(self.log_pvec[is_good]))
            log_dvec[is_good] = self.log_pvec[is_good] - log_nvec[is_good]
            log_dvec[is_zero] = -np.inf

            self.log_nvec = None
            self.log_dvec = log_dvec
            self.log_nsum = np.sum(log_nvec)
            '''
            log_nvec = np.log1p(-np.exp(self.log_pvec))
            self.log_nvec = None
            self.log_dvec = self.log_pvec - log_nvec
            self.log_nsum = np.sum(log_nvec[np.isfinite(log_nvec)])
        else:
            self.log_nvec = np.asarray(log_nvec, dtype=np.float64)
            self.log_dvec = self.log_pvec - self.log_nvec
            self.log_nsum = np.sum(self.log_nvec[np.isfinite(self.log_nvec)])

    def __str__(self):
        s1 = repr(list(self.log_pvec))
        s2 = repr(None if self.log_nvec is None else list(self.log_nvec))
        s3 = repr(self.name)
        return 'IntegerBernoulliSetDistribution(%s, log_nvec=%s, name=%s)'%(s1, s2, s3)

    def density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        xx = np.asarray(x, dtype=int)
        return np.sum(self.log_dvec[xx]) + self.log_nsum

    def seq_log_density(self, x: Tuple[int, np.ndarray, np.ndarray]) -> np.ndarray:
        sz, idx, xs = x
        rv = np.zeros(sz, dtype=np.float64)
        rv += np.bincount(idx, weights=self.log_dvec[xs], minlength=sz)
        rv += self.log_nsum
        return rv

    def seq_encode(self, x: Sequence[Sequence[int]]):
        idx = []
        xs  = []
        for i,xx in enumerate(x):
            idx.extend([i] * len(xx))
            xs.extend(xx)

        idx = np.asarray(idx, dtype=np.int32)
        xs  = np.asarray(xs, dtype=np.int32)

        return len(x), idx, xs

    def sampler(self, seed: Optional[int] = None):
        return IntegerBernoulliSetSampler(self, seed)

    def estimator(self, pseudo_count=None):
        return IntegerBernoulliSetEstimator(self.num_vals, pseudo_count=pseudo_count, name=self.name)


class IntegerBernoulliSetSampler(object):

    def __init__(self, dist: IntegerBernoulliSetDistribution, seed: Optional[int] = None):
        self.rng  = np.random.RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None):
        if size is None:
            return list(np.flatnonzero(np.log(self.rng.rand(self.dist.num_vals)) <= self.dist.log_pvec))
        else:
            rv = []
            for i in range(size):
                rv.append(list(np.flatnonzero(np.log(self.rng.rand(self.dist.num_vals)) <= self.dist.log_pvec)))
            return rv

class IntegerBernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals, keys):
        self.pcnt    = np.zeros(num_vals, dtype=np.float64)
        self.key     = keys
        self.num_vals = num_vals
        self.tot_sum = 0.0

    def update(self, x, weight, estimate):
        xx = np.asarray(x, dtype=int)
        self.pcnt[xx] += weight
        self.tot_sum += weight

    def initialize(self, x, weight, rng):
        xx = np.asarray(x, dtype=int)
        #rv = rng.rand(self.num_vals)
        #self.pcnt += (1-rv)*weight
        self.pcnt[xx] += weight
        self.tot_sum += weight

    def seq_update(self, x, weights, estimate):
        sz, idx, xs = x
        agg_cnt = np.bincount(xs, weights=weights[idx])
        n = len(agg_cnt)
        self.pcnt[:n] += agg_cnt
        self.tot_sum += weights.sum()

    def combine(self, suff_stat):
        self.pcnt    += suff_stat[0]
        self.tot_sum += suff_stat[1]
        return self

    def value(self):
        return self.pcnt, self.tot_sum

    def from_value(self, x):
        self.pcnt = x[0]
        self.tot_sum = x[1]
        return self

    def key_merge(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                temp = stats_dict[self.key]
                stats_dict[self.key] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.key] = (self.pcnt, self.tot_sum)

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.key]



class IntegerBernoulliSetAccumulatorFactory(object):
    def __init__(self, num_vals, keys):
        self.keys = keys
        self.num_vals = num_vals

    def make(self):
        return IntegerBernoulliSetAccumulator(self.num_vals, keys=self.keys)


class IntegerBernoulliSetEstimator(ParameterEstimator):

    def __init__(self, num_vals: int, min_prob: float = 1.0e-128, pseudo_count: Optional[float] = None, suff_stat: Optional[np.ndarray] = None, name=None, keys=None):
        self.num_vals      = num_vals
        self.keys          = keys
        self.pseudo_count  = pseudo_count
        self.suff_stat     = suff_stat
        self.name          = name
        self.min_prob      = min_prob

    def accumulatorFactory(self):
        return IntegerBernoulliSetAccumulatorFactory(self.num_vals, self.keys)

    def estimate(self, nobs, suff_stat):

        if self.pseudo_count is not None and self.suff_stat is not None:
            p0 = np.product(self.suff_stat, self.pseudo_count)
            p1 = np.product(np.subtract(1.0, self.suff_stat), self.pseudo_count)
            pvec = np.log(suff_stat[0] + p0)
            nvec = np.log((suff_stat[1] - suff_stat[0]) + p1)
            tsum = np.log(suff_stat[1] + self.pseudo_count)
            log_pvec = np.log(pvec) - tsum
            log_nvec = np.log(nvec) - tsum

        elif self.pseudo_count is not None and self.suff_stat is None:
            p   = self.pseudo_count
            log_c    = np.log(suff_stat[1] + p)
            log_pvec = np.log(suff_stat[0] + (p/2.0)) - log_c
            log_nvec = np.log((suff_stat[1] - suff_stat[0]) + (p/2.0)) - log_c

        else:

            if suff_stat[1] == 0:
                log_pvec = np.zeros(self.num_vals, dtype=np.float64) + 0.5
                log_nvec = np.zeros(self.num_vals, dtype=np.float64) + 0.5

            elif self.min_prob > 0:
                log_pvec = np.log(np.maximum(suff_stat[0]/suff_stat[1], self.min_prob))
                log_nvec = np.log(np.maximum((suff_stat[1] - suff_stat[0])/suff_stat[1], self.min_prob))

            else:
                pvec = suff_stat[0]/suff_stat[1]
                nvec = (suff_stat[1] - suff_stat[0])/suff_stat[1]

                is_zero = (pvec == 0)
                is_one  = (nvec == 0)

                log_pvec = np.zeros(self.num_vals, dtype=np.float64)
                log_nvec = np.zeros(self.num_vals, dtype=np.float64)

                log_pvec[~is_zero] = np.log(pvec[~is_zero])
                log_pvec[is_zero]  = -np.inf
                log_nvec[~is_one]  = np.log(nvec[~is_one])
                log_nvec[is_one]   = -np.inf

        return IntegerBernoulliSetDistribution(log_pvec, log_nvec, name=self.name)
