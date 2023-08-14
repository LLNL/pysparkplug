from typing import Dict, Optional, Tuple, Any
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import numpy as np
import math

default_params = dict()

class CategoricalDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, pMap: Optional[Dict[Any,float]] = None, default_value: float = 0.0, name: Optional[str] = None):

        self.name = name
        self.pMap = pMap

        self.no_default = default_value != 0
        self.default_value = default_value
        self.log_default_value = float(-np.inf if default_value == 0 else math.log(default_value))
        self.log1p_default_value = float(math.log1p(default_value))

    def __str__(self):
        s1 = ', '.join(['%s: %s'%(repr(k),repr(v)) for k,v in sorted(self.pMap.items(), key=lambda u: u[0])])
        s2 = repr(self.default_value)
        s3 = repr(self.name)
        return 'CategoricalDistribution({%s}, default_value=%s, name=%s)' % (s1, s2, s3)

    def density(self, x):
        return self.pMap.get(x, self.default_value) / (1.0 + self.default_value)

    def log_density(self, x):
        return np.log(self.pMap.get(x, self.default_value)) - self.log1p_default_value

    def seq_log_density(self, x):

        with np.errstate(divide='ignore'):

            xs, val_map_inv = x
            mapped_log_prob = np.asarray([self.pMap.get(u, self.default_value) for u in val_map_inv], dtype=np.float64)
            np.log(mapped_log_prob, out=mapped_log_prob)
            mapped_log_prob -= self.log1p_default_value
            rv = mapped_log_prob[xs]

        return rv

    def seq_encode(self, x):
        val_map_inv, uidx, xs = np.unique(x, return_index=True, return_inverse=True)
        val_map_inv = np.asarray([x[i] for i in uidx], dtype=object)
        return xs, val_map_inv

    def sampler(self, seed=None):
        return CategoricalSampler(self, seed)

    def estimator(self, pseudo_count=None):
        if pseudo_count is None:
            return CategoricalEstimator(name=self.name)
        else:
            return CategoricalEstimator(pseudo_count=pseudo_count, suff_stat=self.pMap, name=self.name)


class CategoricalSampler(object):

    def __init__(self, dist: CategoricalDistribution, seed=None):
        self.rng = RandomState(seed)

        temp            = list(dist.pMap.items())
        self.levels     = [u[0] for u in temp]
        self.probs      = [u[1] for u in temp]
        self.num_levels = len(self.levels)

    def sample(self, size=None):

        if size is None:
            idx = self.rng.choice(self.num_levels, p=self.probs, size=size)
            return self.levels[idx]
        else:
            levels = self.levels
            rv = self.rng.choice(self.num_levels, p=self.probs, size=size)
            return [levels[i] for i in rv]


class CategoricalAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self, keys):
        self.countMap = dict()
        self.key = keys

    def update(self, x, weight, estimate):
        self.countMap[x] = self.countMap.get(x,0) + weight

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def get_seq_lambda(self):
        return [self.seq_update]

    def seq_update(self, x, weights, estimate):
        inv_key_map = x[1]
        bcnt = np.bincount(x[0], weights=weights)
        if len(self.countMap) == 0:
            self.countMap = dict(zip(inv_key_map, bcnt))
        else:
            for i in range(0, len(bcnt)):
                self.countMap[inv_key_map[i]] += bcnt[i]

    def combine(self, suff_stat):
        for k,v in suff_stat.items():
            self.countMap[k] = self.countMap.get(k, 0) + v
        return self

    def value(self):
        return self.countMap.copy()

    def from_value(self, x):
        self.countMap = x
        return self

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


class CategoricalAccumulatorFactory(object):

    def __init__(self, keys):
        self.keys = keys

    def make(self):
        return CategoricalAccumulator(keys=self.keys)


class CategoricalEstimator(ParameterEstimator):

    def __init__(self, pseudo_count=None, suff_stat=None, default_value=False, name=None, keys=None):

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.default_value = default_value
        self.name = name
        self.keys = keys

    def accumulatorFactory(self):
        return CategoricalAccumulatorFactory(self.keys)

    def estimate(self, nobs, suff_stat):

        stats_sum = sum(suff_stat.values())

        if self.default_value:
            if stats_sum > 0:
                default_value = 1.0/stats_sum
                default_value *= default_value
            else:
                default_value = 0.5
        else:
            default_value = 0.0

        if self.pseudo_count is None and self.suff_stat is None:

            nobs_loc = stats_sum

            if nobs_loc == 0:
                pMap = {k : 1.0/float(len(suff_stat)) for k in suff_stat.keys()}
            else:
                pMap = {k: v / nobs_loc for k, v in suff_stat.items()}

        elif self.pseudo_count is not None and self.suff_stat is None:

            nobs_loc = stats_sum
            pseudo_countPerLevel = self.pseudo_count/len(suff_stat)
            adjustedNobs = nobs_loc + self.pseudo_count

            for k,v in suff_stat.items():
                suff_stat[k] = (v + pseudo_countPerLevel) / adjustedNobs
            #pMap = {k: (v + pseudo_countPerLevel) / adjustedNobs for k, v in suff_stat.items()}
            pMap = suff_stat

        else:  # self.pseudo_count is not None and self.suff_stat is not None:

            suff_stat_sum = sum(self.suff_stat.values())

            levels = set(suff_stat.keys()).union(self.suff_stat.keys())
            adjustedNobs = suff_stat_sum * self.pseudo_count + stats_sum

            pMap = {k: (suff_stat.get(k, 0) + self.suff_stat.get(k, 0) * self.pseudo_count) / adjustedNobs for k in levels}


        return CategoricalDistribution(pMap=pMap, default_value=default_value, name=self.name)
