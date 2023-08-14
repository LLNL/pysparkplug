from typing import Optional, Union, Sequence, Tuple
from pysp.utils.special import gammaln
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import numpy as np

DatumType = int
EncodedDataType = Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]

class BinomialDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, p, n, min_val=None, name=None, keys=None):
        self.p      = p
        self.n      = n
        self.log_p  = np.log(p)
        self.log_1p = np.log1p(-p)
        self.name   = name
        self.keys   = keys
        self.min_val = min_val
        self.has_min = min_val is not None

    def __str__(self):
        return 'BinomialDistribution(p=%s, n=%s, min_val=%s, name=%s, keys=%s)'%(repr(self.p), repr(self.n), repr(self.min_val), repr(self.name), repr(self.keys))

    def density(self, x: DatumType) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: DatumType) -> float:
        n = self.n
        if self.has_min:
            xx = x - self.min_val
        else:
            xx = x

        return (gammaln(xx+1)-gammaln(n)-gammaln(n-xx+1)) + self.log_1p*(n-xx) + self.log_p*xx

    def seq_log_density(self, x: EncodedDataType) -> np.ndarray:
        ux, ix, _, _, _ = x
        n = self.n
        gn = gammaln(n)

        if self.has_min:
            xx = ux - self.min_val
        else:
            xx = ux

        cc = (gammaln(xx+1) - gn - gammaln((n+1)-xx)) + self.log_1p*(n-xx) + self.log_p*xx
        return cc[ix]

    def seq_encode(self, x: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        xx = np.asarray(x, dtype=np.int32)
        ux, ix = np.unique(xx, return_inverse=True)
        min_val = np.min(ux)
        max_val = np.max(ux)
        return ux, ix, xx, min_val, max_val

    def sampler(self, seed: Optional[int] = None):
        return BinomialSampler(self, seed)

    def estimator(self, pseudo_count=None):
        if pseudo_count is None:
            return BinomialEstimator(name=self.name, keys=self.keys)
        else:
            return BinomialEstimator(max_val=self.n, min_val=self.min_val, pseudo_count=pseudo_count, suff_stat=self.p*self.n*pseudo_count, name=self.name)


class BinomialSampler(object):

    def __init__(self, dist: BinomialDistribution, seed=None):
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size=None) -> Union[int, Sequence[int]]:
        rv = self.rng.binomial(n=self.dist.n, p=self.dist.p, size=size)

        if size is None:
            if self.dist.has_min:
                return rv - self.dist.min_val
            else:
                return rv
        else:
            if self.dist.has_min:
                return list(rv + self.dist.min_val)
            else:
                return list(rv)


class BinomialAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, max_val=None, min_val=None, name=None, keys=None):
        self.sum     = 0.0
        self.count   = 0.0
        self.key     = keys
        self.name    = name
        self.max_val = max_val
        self.min_val = min_val

    def update(self, x, weight, estimate):
        self.sum += x*weight
        self.count += weight

        if self.min_val is None:
            self.min_val = x
        else:
            self.min_val = min(self.min_val, x)

        if self.max_val is None:
            self.max_val = x
        else:
            self.max_val = max(self.max_val, x)

    def seq_update(self, x, weights, estimate):
        _, _, xx, min_val, max_val = x
        self.sum += np.dot(xx, weights)
        self.count += weights.sum()

        if self.min_val is None:
            self.min_val = min_val
        else:
            self.min_val = min(self.min_val, min_val)

        if self.max_val is None:
            self.max_val = max_val
        else:
            self.max_val = max(self.max_val, max_val)

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def combine(self, suff_stat):
        self.sum   += suff_stat[1]
        self.count += suff_stat[0]

        if self.min_val is None:
            self.min_val = suff_stat[2]
        else:
            self.min_val = min(self.min_val, suff_stat[2])

        if self.max_val is None:
            self.max_val = suff_stat[3]
        else:
            self.max_val = max(self.max_val, suff_stat[3])

        return self

    def value(self):
        return self.count, self.sum, self.min_val, self.max_val

    def from_value(self, x):
        self.count = x[0]
        self.sum = x[1]
        self.min_val = x[2]
        self.max_val = x[3]
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


class BinomialAccumulatorFactory(object):

    def __init__(self, max_val, min_val, name, keys):
        self.max_val = max_val
        self.min_val = min_val
        self.name = name
        self.keys = keys

    def make(self):
        return BinomialAccumulator(self.max_val, self.min_val, self.name, self.keys)

class BinomialEstimator(ParameterEstimator):

    def __init__(self, max_val: Optional[int] = None, min_val: Optional[int] = 0, pseudo_count: Optional[float] = None, suff_stat: Optional[float] = None, name: Optional[str] = None, keys: Optional[str] = None):

        self.pseudo_count  = pseudo_count
        self.suff_stat     = suff_stat
        self.keys          = keys
        self.name          = name
        self.min_val       = min_val
        self.max_val       = max_val

    def accumulatorFactory(self):
        return BinomialAccumulatorFactory(self.max_val, self.min_val, self.name, self.keys)

    def estimate(self, nobs, suff_stat):

        count, sum, min_val, max_val = suff_stat

        if min_val is not None:
            if self.min_val is not None:
                min_val = min(min_val, self.min_val)
        else:
            if self.min_val is not None:
                min_val = self.min_val
            else:
                min_val = 0

        if max_val is not None:
            if self.max_val is not None:
                max_val = max(max_val, self.max_val)
        else:
            if self.max_val is not None:
                max_val = self.max_val
            else:
                max_val = 0

        n = max_val - min_val

        if self.pseudo_count is not None and self.suff_stat is not None:
            pn = self.pseudo_count
            pp = self.suff_stat
            p = (sum - min_val * count + pp)/((count + pn) * n)

        elif self.pseudo_count is not None and self.suff_stat is None:
            pn = self.pseudo_count
            pp = self.pseudo_count * 0.5 * n
            p = (sum - min_val * count + pp) / ((count + pn) * n)

        else:
            if count > 0 and n > 0:
                p = (sum - min_val * count)/(count * n)
            else:
                p = 0.5

        return BinomialDistribution(p, max_val - min_val, min_val=min_val, name=self.name, keys=self.keys)
