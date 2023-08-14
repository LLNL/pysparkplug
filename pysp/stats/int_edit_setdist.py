from typing import Sequence, Optional, Tuple, Union, Any
from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
from pysp.arithmetic import maxrandint


DatumType = Tuple[Union[Sequence[int], np.ndarray], Union[Sequence[int], np.ndarray]]
EncodedDatumType = Tuple[int, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray,np.ndarray,np.ndarray], Any]

class IntegerBernoulliEditDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, log_edit_pmat: Union[Sequence[Tuple[float,float]], np.ndarray],  init_dist: SequenceEncodableProbabilityDistribution = None, name: Optional[str] = None):

        num_vals = len(log_edit_pmat)
        self.name = name
        self.init_dist = init_dist
        self.num_vals = num_vals

        pmat = np.asarray(log_edit_pmat, dtype=np.float64).copy()
        if pmat.shape[1] == 2:
            log_pmat = np.zeros((num_vals, 4), dtype=np.float64)
            log_pmat[:, 0] = np.log1p(-np.exp(pmat[:, 0])) # P(missing | missing) = 1 - P(present | missing)
            log_pmat[:, 1] = np.log1p(-np.exp(pmat[:, 1])) # P(missing | present) = 1 - P(present | present)
            log_pmat[:, 2] = pmat[:, 0] # P(present | missing)
            log_pmat[:, 3] = pmat[:, 1] # P(present | present)
        else:
            log_pmat = pmat

        self.orig_log_edit_pmat = pmat
        self.log_edit_pmat = log_pmat
        self.log_nsum = self.log_edit_pmat[np.isfinite(self.log_edit_pmat[:,0]),0].sum() # sum [ln P(missing | missing)]
        self.log_dvec = self.log_edit_pmat[:,1:] - self.log_edit_pmat[:,0,None] #ln P (?? | ??) - ln P(missing | missing)

    def __str__(self):
        s1 = repr(list(map(list,self.orig_log_edit_pmat)))
        s2 = repr(self.init_dist) if self.init_dist is None else str(self.init_dist)
        s3 = repr(self.name)
        return 'IntegerBernoulliEditDistribution(%s, init_dist=%s, name=%s)'%(s1, s2, s3)

    def density(self, x: DatumType) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: DatumType) -> float:

        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        in10 = np.isin(xx1, xx0, invert=False) # xx0 \cap xx1
        in01 = np.isin(xx0, xx1, invert=True) # xx0 \cap xx1

        yy = np.ones(len(xx1), dtype=int)
        yy[in10] = 2
        rv = self.log_nsum # ln P(missing | missing) for the empty set
        rv += np.sum(self.log_dvec[xx1[in10],  2]) # ln P(present | present) same stuff that was there
        rv += np.sum(self.log_dvec[xx1[~in10], 1]) # ln P(present | missing) new additions
        rv += np.sum(self.log_dvec[xx0[in01],  0]) # ln P(missing | present) stuff to remove
        # rv = ln P(x[1] | x[0])

        # rv = ln P(x[1] | x[0]) + ln(P(x[0]) = ln P(x[0], x[1])
        if self.init_dist is not None:
            rv += self.init_dist.log_density(x[0])

        return rv

    def seq_log_density(self, x: EncodedDatumType) -> np.ndarray:

        sz, idx, xs, ys, ym, init_enc = x
        rv = np.bincount(idx, weights=self.log_dvec[xs,ys], minlength=sz)
        rv += self.log_nsum

        if self.init_dist is not None:
            rv += self.init_dist.seq_log_density(init_enc)

        return rv

    def seq_encode(self, x: Sequence[DatumType]) -> EncodedDatumType:

        idx = []
        xs  = []
        ys  = []
        pre = []

        for i,xx in enumerate(x):
            pre.append(xx[0])

            xx0 = np.asarray(xx[0], dtype=int)
            xx1 = np.asarray(xx[1], dtype=int)

            to_add = np.isin(xx1, xx0, invert=False)
            to_rem = np.isin(xx0, xx1, invert=True)

            new_x = np.concatenate([xx0[to_rem], xx1[~to_add], xx1[to_add]])
            new_i = np.concatenate([[0]*np.sum(to_rem), [1]*np.sum(~to_add), [2]*np.sum(to_add)])

            idx.extend([i] * len(new_x))
            xs.extend(list(new_x))
            ys.extend(list(new_i))

        idx = np.asarray(idx, dtype=np.int32)
        xs  = np.asarray(xs,  dtype=np.int32)
        ys  = np.asarray(ys,  dtype=np.int32)
        ym  = (np.flatnonzero(ys==0), np.flatnonzero(ys==1), np.flatnonzero(ys==2))

        if self.init_dist is not None:
            init_enc = self.init_dist.seq_encode(pre)
        else:
            init_enc = None

        return len(x), idx, xs, ys, ym, init_enc

    def sampler(self, seed: Optional[int] = None):
        return IntegerBernoulliEditSampler(self, seed)

    def estimator(self, pseudo_count=None):
        return IntegerBernoulliEditEstimator(self.num_vals, pseudo_count=pseudo_count, name=self.name)


class IntegerBernoulliEditSampler(object):

    def __init__(self, dist: IntegerBernoulliEditDistribution, seed: Optional[int] = None):
        self.rng  = np.random.RandomState(seed)
        self.dist = dist
        self.init_rng = dist.init_dist.sampler(self.rng.randint(0, maxrandint))
        self.next_rng = np.random.RandomState(self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None):
        if size is None:

            temp = np.log(self.rng.rand(self.dist.num_vals))
            rv = np.zeros(self.dist.num_vals, dtype=bool)
            prev_ob = np.asarray(self.init_rng.sample(), dtype=int)

            rv[temp <= self.dist.log_edit_pmat[:,2]] = True
            rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob,3]

            return list(prev_ob), list(np.flatnonzero(rv))
        else:
            rv = []
            for i in range(size):
                rv.append(self.sample())
            return rv

    def sample_given(self, x: Sequence[Sequence[int]]) -> Sequence[int]:

        temp = np.log(self.rng.rand(self.dist.num_vals))
        rv = np.zeros(self.dist.num_vals, dtype=bool)
        prev_ob = np.asarray(x[-1], dtype=int)

        rv[temp <= self.dist.log_edit_pmat[:,2]] = True
        rv[prev_ob] = temp[prev_ob] <= self.dist.log_edit_pmat[prev_ob,3]

        return np.flatnonzero(rv).tolist()


class IntegerBernoulliEditAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals, init_acc, keys):
        self.pcnt     = np.zeros((num_vals, 3), dtype=np.float64)
        self.key      = keys
        self.num_vals = num_vals
        self.init_acc = init_acc
        self.tot_sum  = 0.0

    def update(self, x, weight, estimate):

        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[ to_rem], 0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[ to_add], 2] += weight

        self.tot_sum += weight

        if self.init_acc is not None:
            if estimate is not None:
                self.init_acc.update(x[0], weight, estimate.init_dist)
            else:
                self.init_acc.update(x[0], weight, None)

    def initialize(self, x, weight, rng):

        xx0 = np.asarray(x[0], dtype=int)
        xx1 = np.asarray(x[1], dtype=int)

        to_add = np.isin(xx1, xx0, invert=False)
        to_rem = np.isin(xx0, xx1, invert=True)

        self.pcnt[xx0[to_rem],  0] += weight
        self.pcnt[xx1[~to_add], 1] += weight
        self.pcnt[xx1[to_add],  2] += weight

        self.tot_sum += weight

        if self.init_acc is not None:
            self.init_acc.initialize(x[0], weight, rng)

    def seq_update(self, x, weights, estimate):

        sz, idx, xs, ys, ym, init_enc = x

        agg_cnt0 = np.bincount(xs[ym[0]], weights=weights[idx[ym[0]]])
        agg_cnt1 = np.bincount(xs[ym[1]], weights=weights[idx[ym[1]]])
        agg_cnt2 = np.bincount(xs[ym[2]], weights=weights[idx[ym[2]]])

        self.pcnt[:len(agg_cnt0), 0] += agg_cnt0
        self.pcnt[:len(agg_cnt1), 1] += agg_cnt1
        self.pcnt[:len(agg_cnt2), 2] += agg_cnt2
        self.tot_sum += weights.sum()

        if self.init_acc is not None:
            self.init_acc.seq_update(init_enc, weights, estimate.init_dist)

    def combine(self, suff_stat):
        self.pcnt    += suff_stat[0]
        self.tot_sum += suff_stat[1]

        if self.init_acc is not None:
            self.init_acc.combine(suff_stat[2])

        return self

    def value(self):
        init_val = None if self.init_acc is None else self.init_acc.value()
        return self.pcnt, self.tot_sum, init_val

    def from_value(self, x):
        self.pcnt = x[0]
        self.tot_sum = x[1]

        if self.init_acc is not None:
            self.init_acc.from_value(x[2])

        return self

    def key_merge(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                temp = stats_dict[self.key]
                stats_dict[self.key] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.key] = (self.pcnt, self.tot_sum)

        if self.init_acc is not None:
            self.init_acc.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.key]

        if self.init_acc is not None:
            self.init_acc.key_replace(stats_dict)

class IntegerBernoulliEditAccumulatorFactory(object):

    def __init__(self, num_vals, init_factory, keys):
        self.keys = keys
        self.init_factory = init_factory
        self.num_vals = num_vals

    def make(self):
        init_acc = None if self.init_factory is None else self.init_factory.make()
        return IntegerBernoulliEditAccumulator(self.num_vals, init_acc=init_acc, keys=self.keys)


class IntegerBernoulliEditEstimator(ParameterEstimator):

    def __init__(self, num_vals: int, init_estimator: ParameterEstimator = None,  min_prob: float = 1.0e-128, pseudo_count: Optional[float] = None, suff_stat: Optional[np.ndarray] = None, name=None, keys=None):
        self.num_vals      = num_vals
        self.keys          = keys
        self.pseudo_count  = pseudo_count
        self.suff_stat     = suff_stat
        self.name          = name
        self.min_prob      = min_prob
        self.init_est      = init_estimator

    def accumulatorFactory(self):
        init_factory = None if self.init_est is None else self.init_est.accumulatorFactory()
        return IntegerBernoulliEditAccumulatorFactory(self.num_vals, init_factory, self.keys)

    def estimate(self, nobs, suff_stat):

        if self.init_est is not None:
            init_dist = self.init_est.estimate(None, suff_stat[2])
        else:
            init_dist = None


        count_mat, tot_sum, _ = suff_stat

        if self.pseudo_count is not None and self.suff_stat is not None:

            p = self.pseudo_count
            s = self.suff_stat

            s1 = count_mat[:,0] + count_mat[:,2]
            s0 = (tot_sum - s1)

            log_s1 = np.log(s1 + p*(s[:,1] + s[:,3]))
            log_s0 = np.log(s0 + p*(s[:,0] + s[:,2]))

            log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)

            log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) + p*s[:,0]) - log_s0
            log_pmat[:, 1] = np.log(count_mat[:,0] + p*s[:,1]) - log_s1
            log_pmat[:, 2] = np.log(count_mat[:,1] + p*s[:,2]) - log_s0
            log_pmat[:, 3] = np.log(count_mat[:,2] + p*s[:,3]) - log_s1

        elif self.pseudo_count is not None and self.suff_stat is None:

            p = self.pseudo_count

            s1 = count_mat[:,0] + count_mat[:,2]
            s0 = tot_sum - s1

            log_s1 = np.log(s1 + p/2.0)
            log_s0 = np.log(s0 + p/2.0)

            log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)

            log_pmat[:, 2] = np.log(count_mat[:,1] + (p/4.0)) - log_s0
            log_pmat[:, 3] = np.log(count_mat[:,2] + (p/4.0)) - log_s1
            log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) + (p/4.0)) - log_s0
            log_pmat[:, 1] = np.log(count_mat[:,0] + (p/4.0)) - log_s1

        else:

            if suff_stat[1] == 0:
                log_pmat = np.zeros((self.num_vals,4), dtype=np.float64) + np.log(0.5)

            elif (self.min_prob is not None) and (self.min_prob > 0):

                s1 = count_mat[:, 0] + count_mat[:, 2]
                s0 = tot_sum - s1

                log_pmat = np.empty((self.num_vals,4), dtype=np.float64)
                log_pmat.fill(np.log(self.min_prob))

                if s0 != 0:
                    log_pmat[:, 0] = np.log(np.maximum((s0 - count_mat[:, 1]) / s0, self.min_prob))
                    log_pmat[:, 2] = np.log(np.maximum(count_mat[:, 1] / s0, self.min_prob))

                if s1 != 0:
                    log_pmat[:, 1] = np.log(np.maximum(count_mat[:, 0] / s1, self.min_prob))
                    log_pmat[:, 3] = np.log(np.maximum(count_mat[:, 2] / s1, self.min_prob))

            else:

                s1 = count_mat[:, 0] + count_mat[:, 2]
                s0 = tot_sum - s1

                log_pmat = np.empty((self.num_vals, 4), dtype=np.float64)
                log_pmat[:, 0] = np.log((s0 - count_mat[:, 1]) / s0)
                log_pmat[:, 1] = np.log(count_mat[:, 0] / s1)
                log_pmat[:, 2] = np.log(count_mat[:, 1] / s0)
                log_pmat[:, 3] = np.log(count_mat[:, 2] / s1)

        return IntegerBernoulliEditDistribution(log_pmat, init_dist=init_dist, name=self.name)
