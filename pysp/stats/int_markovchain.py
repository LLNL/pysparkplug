from typing import Optional
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator
import numpy as np
from scipy.sparse import csr_matrix
import collections
from pysp.arithmetic import maxrandint


class IntegerMarkovChainDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, num_values, cond_dist, lag=1, init_dist=None, len_dist=None, min_prob=0.0, name=None):

        self.lag          = lag
        self.name         = name
        self.init_dist    = init_dist
        self.cond_dist    = np.asarray(cond_dist)
        self.len_dist     = len_dist
        self.min_prob     = min_prob
        self.num_values   = num_values

    def __str__(self):

        s1 = repr(self.num_values)
        s2 = repr(self.cond_dist.tolist())
        s3 = repr(self.lag)
        s4 = repr(self.init_dist) if self.init_dist is None else str(self.init_dist)
        s5 = repr(self.len_dist) if self.len_dist is None else str(self.len_dist)
        s6 = repr(self.name)

        return 'IntegerMarkovChainDistribution(%s, %s, lag=%s, init_dist=%s, len_dist=%s, name=%s)' % (s1, s2, s3, s4, s5, s6)

    def density(self, x):
        return np.exp(self.log_density(x))

    def log_density(self, x):

        rv = 0.0
        lag = self.lag

        if len(x) >= lag:

            mshape = [self.num_values]*lag

            if self.init_dist is not None:
                rv += self.init_dist.log_density(x[:lag])

            for i in range(len(x)-lag):
                idx = np.ravel_multi_index(x[i:(i+lag)], mshape)
                rv += np.log(self.cond_dist[idx, x[i+lag]])

        if self.len_dist is not None:
            rv += self.len_dist.log_density(len(x))

        return rv

    def seq_encode(self, x):

        lag = self.lag

        cnt  = len(x)
        lens = np.asarray([len(u) for u in x])
        lcnt = (lens >= lag).sum()
        rcnt = np.maximum(lens - lag, 0).sum()

        init_entries = np.zeros(lcnt, dtype=object)
        seq_entries = np.zeros(rcnt, dtype=object)

        iidx = []
        sidx = []
        slen = []

        i0 = 0
        i1 = 0

        for i in range(len(x)):
            xx = x[i]
            slen.append(max(len(xx)-lag+1, 0))

            if len(xx) < lag:
                continue

            iidx.append(i)
            init_entries[i0] = tuple(xx[:lag])
            i0 += 1

            for j in range(len(xx)-lag):
                sidx.append(i)
                seq_entries[i1] = (tuple(xx[j:(j+lag)]), xx[j+lag])
                i1 += 1


        seq_values, seq_idx = np.unique(seq_entries, return_inverse=True)

        iidx = np.asarray(iidx, dtype=np.int32)
        sidx = np.asarray(sidx, dtype=np.int32)
        slen = np.asarray(slen, dtype=np.int32)

        len_enc = None
        if self.len_dist is not None:
            len_enc = self.len_dist.seq_encode(slen)

        init_enc = None
        if self.init_dist is not None:
            init_enc = self.init_dist.seq_encode(init_entries)

        return slen, iidx, sidx, seq_idx, seq_values, init_enc, len_enc


    def seq_log_density(self, x):

        slen, iidx, sidx, seq_idx, seq_values, init_enc, len_enc = x

        left_idx = [np.ravel_multi_index(u[0], [self.num_values] * self.lag) for u in seq_values]
        right_idx = np.asarray([u[1] for u in seq_values])
        temp_prob = np.log(self.cond_dist[left_idx, right_idx])
        temp_prob = temp_prob[seq_idx]

        rv = np.bincount(sidx, weights=temp_prob, minlength=len(slen))

        if self.init_dist is not None:
            rv[iidx] += self.init_dist.seq_log_density(init_enc)

        if self.len_dist is not None and len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def sampler(self, seed=None):
        return IntegerMarkovChainSampler(self, seed)

    def estimator(self, pseudo_count=None):
        return None


class IntegerMarkovChainSampler(object):

    def __init__(self, dist: IntegerMarkovChainDistribution, seed):

        rng = np.random.RandomState(seed)
        seeds = rng.randint(0, maxrandint, size=3)

        self.dist = dist
        self.rng = rng
        self.trans_sampler = np.random.RandomState(seeds[0])
        if self.dist.init_dist is not None:
            self.init_sampler = dist.init_dist.sampler(seeds[1])
        if self.dist.len_dist is not None:
            self.len_sampler = dist.len_dist.sampler(seeds[2])

    def sample(self, size=None):

        if size is not None:
            return [self.sample() for i in range(size)]

        else:
            cnt = self.len_sampler.sample()
            lag = self.dist.lag
            nval = self.dist.num_values
            mshape = [nval] * lag

            if cnt >=lag:
                rv = self.init_sampler.sample()
                for i in range(lag, cnt):
                    idx = np.ravel_multi_index(rv[-lag:], mshape)
                    rv.append(self.trans_sampler.choice(nval, p=self.dist.cond_dist[idx,:]))
                return rv
            else:
                return []

    def sample_given(self, x):

        lag = self.dist.lag
        nval = self.dist.num_values
        mshape = [nval] * lag
        idx = np.ravel_multi_index(x[-lag:], mshape)

        return self.trans_sampler.choice(nval, p=self.dist.cond_dist[idx, :])



class IntegerMarkovChainAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, lag, init_accumulator, len_accumulator):

        self.lag = lag
        self.trans_count_map = dict()
        self.len_accumulator = len_accumulator
        self.init_accumulator = init_accumulator
        self.max_value = -1

    def update(self, x, weight, estimate):

        lag = self.lag

        if len(x) >= lag and self.len_accumulator is not None:
            l_est = None
            if estimate is not None and estimate.len_dist is not None:
                l_est = estimate.len_dist
            self.len_accumulator.update(len(x)-lag, weight, l_est)

        if len(x) >= lag and self.init_accumulator is not None:
            i_est = None
            if estimate is not None and estimate.init_dist is not None:
                i_est = estimate.init_dist
            self.init_accumulator.update(x[:lag], weight, i_est)

        for i in range(len(x)-lag):
            entry = (tuple(x[i:(i+lag)]), x[i+lag])
            self.trans_count_map[entry] = self.trans_count_map.get(entry, 0) + weight


    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_update(self, x, weights, estimate):

        slen, iidx, sidx, seq_idx, seq_values, init_enc, len_enc = x

        scnt = np.bincount(seq_idx, weights=weights[sidx])

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(seq_values, scnt))
        else:
            for k, v in zip(seq_values, scnt):
                self.trans_count_map[k] = self.trans_count_map.get(k,0) + v

        if self.init_accumulator is not None:
            i_est = None if estimate is None else estimate.init_dist
            self.init_accumulator.seq_update(init_enc, weights[iidx], i_est)

        if self.len_accumulator is not None:
            l_est = None if estimate is None else estimate.len_dist
            self.len_accumulator.seq_update(len_enc, weights, l_est)


    def combine(self, suff_stat):

        for k,v in suff_stat[0].items():
            self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        if self.init_accumulator is not None and suff_stat[1] is not None:
            self.init_accumulator = self.init_accumulator.combine(suff_stat[1])

        if self.len_accumulator is not None and suff_stat[2] is not None:
            self.len_accumulator = self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self):
        len_val = None if self.len_accumulator is None else self.len_accumulator.value()
        init_val = None if self.init_accumulator is None else self.init_accumulator.value()

        return self.trans_count_map, init_val, len_val

    def from_value(self, x):

        self.trans_count_map = x[0]

        if self.init_accumulator is not None and x[1] is not None:
            self.init_accumulator = self.init_accumulator.from_value(x[1])

        if self.len_accumulator is not None and x[2] is not None:
            self.len_accumulator = self.len_accumulator.from_value(x[2])

        return self


class IntegerMarkovChainAccumulatorFactory(object):

    def __init__(self, lag, init_factory, len_factory):
        self.lag = lag
        self.init_factory = init_factory
        self.len_factory = len_factory

    def make(self):
        init_acc = None if self.init_factory is None else self.init_factory.make()
        len_acc = None if self.len_factory is None else self.len_factory.make()

        return IntegerMarkovChainAccumulator(self.lag, init_acc, len_acc)


class IntegerMarkovChainEstimator(ParameterEstimator):

    def __init__(self, num_values, lag=1, init_estimator=None, len_estimator=None, init_dist=None, len_dist=None, pseudo_count=None, name=None):

        self.name = name
        self.pseudo_count = pseudo_count
        self.len_dist = len_dist
        self.init_dist = init_dist
        self.lag = lag
        self.num_values = num_values
        self.init_estimator = init_estimator
        self.len_estimator = len_estimator

    def accumulatorFactory(self):

        len_factory = None if self.len_estimator is None else self.len_estimator.accumulatorFactory()
        init_factory = None if self.init_estimator is None else self.init_estimator.accumulatorFactory()

        return IntegerMarkovChainAccumulatorFactory(self.lag, init_factory, len_factory)

    def estimate(self, nobs, suff_stat):

        trans_count_map, init_ss, len_ss = suff_stat
        lag = self.lag

        len_dist = self.len_dist
        if self.len_estimator is not None:
            len_dist = self.len_estimator.estimate(None, len_ss)

        init_dist = self.init_dist
        if self.init_estimator is not None:
            init_dist = self.init_estimator.estimate(None, init_ss)


        num_values = 1 + max([max(max(u[0]),u[1]) for u in trans_count_map.keys()])

        cond_mat = np.zeros((num_values**lag, num_values), dtype=np.float32)

        vv = list(trans_count_map.items())
        yidx = np.asarray([np.ravel_multi_index(u[0], [num_values]*lag) for u,_ in vv])
        xidx = np.asarray([u[1] for u,_ in vv])
        zidx = np.asarray([u[1] for u in vv])

        cond_mat[yidx, xidx] = zidx

        if self.pseudo_count is not None:
            cond_mat += self.pseudo_count

        cond_mat /= cond_mat.sum(axis=1, keepdims=True)


        return IntegerMarkovChainDistribution(num_values, cond_mat, init_dist=init_dist, lag=lag, len_dist=len_dist, name=self.name)

