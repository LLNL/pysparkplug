from typing import Optional, List, Tuple, Any
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
import math
from pysp.utils.optsutil import countByValue
from pysp.arithmetic import maxrandint


class HiddenAssociationDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, cond_dist, given_dist=None, null_prob=None, null_dist=None, len_dist=None, name=None):

        self.cond_dist = cond_dist
        self.len_dist  = len_dist
        self.given_dist = given_dist
        self.null_dist = null_dist
        self.null_prob = null_prob
        self.name = name


    def __str__(self):
        s1 = repr(self.cond_dist)
        s2 = repr(self.given_dist)
        s3 = repr(self.null_prob)
        s4 = repr(self.null_dist)
        s5 = repr(self.len_dist)
        s6 = repr(self.name)
        return 'HiddenAssociationDistribution(%s, given_dist=%s, null_prob=%s, null_dist=%s, len_dist=%s, name=%s)' % (s1, s2, s3, s4, s5, s6)

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x: Tuple[List[Tuple[Any,float]], List[Tuple[Any,float]]]) -> float:

        rv = 0
        nn = 0
        for x1, c1 in x[1]:
            cc = 0
            nn += c1
            ll = -np.inf
            for x0, c0 in x[0]:
                tt = self.cond_dist.log_density((x0, x1)) + math.log(c0)
                cc += c0

                if tt == -np.inf:
                    continue

                if ll > tt:
                    ll = math.log1p(math.exp(tt-ll)) + ll
                else:
                    ll = math.log1p(math.exp(ll-tt)) + tt

            ll -= math.log(cc)
            rv += ll * c1

        if self.given_dist is not None:
            rv += self.given_dist.log_density(x[0])

        if self.len_dist is not None:
            rv += self.len_dist.log_density(nn)

        return rv

    def seq_log_density(self, x):
        return np.asarray([self.log_density(xx) for xx in x])

    def seq_encode(self, x):
        return x

    def sampler(self, seed=None):
        return HiddenAssociationSampler(self, seed)


class HiddenAssociationSampler(object):

    def __init__(self, dist: HiddenAssociationDistribution, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.dist = dist

        self.cond_sampler = dist.cond_dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.idx_sampler = np.random.RandomState(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

        if self.dist.given_dist is not None:
            self.given_sampler = self.dist.given_dist.sampler(seed=self.rng.randint(0, maxrandint))


    def sample(self, size: Optional[int] = None):

        if size is None:

            prev_obs = self.given_sampler.sample()
            cnt = self.len_sampler.sample()
            rng = np.random.RandomState(self.idx_sampler.randint(0, maxrandint))
            rv = []
            pp = np.asarray([u[1] for u in prev_obs], dtype=float)
            pp /= pp.sum()

            for i in rng.choice(len(prev_obs), p=pp, size=cnt):
                rv.append(self.cond_sampler.sample_given(prev_obs[i][0]))

            rv = list(countByValue(rv).items())

            return prev_obs, rv

        else:
            return [self.sample() for i in range(size)]

    def sample_given(self, x):
        cnt = self.len_sampler.sample()
        rng = np.random.RandomState(self.idx_sampler.randint(0, maxrandint))
        rv = []
        pp = np.asarray([u[1] for u in x], dtype=float)
        pp /= pp.sum()

        for i in rng.choice(len(x), p=pp, size=cnt):
            rv.append(self.cond_sampler.sample_given(x[i][0]))

        rv = list(countByValue(rv).items())

        return rv

class HiddenAssociationAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, cond_acc, given_acc=None, size_acc=None, keys=(None, None)):

        self.cond_accumulator  = cond_acc
        self.given_accumulator = given_acc
        self.size_accumulator  = size_acc

        self.init_key = keys[0]
        self.trans_key = keys[1]

    def update(self, x, weight, estimate):

        nn = 0
        pv = np.zeros(len(x[0]))

        for x1, c1 in x[1]:
            cc = 0
            nn += c1
            ll = -np.inf

            for i, (x0, c0) in enumerate(x[0]):
                tt = estimate.cond_dist.log_density((x0, x1)) + math.log(c0)
                cc += c0
                pv[i] = tt

                if tt == -np.inf:
                    continue

                if ll > tt:
                    ll = math.log1p(math.exp(tt-ll)) + ll
                else:
                    ll = math.log1p(math.exp(ll-tt)) + tt

            pv -= ll
            np.exp(pv, out=pv)

            for i, (x0, c0) in enumerate(x[0]):
                self.cond_accumulator.update((x0, x1), pv[i]*c1*weight, estimate.cond_dist)

        if self.given_accumulator is not None:
            given_dist = None if estimate is None else estimate.given_dist
            self.given_accumulator.update(x[0], weight, given_dist)

        if self.size_accumulator is not None:
            len_dist = None if estimate is None else estimate.len_dist
            self.size_accumulator.update(nn, weight, len_dist)

    def initialize(self, x, weight, rng):

        w = rng.dirichlet(np.ones(len(x[0])),size=len(x[1]))
        nn = 0
        for j, (x1, c1) in enumerate(x[1]):
            nn += c1
            for i, (x0, c0) in enumerate(x[0]):
                self.cond_accumulator.initialize((x0, x1), w[j,i]*weight, rng)

        if self.given_accumulator is not None:
            self.given_accumulator.initialize(x[0], weight, rng)

        if self.size_accumulator is not None:
            self.size_accumulator.initialize(nn, weight, rng)

    def seq_update(self, x, weights, estimate):

        for xx, ww in zip(x, weights):
            self.update(xx, ww, estimate)

    def combine(self, suff_stat):

        cond_acc, given_acc, size_acc = suff_stat

        self.cond_accumulator.combine(cond_acc)

        if self.given_accumulator is not None:
            self.given_accumulator.combine(given_acc)
        if self.size_accumulator is not None:
            self.size_accumulator.combine(size_acc)

        return self

    def value(self):
        given_val = None if self.given_accumulator is None else self.given_accumulator.value()
        size_val = None if self.size_accumulator is None else self.size_accumulator.value()
        return self.cond_accumulator.value(), given_val, size_val

    def from_value(self, x):

        cond_acc, given_acc, size_acc = x

        self.cond_accumulator.from_value(cond_acc)
        if self.given_accumulator is not None:
            self.given_accumulator.from_value(given_acc)
        if self.size_accumulator is not None:
            self.size_accumulator.from_value(size_acc)

        return self

    def key_merge(self, stats_dict):

        if self.size_accumulator is not None:
            self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict):

        if self.size_accumulator is not None:
            self.size_accumulator.key_replace(stats_dict)


class HiddenAssociationAccumulatorFactory(object):

    def __init__(self, cond_factory, given_factory, len_factory, keys):
        self.cond_factory = cond_factory
        self.given_factory = given_factory
        self.len_factory = len_factory
        self.keys = keys

    def make(self):
        len_acc = None if self.len_factory is None else self.len_factory.make()
        given_acc = None if self.given_factory is None else self.given_factory.make()
        return HiddenAssociationAccumulator(self.cond_factory.make(), given_acc, len_acc, self.keys)


class HiddenAssociationEstimator(ParameterEstimator):

    def __init__(self, cond_estimator, given_estimator=None, len_estimator=None, suff_stat=None, pseudo_count=None, name=None, keys=(None, None)):
        self.keys = keys
        self.len_estimator = len_estimator
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.cond_estimator = cond_estimator
        self.given_estimator = given_estimator
        self.name = name

    def accumulatorFactory(self):
        len_factory = None if self.len_estimator is None else self.len_estimator.accumulatorFactory()
        given_factory = None if self.given_estimator is None else self.given_estimator.accumulatorFactory()
        cond_factory = self.cond_estimator.accumulatorFactory()
        return HiddenAssociationAccumulatorFactory(cond_factory, given_factory, len_factory, self.keys)

    def estimate(self, nobs, suff_stat):

        cond_stats, given_stats, size_stats = suff_stat

        cond_dist = self.cond_estimator.estimate(None, cond_stats)

        if self.given_estimator is not None:
            given_dist = self.given_estimator.estimate(nobs, given_stats)
        else:
            given_dist = None

        if self.len_estimator is not None:
            len_dist = self.len_estimator.estimate(nobs, size_stats)
        else:
            len_dist = None


        return HiddenAssociationDistribution(cond_dist=cond_dist, given_dist=given_dist, len_dist=len_dist, name=self.name)


