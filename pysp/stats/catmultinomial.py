import numpy as np
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from pysp.arithmetic import maxrandint

class MultinomialDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, dist, len_dist=None, len_normalized=False, name=None):
        self.dist = dist
        self.len_dist = len_dist
        self.len_normalized = len_normalized
        self.name = name

    def __str__(self):
        s1 = str(self.dist)
        s2 = str(self.len_dist)
        s3 = repr(self.len_normalized)
        s4 = repr(self.name)
        return 'MultinomialDistribution(%s, len_dist=%s, len_normalized=%s, name=%s)'%(s1, s2, s3, s4)

    def density(self, x):
        return np.exp(self.log_density(x))

    def log_density(self, x):
        rv = 0.0
        cc = 0.0
        for i in range(len(x)):
            rv += self.dist.log_density(x[i][0])*x[i][1]
            cc += x[i][1]

        if self.len_normalized and len(x) > 0:
            rv /= cc

        if self.len_dist is not None:
            rv += self.len_dist.log_density(cc)

        return rv



    def seq_log_density(self, x):

        idx, icnt, inz, enc_seq, enc_nseq, enc_w, enc_ww = x

        ll = self.dist.seq_log_density(enc_seq)
        ll_sum = np.bincount(idx, weights=ll*enc_w, minlength=len(icnt))

        if self.len_normalized:
            ll_sum *= icnt

        if self.len_dist is not None and enc_nseq is not None:
            nll = self.len_dist.seq_log_density(enc_nseq)
            ll_sum += nll

        return ll_sum



    def seq_encode(self, x):

        tx   = []
        nx   = []
        tidx = []
        cc   = []
        ccc  = []

        for i in range(len(x)):
            nx.append(len(x[i]))
            aa = 0
            for j in range(len(x[i])):
                tidx.append(i)
                tx.append(x[i][j][0])
                cc.append(x[i][j][1])
                aa += x[i][j][1]
            ccc.append(aa)

        rv1 = np.asarray(tidx, dtype=int)
        rv2 = np.asarray(ccc, dtype=float)
        rv3 = (rv2 != 0)
        rv6 = np.asarray(cc, dtype=float)
        rv7 = np.asarray(ccc, dtype=float)

        rv2[rv3] = 1.0/rv2[rv3]
        #rv2[rv3] = 1.0

        rv4 = self.dist.seq_encode(tx)

        if self.len_dist is not None:
            rv5 = self.len_dist.seq_encode(ccc)
        else:
            rv5 = None

        return rv1, rv2, rv3, rv4, rv5, rv6, rv7

    def sampler(self, seed=None):
        return MultinomialSampler(self, seed)

    def estimator(self, pseudo_count=None):
        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        dist_est = self.dist.estimator(pseudo_count=pseudo_count)
        return MultinomialEstimator(dist_est, len_estimator=len_est, len_normalized=self.len_normalized, name=self.name)


class MultinomialSampler(object):
    def __init__(self, dist, seed=None):
        self.dist        = dist
        self.rng         = RandomState(seed)
        self.distSampler = self.dist.dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.lenSampler  = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size=None):

        if size is None:
            n = self.lenSampler.sample()
            rv = dict()
            for i in range(n):
                v = self.distSampler.sample()
                if v in rv:
                    rv[v] +=1
                else:
                    rv[v] = 1
            return list(rv.items())

        else:
            return [self.sample() for i in range(size)]


class MultinomialAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulator, len_normalized, len_accumulator=None, keys=None):
        self.accumulator = accumulator
        self.len_accumulator = len_accumulator
        self.key = keys
        self.len_normalized = len_normalized

    def update(self, x, weight, estimate):

        xx = [u[0] for u in x]
        cc = [u[1] for u in x]
        ss = sum(cc)

        if estimate is None:

            w = weight / ss if (self.len_normalized and ss > 0) else weight

            for i in range(len(x)):
                self.accumulator.update(x[i][0], w*x[i][1], None)

            if self.len_accumulator is not None:
                self.len_accumulator.update(ss, weight, None)

        else:
            w = weight / ss if (self.len_normalized and ss > 0) else weight

            for i in range(len(x)):
                self.accumulator.update(x[i][0], w*x[i][1], estimate.dist)

            if self.len_accumulator is not None:
                self.len_accumulator.update(ss, weight, estimate.len_dist)

    def initialize(self, x, weight, rng):

        cc = [u[1] for u in x]
        ss = sum(cc)
        w  = weight / ss if self.len_normalized else weight

        for i in range(len(x)):
            self.accumulator.initialize(x[i][0], w*x[i][1], rng)

        if self.len_accumulator is not None:
            self.len_accumulator.initialize(ss, weight, rng)


    def combine(self, suff_stat):
        self.accumulator.combine(suff_stat[0])
        if self.len_accumulator is not None:
            self.len_accumulator.combine(suff_stat[1])
        return self

    def value(self):
        if self.len_accumulator is not None:
            return self.accumulator.value(), self.len_accumulator.value()
        else:
            return self.accumulator.value(), None

    def from_value(self, x):
        self.accumulator.from_value(x[0])
        if self.len_accumulator is not None:
            self.len_accumulator.from_value(x[1])
        return self


    def seq_update(self, x, weights, estimate):
        idx, icnt, inz, enc_seq, enc_nseq, enc_w, enc_ww = x

        w = weights[idx]*icnt[idx] if self.len_normalized else weights[idx]
        w *= enc_w

        self.accumulator.seq_update(enc_seq, w, estimate.dist if estimate is not None else None)

        if self.len_accumulator is not None:
            self.len_accumulator.seq_update(enc_nseq, weights*enc_ww, estimate.len_dist if estimate is not None else None)

    def key_merge(self, stats_dict):

        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        self.accumulator.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)


    def key_replace(self, stats_dict):

        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        self.accumulator.key_replace(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)


class MultinomialAccumulatorFactory(object):

    def __init__(self, est_factory, len_normalized, len_factory, keys):
        self.est_factory = est_factory
        self.len_normalized = len_normalized
        self.len_factory = len_factory
        self.keys = keys

    def make(self):
        len_acc = None if self.len_factory is None else self.len_factory.make()
        return MultinomialAccumulator(self.est_factory.make(), self.len_normalized, len_accumulator=len_acc, keys=self.keys)


class MultinomialEstimator(ParameterEstimator):

    def __init__(self, estimator, name=None, len_estimator=None, len_dist=None, len_normalized=False, keys=None):
        self.name = name
        self.estimator = estimator
        self.len_estimator = len_estimator
        self.len_dist = len_dist
        self.keys = keys
        self.len_normalized=len_normalized

    def accumulatorFactory(self):
        est_factory = self.estimator.accumulatorFactory()
        len_factory = None if self.len_estimator is None else self.len_estimator.accumulatorFactory()
        return MultinomialAccumulatorFactory(est_factory, self.len_normalized, len_factory, self.keys)

    def estimate(self, nobs, suff_stat):

        if self.len_estimator is None:
            len_dist = self.len_dist
        else:
            len_dist = self.len_estimator.estimate(nobs, suff_stat[1])

        dist = self.estimator.estimate(nobs, suff_stat[0])

        return MultinomialDistribution(dist, len_dist, len_normalized=self.len_normalized, name=self.name)

