import pysp.utils.vector as vec
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from pysp.stats.mixture import MixtureDistribution
from pysp.stats.sequence import SequenceDistribution
import numpy as np

class HierarchicalMixtureDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, topics, mixture_weights, topic_weights, len_dist=None, name=None, keys=(None, None)):

        with np.errstate(divide='ignore'):

            self.topics           = topics
            self.num_topics       = len(topics)
            self.num_mixtures     = len(mixture_weights)
            self.w                = np.asarray(mixture_weights, dtype=np.float64)
            self.logW             = np.log(self.w)
            self.taus             = np.asarray(topic_weights, dtype=np.float64)
            self.logTaus          = np.log(self.taus)
            self.len_dist         = len_dist
            self.name             = name
            self.keys = keys

    def __str__(self):
        s1 = '[' + ','.join([str(u) for u in self.topics]) + ']'
        s2 = repr(list(self.w))
        s3 = repr(list(map(list,self.taus)))
        s4 = repr(self.len_dist) if self.len_dist is None else str(self.len_dist)
        s5 = repr(self.name)
        s6 = repr(self.keys)
        return 'HierarchicalMixtureDistribution(%s, %s, %s, len_dist=%s, name=%s, keys=%s)'%(s1, s2, s3, s4, s5, s6)

    def log_density(self, x):
        return self.seq_log_density(self.seq_encode([x]))[0]

    def posterior(self, x):
        return self.seq_posterior(self.seq_encode([x]))[0,:]

    def component_log_density(self, x):
        return self.seq_component_log_density(self.seq_encode([x]))[0, :]

    def to_mixture(self):
        topics = [SequenceDistribution(MixtureDistribution(self.topics, self.taus[i,:]), len_dist=self.len_dist) for i in range(self.num_mixtures)]
        return MixtureDistribution(topics, self.w)

    def seq_component_log_density(self, x):

        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        if (sz > 0) and np.all(cnt == 0):
            return np.zeros(sz, dtype=np.float64)
        elif sz == 0:
            return np.zeros(0, dtype=np.float64)

        # Compute P(data|topic) for each topic

        ll_mat = np.zeros((tsz, self.num_topics), dtype=np.float64)
        rv = np.zeros((sz, self.num_mixtures), dtype=np.float64)

        for i in range(self.num_topics):
            ll_mat[:, i] = self.topics[i].seq_log_density(enc_data)

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = 0.0
        ll_max[bad_rows] = 0.0

        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)

        # Compute ln P(data | mixture)
        ll_mat = np.dot(ll_mat, self.taus.T)
        np.log(ll_mat, out=ll_mat)
        ll_mat += ll_max

        # Compute ln P(bag of data | mixture)
        for i in range(self.num_mixtures):
            rv[:, i] = np.bincount(idx, weights=ll_mat[:, i], minlength=sz)

        return rv


    def seq_log_density(self, x):

        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        # Compute ln P(bag of data | mixture)
        rv = self.seq_component_log_density(x)

        # Compute ln P(bag of data, mixture)
        rv += self.logW

        # Compute ln P(bag of data)
        ll_max2 = np.max(rv, axis=1, keepdims=True)
        rv     -= ll_max2
        np.exp(rv, out=rv)
        ll_sum  = rv.sum(axis=1, keepdims=True)
        np.log(ll_sum, out=ll_sum)
        ll_sum += ll_max2

        rv = ll_sum.flatten()

        if self.len_dist is not None:
            rv += self.len_dist.seq_log_density(enc_len)

        return rv

    def seq_posterior(self, x):

        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        # Compute ln P(bag of data | mixture)
        rv = self.seq_component_log_density(x)

        # Compute ln P(bag of data, mixture)
        rv += self.logW

        # Compute ln P(bag of data)
        ll_max2 = np.max(rv, axis=1, keepdims=True)
        rv -= ll_max2
        np.exp(rv, out=rv)
        rv /= np.sum(rv, axis=1, keepdims=True)

        return rv

    def seq_encode(self, x):

        sx  = []
        idx = []
        cnt = []

        for i in range(len(x)):
            idx.extend([i]*len(x[i]))
            sx.extend(x[i])
            cnt.append(len(x[i]))

        if self.len_dist is not None:
            enc_len = self.len_dist.seq_encode(cnt)
        else:
            enc_len = None

        idx = np.asarray(idx, dtype=np.int32)
        cnt = np.asarray(cnt, dtype=np.int32)

        enc_data = self.topics[0].seq_encode(sx)

        return len(x), idx, cnt, enc_data, enc_len


    def sampler(self, seed=None):
        return HierarchicalMixtureSampler(self, seed)

    def estimator(self, pseudo_count=None):

        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_est = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]

        return HierarchicalMixtureEstimator(comp_est, self.num_mixtures, len_estimator=len_est, pseudo_count=pseudo_count, name=self.name, keys=self.keys)


class HierarchicalMixtureSampler(object):

    def __init__(self, dist: HierarchicalMixtureDistribution, seed=None):
        self.rng     = np.random.RandomState(seed)
        self.dist    = dist
        self.sampler = dist.to_mixture().sampler(seed)

    def sample(self, size=None):
        return self.sampler.sample(size=size)

class HierarchicalMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators, num_mixtures, len_accumulator, keys=(None, None)):

        self.accumulators = accumulators
        self.num_topics   = len(accumulators)
        self.num_mixtures = num_mixtures
        self.comp_counts  = vec.zeros((self.num_mixtures, self.num_topics))
        self.len_accumulator = len_accumulator
        self.weight_key   = keys[0]
        self.comp_key     = keys[1]

    def update(self, x, weight, estimate):
        self.seq_update(estimate.seq_encode([x]), np.asarray([weight]), estimate)

    def initialize(self, x, weight, rng):

        idx1 = rng.choice(self.num_mixtures)

        for j in range(len(x)):
            idx2 = rng.choice(self.num_topics)

            for i in range(self.num_topics):
                w = weight if i == idx2 else 0.0
                self.accumulators[i].initialize(x[j], w, rng)
                self.comp_counts[idx1, i] += w

        if self.len_accumulator is not None:
            self.len_accumulator.initialize(len(x), weight, rng)

    def seq_update(self, x, weights, estimate):

        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        ll_mat = np.zeros((tsz, self.num_topics))
        ll_mat.fill(-np.inf)
        rv  = np.zeros((sz, self.num_mixtures))
        rv3 = np.zeros((tsz, self.num_topics))

        for i in range(self.num_topics):
            ll_mat[:, i]  = estimate.topics[i].seq_log_density(enc_data)

        ll_max  = ll_mat.max(axis = 1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = 0.0
        ll_max[bad_rows] = 0.0

        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)

        ll_mat_t = np.dot(ll_mat, estimate.taus.T)
        ll_mat_t2 = np.log(ll_mat_t)

        ll_max = np.bincount(idx, weights=ll_max.flatten(), minlength=sz)
        for i in range(self.num_mixtures):
            rv[:, i] = np.bincount(idx, weights=ll_mat_t2[:,i], minlength=sz)

        rv += estimate.logW
        rv += ll_max[:,None]
        ll_max2 = np.max(rv, axis=1, keepdims=True)
        rv -= ll_max2

        np.exp(rv, out=rv)
        ll_sum  = rv.sum(axis=1, keepdims=True)
        rv  /= ll_sum
        rv  = rv[idx, :]
        ww = np.reshape(weights[idx], (-1, 1))

        for i in range(self.num_mixtures):
            temp = estimate.taus[i, None, :] * (rv[:, i, None] / ll_mat_t[:, i, None])
            temp *= ll_mat
            temp *= ww
            rv3 += temp
            self.comp_counts[i, :] += temp.sum(axis=0)

        for i in range(self.num_topics):
            self.accumulators[i].seq_update(enc_data, rv3[:,i], estimate.topics[i])


        if self.len_accumulator is not None:
            len_est = None if estimate is None else estimate.len_dist
            self.len_accumulator.seq_update(enc_len, weights, len_est)


    def combine(self, suff_stat):

        self.comp_counts += suff_stat[0]
        for i in range(self.num_topics):
            self.accumulators[i].combine(suff_stat[1][i])

        if self.len_accumulator is not None:
            self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self):
        len_value = None if self.len_accumulator is None else self.len_accumulator.value()
        return self.comp_counts, tuple([u.value() for u in self.accumulators]), len_value

    def from_value(self, x):
        self.comp_counts = x[0]
        for i in range(self.num_topics):
            self.accumulators[i].from_value(x[1][i])

        if self.len_accumulator is not None:
            self.len_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict):

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                for i in range(len(acc)):
                     acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict):

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

class HierarchicalMixtureEstimatorAccumulatorFactory(ParameterEstimator):

    def __init__(self, factories, num_mixtures, len_factory, keys):
        self.factories = factories
        self.num_mixtures = num_mixtures
        self.dim = len(factories)
        self.len_factory = len_factory
        self.keys = keys

    def make(self):
        len_acc = None if self.len_factory is None else self.len_factory.make()
        return HierarchicalMixtureEstimatorAccumulator([self.factories[i].make() for i in range(self.dim)], self.num_mixtures, len_acc, self.keys)


class HierarchicalMixtureEstimator(object):
    
    def __init__(self, estimators, num_mixtures, len_estimator=None, len_dist=None, suff_stat=None, pseudo_count=None, name=None, keys=(None, None)):
        # self.estimator   = estimator
        # self.dim         = num_components
        self.num_components = len(estimators)
        self.num_mixtures = num_mixtures
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.len_estimator = len_estimator
        self.keys = keys
        self.len_dist = len_dist
        self.name = name

    def accumulatorFactory(self):
        est_factories = [u.accumulatorFactory() for u in self.estimators]
        len_factory = None if self.len_estimator is None else self.len_estimator.accumulatorFactory()
        return HierarchicalMixtureEstimatorAccumulatorFactory(est_factories, self.num_mixtures, len_factory, self.keys)

    def estimate(self, nobs, suff_stat):

        num_mixtures = self.num_mixtures
        num_components = self.num_components
        counts, comp_suff_stats, len_suff_stats = suff_stat

        if len_suff_stats is not None:
            len_dist = self.len_estimator.estimate(None, len_suff_stats)
        else:
            len_dist = self.len_dist

        components = [self.estimators[i].estimate(None, comp_suff_stats[i]) for i in range(num_components)]

        if self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / (num_components*num_mixtures)
            taus = counts + p
            w = taus.sum(axis=1, keepdims=True)
            taus /= w
            w /= w.sum()
            w  = w.flatten()

            #elif self.pseudo_count is not None and self.suff_stat is not None:
            #    w = (counts + self.suff_stat*self.pseudo_count) / (counts.sum() + self.pseudo_count)
        else:

            taus  = counts
            w     = taus.sum(axis=1, keepdims=True)
            taus /= w
            w    /= w.sum()
            w     = w.flatten()

        return HierarchicalMixtureDistribution(components, w, taus, len_dist=len_dist, name=self.name, keys=self.keys)
