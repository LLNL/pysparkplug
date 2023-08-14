from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import numpy as np
import pysp.utils.vector as vec
from pysp.arithmetic import maxrandint


class MixtureDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, components, w, name=None):
        self.components = components
        self.num_components = len(components)
        self.w = np.asarray(w)
        self.zw = (self.w == 0.0)
        self.log_w = np.log(w+self.zw)
        self.log_w[self.zw] = -np.inf
        self.name = name

    def __str__(self):
        s1 = ','.join([str(u) for u in self.components])
        s2 = repr(list(self.w))
        s3 = repr(self.name)
        return 'MixtureDistribution([%s], %s, name=%s)' % (s1, s2, s3)

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def component_log_density(self, x):
        return np.asarray([m.log_density(x) for m in self.components], dtype=np.float64)

    def posterior(self, x):

        comp_log_density = np.asarray([m.log_density(x) for m in self.components])
        comp_log_density += self.log_w
        comp_log_density[self.w == 0] = -np.inf

        max_val = np.max(comp_log_density)

        if max_val == -np.inf:
            return self.w.copy()
        else:
            comp_log_density -= max_val
            np.exp(comp_log_density, out=comp_log_density)
            comp_log_density /= comp_log_density.sum()
            return comp_log_density


    def seq_log_density(self, x):

        enc_data = x
        ll_mat = None

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(enc_data)
                if ll_mat is None:
                    ll_mat = np.zeros((len(temp), self.num_components))
                    ll_mat.fill(-np.inf)
                ll_mat[:, i]  = temp
                ll_mat[:, i] += self.log_w[i]

        ll_max  = ll_mat.max(axis = 1, keepdims=True)

        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max

            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            return ll_sum.flatten()

        else:
            ll_mat = ll_mat[good_rows, :]
            ll_max = ll_max[good_rows]

            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max
            rv = np.zeros(good_rows.shape, dtype=float)
            rv[good_rows] = ll_sum.flatten()
            rv[~good_rows] = -np.inf

            return rv


    def seq_component_log_density(self, x):

        enc_data = x
        ll_mat = None

        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(enc_data)
                if ll_mat is None:
                    ll_mat = np.zeros((len(temp), self.num_components))
                    ll_mat.fill(-np.inf)
                ll_mat[:, i]  = temp

        return ll_mat

    def seq_posterior(self, x):

        enc_data = x
        ll_mat = None
        
        for i in range(self.num_components):
            if not self.zw[i]:
                temp = self.components[i].seq_log_density(enc_data)
                if ll_mat is None:
                    ll_mat = np.zeros((len(temp), self.num_components))
                    ll_mat.fill(-np.inf)
                ll_mat[:, i]  = temp
                ll_mat[:, i] += self.log_w[i]

        ll_max  = ll_mat.max(axis = 1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        #if np.any(bad_rows):
        #    print('bad')

        ll_mat[bad_rows, :] = self.log_w.copy()
        ll_max[bad_rows]    = np.max(self.log_w)

        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        ll_mat /= ll_max

        return ll_mat

    def seq_encode(self, x):
        return self.components[0].seq_encode(x)

    def sampler(self, seed=None):
        return MixtureSampler(self, seed)

    def estimator(self, pseudo_count=None):
        if pseudo_count is not None:
            return MixtureEstimator([u.estimator(pseudo_count=1.0/self.num_components) for u in self.components], pseudo_count=pseudo_count, name=self.name)
        else:
            return MixtureEstimator([u.estimator() for u in self.components], name=self.name)


class MixtureSampler(object):
    def __init__(self, dist: MixtureDistribution, seed=None):
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.compSamplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size=None):

        compState = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
            return self.compSamplers[compState].sample()
        else:
            return [self.compSamplers[i].sample() for i in compState]


class MixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators, keys=(None, None)):
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]

    def update(self, x, weight, estimate):

        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior

        for i in range(self.num_components):
            self.accumulators[i].update(x, posterior[i], estimate.components[i])

    def initialize(self, x, weight, rng):

        if weight != 0:
            ww = rng.dirichlet(np.ones(self.num_components)/(self.num_components * self.num_components))
        else:
            ww = [0.0]*self.num_components

        for i in range(self.num_components):
            w = weight * ww[i]
            self.accumulators[i].initialize(x, w, rng)
            self.comp_counts[i] += w

    def seq_update(self, x, weights, estimate):

        enc_data = x
        ll_mat = None

        for i in range(estimate.num_components):
            if not estimate.zw[i]:
                temp = estimate.components[i].seq_log_density(enc_data)
                if ll_mat is None:
                    ll_mat = np.zeros((len(temp), self.num_components), dtype=np.float64)
                    ll_mat.fill(-np.inf)
                ll_mat[:, i]  = temp
                ll_mat[:, i] += estimate.log_w[i]


        ll_max = ll_mat.max(axis = 1, keepdims=True)


        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = estimate.log_w.copy()
        ll_max[bad_rows]    = np.max(estimate.log_w)


        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, keepdims=True, out=ll_max)
        np.divide(weights[:,None], ll_max, out=ll_max)
        ll_mat *= ll_max

        for i in range(self.num_components):
            w_loc = ll_mat[:, i]
            self.comp_counts[i] += w_loc.sum()
            self.accumulators[i].seq_update(enc_data, w_loc, estimate.components[i])

    def combine(self, suff_stat):
        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])
        return self

    def value(self):
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x):
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
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

class MixtureEstimatorAccumulatorFactory(object):
    def __init__(self, factories, dim, keys):
        self.factories = factories
        self.dim = dim
        self.keys = keys

    def make(self):
        return MixtureEstimatorAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class MixtureEstimator(ParameterEstimator):

    def __init__(self, estimators, fixed_weights=None, suff_stat=None, pseudo_count=None, name=None, keys=(None, None)):
        # self.estimator   = estimator
        # self.dim         = num_components
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.fixed_weights = fixed_weights

    def accumulatorFactory(self):
        est_factories = [u.accumulatorFactory() for u in self.estimators]
        return MixtureEstimatorAccumulatorFactory(est_factories, self.num_components, self.keys)

    def estimate(self, nobs, suff_stat):

        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(counts[i], comp_suff_stats[i]) for i in range(num_components)]

        if self.fixed_weights is not None:
            w = np.asarray(self.fixed_weights)

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / num_components
            w = counts + p
            w /= w.sum()

        elif self.pseudo_count is not None and self.suff_stat is not None:
            w = (counts + self.suff_stat*self.pseudo_count) / (counts.sum() + self.pseudo_count)
        else:

            nobs_loc = counts.sum()

            if nobs_loc == 0:
                w = np.ones(num_components)/float(num_components)
            else:
                w = counts / counts.sum()

        return MixtureDistribution(components, w, name=self.name)
