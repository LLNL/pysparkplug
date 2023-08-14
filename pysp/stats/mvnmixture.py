from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import numpy as np
import pysp.utils.vector as vec
from scipy.special import gammaln, digamma
import scipy.linalg
from pysp.arithmetic import maxrandint


class GaussianMixtureDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, mu, sig2, w, name=None):

        num_comp = len(w)
        dim = len(mu[0])

        self.w     = np.asarray(w)
        self.mu    = np.asarray(mu)
        self.sig2  = np.asarray(sig2)
        self.name  = name

        self.s_mu   = np.reshape(mu, (num_comp, dim))
        self.s_sig2 = np.reshape(sig2, (num_comp, dim))
        self.log_w  = np.log(w)
        self.log_c  = -0.5*np.log(2*np.pi)*dim - 0.5*np.log(self.sig2).sum(axis=1) + self.log_w
        self.dim    = dim
        self.num_components = num_comp

        self.ca    = -0.5/self.sig2
        self.cb    = self.mu/self.sig2
        self.cc    = -0.5*(self.mu*self.mu/self.sig2).sum(axis=1) + self.log_c

    def __str__(self):
        s_w = ','.join(map(str, self.w))
        s_m = ','.join([str(u) for u in self.mu.flatten()])
        s_s = ','.join([str(u) for u in self.sig2.flatten()])
        return 'GaussianMixtureDistribution([%s], [%s], [%s], name=%s)' % (s_m, s_s, s_w, str(self.name))

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):
        rv = np.subtract(x, self.s_mu)
        rv *= rv
        rv /= self.s_sig2
        rv *=-0.5
        rv += self.log_c
        ll_max = rv.max()
        rv -= ll_max
        np.exp(rv, out=rv)
        return np.log(rv.sum()) + ll_max

    def seq_posterior(self, x):
        sz, enc_data = x
        ll_mat  = -0.5*(np.power(enc_data-self.s_mu,2)/self.s_sig2).sum(axis=2)
        ll_mat += self.log_c
        ccc     = np.max(ll_mat, axis=1, keepdims=True)
        ll_mat -= ccc
        np.exp(ll_mat, out=ll_mat)
        np.sum(ll_mat, axis=1, out=ccc)
        ll_mat /= ccc
        return ll_mat

    def seq_log_density(self, x):
        sz, (xx, xxx) = x
        ll_mat = np.dot(xx * xx, self.ca.T) + np.dot(xx, self.cb.T)
        ll_mat += self.cc
        ll_max  = np.max(ll_mat, axis=1, keepdims=True)
        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        rv = np.sum(ll_mat, axis=1)
        np.log(rv, out=rv)
        rv += ll_max.flatten()
        return rv

    def seq_encode(self, x):
        xv = np.asarray(x)
        return len(x), (xv, xv*xv)

    def sampler(self, seed=None):
        return GaussianMixtureSampler(self, seed)

    def estimator(self, pseudo_count=None):
        pass


class GaussianMixtureSampler(object):
    def __init__(self, dist, seed=None):

        rng_loc = RandomState(seed)

        self.rng = RandomState(rng_loc.randint(0, maxrandint))
        self.dist = dist
        self.compSamplers = [d.sampler(seed=rng_loc.randint(0, maxrandint)) for d in self.dist.components]

    def sample(self, size=None):

        compState = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
                return self.compSamplers[compState].sample()
        else:
                return [self.compSamplers[i].sample() for i in compState]


class GaussianMixtureAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators, keys=(None, None)):
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]

    def update(self, x, weight, estimate):

        likelihood = np.asarray([estimate.components[i].log_density(x) for i in range(self.num_components)])
        likelihood += estimate.log_w
        max_likelihood = likelihood.max()
        likelihood -= max_likelihood

        np.exp(likelihood, out=likelihood)
        pp = likelihood.sum()
        likelihood /= pp

        self.comp_counts += likelihood * weight

        for i in range(self.num_components):
            self.accumulators[i].update(x, likelihood[i] * weight, estimate.components[i])

    def initialize(self, x, weight, rng):

        #if self.comp_counts.sum() == 0:
        #	p = np.ones(self.num_components)/float(self.num_components)
        #else:
        #	p = self.num_components - self.comp_counts
        #	p /= p.sum()
        #idx  = rng.choice(self.num_components, p=p)

        idx  = rng.choice(self.num_components)
        wc0  = 0.001
        wc1  = wc0/max((float(self.num_components)-1.0),1.0)
        wc2  = 1.0 - wc0

        for i in range(self.num_components):
            w = weight*wc2 if i == idx else wc1
            self.accumulators[i].initialize(x, w, rng)
            self.comp_counts[i] += w

    def seq_update(self, x, weights, estimate):

        sz, enc_data = x
        ll_mat = np.zeros((sz, self.num_components))
        ll_mat.fill(-np.inf)

        for i in range(estimate.num_components):
            if not estimate.zw[i]:
                ll_mat[:, i] = estimate.components[i].seq_log_density(enc_data)
                ll_mat[:, i] += estimate.log_w[i]


        ll_max = ll_mat.max(axis = 1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        #if np.any(bad_rows):
        #	print('bad')

        ll_mat[bad_rows, :] = estimate.log_w
        ll_max[bad_rows]    = np.max(estimate.log_w)

        #ll_mat[bad_rows, :] = -np.log(self.num_components)
        #ll_max[bad_rows]    = -np.log(self.num_components)

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
        ll_mat /= ll_sum

        ttt = 1

        for i in range(self.num_components):
            w_loc = ll_mat[:, i]*weights
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

class GaussianMixtureEstimatorAccumulatorFactory(object):
    def __init__(self, factories, dim, keys):
        self.factories = factories
        self.dim = dim
        self.keys = keys

    def make(self):
        return GaussianMixtureAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class GaussianMixtureEstimator(object):
    def __init__(self, estimators, name=None, conj_prior_params=None, suff_stat=None, pseudo_count=None, keys=(None, None)):
        # self.estimator   = estimator
        # self.dim         = num_components
        self.num_components = len(estimators)
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.conj_prior_params = conj_prior_params
        self.name = name

    def accumulatorFactory(self):
        est_factories = [u.accumulatorFactory() for u in self.estimators]
        return GaussianMixtureEstimatorAccumulatorFactory(est_factories, self.num_components, self.keys)

    def estimate(self, nobs, suff_stat):
        pass

