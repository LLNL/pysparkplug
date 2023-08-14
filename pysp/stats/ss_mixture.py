from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import numpy as np
import pysp.utils.vector as vec
from pysp.arithmetic import maxrandint


class SemiSupervisedMixtureDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, components, w):
		self.components = components
		self.num_components = len(components)
		self.w = np.asarray(w)
		self.zw = (self.w == 0.0)
		self.log_w = np.log(w+self.zw)
		self.log_w[self.zw] = -np.inf

	def __str__(self):
		return 'SemiSupervisedMixtureDistribution([%s], [%s])' % (','.join([str(u) for u in self.components]), ','.join(map(str, self.w)))

	def density(self, x):
		return exp(self.log_density(x))

	def log_density(self, x):

		datum, prior = x
		if prior is None:
			return vec.log_sum(np.asarray([u.log_density(datum) for u in self.components]) + self.log_w)
		else:
			w_loc = np.zeros(self.num_components)
			h_loc = np.zeros(self.num_components, dtype=bool)
			i_loc = np.zeros(self.num_components, dtype=int)

			for idx,val in prior:
				w_loc[idx] += np.log(val)
				h_loc[idx]  = True
				i_loc[idx]  = idx

			w_loc[h_loc] += self.log_w[h_loc]
			w_loc = vec.log_posterior(w_loc[h_loc])

			return vec.log_sum(np.asarray([self.components[i].log_density(datum) for i in np.flatnonzero(h_loc) ]) + w_loc)


	def posterior(self, x):

		datum, prior = x

		if prior is None:
			rv = vec.posterior(np.asarray([u.log_density(datum) for u in self.components]) + self.log_w)
		else:

			w_loc = np.zeros(self.num_components)
			h_loc = np.zeros(self.num_components, dtype=bool)

			for idx,val in prior:
				w_loc[idx] += np.log(val)
				h_loc[idx]  = True

			w_loc[h_loc] += self.log_w[h_loc]
			for i in np.flatnonzero(h_loc):
				w_loc[i] += self.components[i].log_density(datum)

			w_loc[h_loc] = vec.posterior(w_loc[h_loc])
			rv = w_loc

		return rv

	def seq_log_density(self, x):

		sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag) = x
		ll_mat = np.zeros((sz, self.num_components))
		ll_mat.fill(-np.inf)

		norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * self.w[enc_prior[1]]), minlength=sz)
		norm_const = np.log(norm_const[enc_prior_flag])

		ll_mat[~enc_prior_flag, :] = self.log_w
		ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + self.log_w[enc_prior[1]]

		for i in range(self.num_components):
			if not self.zw[i]:
				ll_mat[:, i] += self.components[i].seq_log_density(enc_data)
				ll_mat[enc_prior_flag, i] -= norm_const

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


	def seq_posterior(self, x):

		sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag) = x
		ll_mat = np.zeros((sz, self.num_components))
		ll_mat.fill(-np.inf)

		norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * self.w[enc_prior[1]]), minlength=sz)
		norm_const = np.log(norm_const[enc_prior_flag])

		ll_mat[~enc_prior_flag, :] = self.log_w
		ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + self.log_w[enc_prior[1]]

		for i in range(self.num_components):
			if not self.zw[i]:
				ll_mat[:, i]  += self.components[i].seq_log_density(enc_data)
				ll_mat[enc_prior_flag, i] -= norm_const

		ll_max  = ll_mat.max(axis = 1, keepdims=True)

		bad_rows = np.isinf(ll_max.flatten())

		#if np.any(bad_rows):
		#	print('bad')

		ll_mat[bad_rows, :] = self.log_w
		ll_max[bad_rows]    = np.max(self.log_w)

		ll_mat -= ll_max

		np.exp(ll_mat, out=ll_mat)
		ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
		ll_mat /= ll_sum

		return ll_mat

	def seq_encode(self, x):

		prior_comp = []
		prior_idx  = []
		prior_val  = []
		data       = []

		for i, xi in enumerate(x):
			datum, prior = xi
			data.append(datum)
			if prior is not None:
				for prior_entry in prior:
					prior_idx.append(i)
					prior_comp.append(prior_entry[0])
					prior_val.append(prior_entry[1])

		prior_comp = np.asarray(prior_comp, dtype=int)
		prior_idx  = np.asarray(prior_idx, dtype=int)
		prior_val  = np.asarray(prior_val, dtype=float)

		#prior_mat = scipy.sparse.csc_matrix((prior_val, (prior_idx, prior_comp)), dtype=float)
		#prior_mat.eliminate_zeros()
		#prior_mat = np.zeros((len(x), prior_comp.max()))
		#prior_mat[prior_idx, prior_comp] = prior_val
		prior_mat = (prior_idx, prior_comp, prior_val, np.log(prior_val))

		prior_sum = np.bincount(prior_idx, weights=prior_val, minlength=len(x))
		has_prior = prior_sum != 0

		return len(x), self.components[0].seq_encode(data), (prior_mat, prior_sum, has_prior)


	def sampler(self, seed=None):
		return SemiSupervisedMixtureSampler(self, seed)

	def estimator(self, pseudo_count=None):
		if pseudo_count is not None:
			return SemiSupervisedMixtureEstimator([u.estimator(pseudo_count=1.0/self.num_components) for u in self.components], pseudo_count=pseudo_count)
		else:
			return SemiSupervisedMixtureEstimator([u.estimator() for u in self.components])


class SemiSupervisedMixtureSampler(object):
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


class SemiSupervisedMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, accumulators, keys=(None, None)):
		self.accumulators = accumulators
		self.num_components = len(accumulators)
		self.comp_counts = np.zeros(self.num_components, dtype=float)
		self.weight_key = keys[0]
		self.comp_key = keys[1]

	def update(self, x, weight, estimate):

		likelihood = estimate.posterior(x)
		datum,prior = x

		# likelihood = np.asarray([estimate.components[i].log_density(x) for i in range(self.num_components)])
		# likelihood += estimate.log_w
		# max_likelihood = likelihood.max()
		# likelihood -= max_likelihood
		#
		# np.exp(likelihood, out=likelihood)
		# pp = likelihood.sum()
		# likelihood /= pp

		self.comp_counts += likelihood * weight

		for i in range(self.num_components):
			self.accumulators[i].update(datum, likelihood[i] * weight, estimate.components[i])

	def initialize(self, x, weight, rng):

		#if self.comp_counts.sum() == 0:
		#	p = np.ones(self.num_components)/float(self.num_components)
		#else:
		#	p = self.num_components - self.comp_counts
		#	p /= p.sum()
		#idx  = rng.choice(self.num_components, p=p)

		datum, prior = x

		if prior is None:

			idx  = rng.choice(self.num_components)
			wc0  = 0.001
			wc1  = wc0/max((float(self.num_components)-1.0),1.0)
			wc2  = 1.0 - wc0

			for i in range(self.num_components):
				w = weight*wc2 if i == idx else wc1
				self.accumulators[i].initialize(datum, w, rng)
				self.comp_counts[i] += w

		else:

			for i,w in prior:
				ww = weight * w
				self.accumulators[i].initialize(datum, ww, rng)
				self.comp_counts[i] += ww

	def seq_update(self, x, weights, estimate):

		sz, enc_data, (enc_prior, enc_prior_sum, enc_prior_flag) = x
		ll_mat = np.zeros((sz, estimate.num_components))
		ll_mat.fill(-np.inf)

		norm_const = np.bincount(enc_prior[0], weights=(enc_prior[2] * estimate.w[enc_prior[1]]), minlength=sz)
		norm_const = np.log(norm_const[enc_prior_flag])

		ll_mat[~enc_prior_flag, :] = estimate.log_w
		ll_mat[enc_prior[0], enc_prior[1]] = enc_prior[3] + estimate.log_w[enc_prior[1]]

		for i in range(self.num_components):
			ll_mat[:, i]  += estimate.components[i].seq_log_density(enc_data)
			ll_mat[enc_prior_flag, i] -= norm_const


		ll_max = ll_mat.max(axis = 1, keepdims=True)

		bad_rows = np.isinf(ll_max.flatten())

		ll_mat[bad_rows, :] = estimate.log_w
		ll_max[bad_rows]    = np.max(estimate.log_w)

		ll_mat -= ll_max
		np.exp(ll_mat, out=ll_mat)
		ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
		ll_mat /= ll_sum


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

class SemiSupervisedMixtureEstimatorAccumulatorFactory(object):
	def __init__(self, factories, dim, keys):
		self.factories = factories
		self.dim = dim
		self.keys = keys

	def make(self):
		return SemiSupervisedMixtureEstimatorAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class SemiSupervisedMixtureEstimator(object):
	def __init__(self, estimators, suff_stat=None, pseudo_count=None, keys=(None, None)):
		# self.estimator   = estimator
		# self.dim         = num_components
		self.num_components = len(estimators)
		self.estimators = estimators
		self.pseudo_count = pseudo_count
		self.suff_stat = suff_stat
		self.keys = keys

	def accumulatorFactory(self):
		est_factories = [u.accumulatorFactory() for u in self.estimators]
		return SemiSupervisedMixtureEstimatorAccumulatorFactory(est_factories, self.num_components, self.keys)

	def estimate(self, nobs, suff_stat):

		num_components = self.num_components
		counts, comp_suff_stats = suff_stat

		components = [self.estimators[i].estimate(counts[i], comp_suff_stats[i]) for i in range(num_components)]

		if self.pseudo_count is not None and self.suff_stat is None:
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

		return SemiSupervisedMixtureDistribution(components, w)
