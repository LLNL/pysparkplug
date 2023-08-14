from typing import Optional, Any
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator, DistributionSampler
import numpy as np


class OptionalDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, dist, p: Optional[float] = None, missing_value: Any = None, name: Optional[str] = None):
		self.dist  = dist
		self.p     = p if p is not None else 0.0
		self.hasP  = p is not None
		self.logP  = -np.inf if self.p == 0 else np.log(self.p)
		self.logPN = -np.inf if self.p == 1 else np.log1p(-self.p)

		self.missing_value_is_nan = isinstance(missing_value, (np.floating, float)) and np.isnan(missing_value)
		self.log1P = np.log1p(self.p)
		self.missing_value = missing_value
		self.name = name

	def __str__(self):
		s1 = str(self.dist)
		s2 = repr(None if not self.hasP else self.p)
		if self.missing_value_is_nan:
			s3 = 'float("nan")'
		else:
			s3 = repr(self.missing_value)
		s4 = repr(self.name)
		return 'OptionalDistribution(%s, p=%s, missing_value=%s, name=%s)'%(s1, s2, s3, s4)

	def density(self, x):
		return np.exp(self.log_density(x))

	def log_density(self, x):

		if self.missing_value_is_nan:
			if isinstance(x, (np.floating, float)) and np.isnan(x):
				not_missing = False
			else:
				not_missing = True
		else:
			if x == self.missing_value:
				not_missing = False
			else:
				not_missing = True

		if self.hasP:
			if not_missing:
				return self.dist.log_density(x) + self.logPN
			else:
				return self.logP
		# This is a degenerate use case that should probably be deprecated
		else:
			if not_missing:
				return self.dist.log_density(x)
			else:
				return 0.0

	def seq_encode(self, x):

		nz_idx = []
		nz_val = []
		z_idx  = []

		if self.missing_value_is_nan:
			for i, v in enumerate(x):
				if isinstance(v, (np.floating, float)) and np.isnan(v):
					z_idx.append(i)
				else:
					nz_idx.append(i)
					nz_val.append(v)
		else:
			for i, v in enumerate(x):
				if v == self.missing_value:
					z_idx.append(i)
				else:
					nz_idx.append(i)
					nz_val.append(v)

		enc_data = self.dist.seq_encode(nz_val)

		nz_idx = np.asarray(nz_idx, dtype=int)
		z_idx = np.asarray(z_idx, dtype=int)

		return len(x), z_idx, nz_idx, enc_data


	def seq_log_density(self, x):

		sz, z_idx, nz_idx, enc_data = x

		rv = np.zeros(sz)

		if self.hasP:
			rv[z_idx]  = self.logP
			rv[nz_idx] = self.dist.seq_log_density(enc_data) + self.logPN
		else:
			rv[nz_idx] = self.dist.seq_log_density(enc_data)

		return rv

	def sampler(self, seed=None):
		return OptionalSampler(self, seed)

	def estimator(self, pseudo_count=None):
		return OptionalEstimator(self.dist.estimator(pseudo_count=pseudo_count), missing_value=self.missing_value, pseudo_count=pseudo_count, est_prob=self.hasP, name=self.name)


class OptionalSampler(DistributionSampler):

	def __init__(self, dist, seed=None):
		super().__init__(dist, seed)
		self.sampler = self.dist.dist.sampler(self.new_seed())

	def sample(self, size=None):

		sampler = self.sampler

		if not self.dist.hasP:
			return self.sampler.sample(size=size)

		if size is None:
			if self.rng.choice([0, 1], replace=True, p=[self.dist.p, 1.0 - self.dist.p]) == 0:
				return self.dist.missing_value
			else:
				return sampler.sample(size=size)
		else:
			states  = self.rng.choice([0, 1], size=size, replace=True, p=[self.dist.p, 1.0 - self.dist.p])

			nz_count = int(np.sum(states))

			if nz_count == size:
				return sampler.sample(size=size)
			elif nz_count == 0:
				return [self.dist.missing_value for i in range(size)]
			else:
				nz_vals = sampler.sample(size=nz_count)
				nz_idx  = np.flatnonzero(states)
				rv      = [self.dist.missing_value for i in range(size)]

				for cnt, i in enumerate(nz_idx):
					rv[i] = nz_vals[cnt]

				return rv


class OptionalEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, accumulator, missing_value):
		self.accumulator = accumulator
		self.weights = [0.0, 0.0]
		self.missing_value = missing_value
		self.missing_value_is_nan = isinstance(missing_value, (np.floating, float)) and np.isnan(missing_value)

	def update(self, x, weight, estimate):

		if self.missing_value_is_nan:
			if isinstance(x, (np.floating, float)) and np.isnan(x):
				self.weights[0] += weight
			else:
				self.accumulator.update(x, weight, estimate)
				self.weights[1] += weight
		else:
			if (x == self.missing_value) or (x is self.missing_value):
				self.weights[0] += weight
			else:
				self.accumulator.update(x, weight, estimate)
				self.weights[1] += weight

	def initialize(self, x, weight, rng):

		if self.missing_value_is_nan:
			if isinstance(x, (np.floating, float)) and np.isnan(x):
				self.weights[0] += weight
			else:
				self.accumulator.initialize(x, weight, rng)
				self.weights[1] += weight
		else:
			if (x == self.missing_value) or (x is self.missing_value):
				self.weights[0] += weight
			else:
				self.accumulator.initialize(x, weight, rng)
				self.weights[1] += weight


	def seq_update(self, x, weights, estimate):

		sz, z_idx, nz_idx, enc_data = x
		nz_weights = weights[nz_idx]
		z_weights = weights[z_idx]

		self.weights[0] += np.sum(z_weights)
		self.weights[1] += np.sum(nz_weights)
		self.accumulator.seq_update(enc_data, nz_weights, estimate.dist if estimate is not None else None)

	def combine(self, suff_stat):

		self.weights[0] += suff_stat[0][0]
		self.weights[1] += suff_stat[0][1]
		self.accumulator.combine(suff_stat[1])

		return self

	def value(self):
		return self.weights, self.accumulator.value()

	def from_value(self, x):
		self.weights = x[0]
		self.accumulator.from_value(x[1])

		return self


class OptionalEstimatorAccumulatorFactory(object):

	def __init__(self, estimator, missing_value):
		self.estimator = estimator
		self.missing_value = missing_value

	def make(self):
		return OptionalEstimatorAccumulator(self.estimator.accumulatorFactory().make(), self.missing_value)


class OptionalEstimator(ParameterEstimator):

	def __init__(self, estimator, missing_value=None, est_prob=False, pseudo_count=None, name=None, keys=(None, None)):
		self.estimator = estimator
		self.est_prob      = est_prob
		self.pseudo_count = pseudo_count
		self.missing_value = missing_value
		self.keys = keys
		self.name = name

	def accumulatorFactory(self):
		return OptionalEstimatorAccumulatorFactory(self.estimator, self.missing_value)

	def estimate(self, nobs, suff_stat):

		dist = self.estimator.estimate(suff_stat[0][1], suff_stat[1])

		if self.pseudo_count is not None and self.est_prob:
			return OptionalDistribution(dist, (suff_stat[0][0] + self.pseudo_count) / ((2*self.pseudo_count) + suff_stat[0][0] + suff_stat[0][1]), missing_value=self.missing_value, name=self.name)

		elif self.est_prob:

			nobs_loc = suff_stat[0][0]+suff_stat[0][1]
			z_nobs = suff_stat[0][0]

			if nobs_loc == 0:
				return OptionalDistribution(dist, None, missing_value=self.missing_value, name=self.name)
			else:
				return OptionalDistribution(dist, p=z_nobs/nobs_loc, missing_value=self.missing_value, name=self.name)
		else:
			return OptionalDistribution(dist, p=None, missing_value=self.missing_value, name=self.name)

