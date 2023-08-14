from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import numpy as np


class GeometricDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, p, name=None):
		self.p      = p
		self.log_p  = np.log(p)
		self.log_1p = np.log1p(-p)
		self.name   = name

	def __str__(self):
		return 'GeometricDistribution(%s, name=%s)'%(repr(self.p), repr(self.name))

	def density(self, x):
		return exp(self.log_density(x))

	def log_density(self, x):
		return (x-1)*self.log_1p + self.log_p

	def seq_log_density(self, x):
		rv = x-1
		rv *= self.log_1p
		rv += self.log_p
		return rv

	def seq_encode(self, x):
		rv = np.asarray(x, dtype=float)
		return rv

	def sampler(self, seed=None):
		return GeometricSampler(self, seed)

	def estimator(self, pseudo_count=None):
		if pseudo_count is None:
			return GeometricEstimator(name=self.name)
		else:
			return GeometricEstimator(pseudo_count=pseudo_count, suff_stat=self.p, name=self.name)


class GeometricSampler(object):

	def __init__(self, dist, seed=None):
		self.rng  = RandomState(seed)
		self.dist = dist

	def sample(self, size=None):
		return self.rng.geometric(p=self.dist.p, size=size)


class GeometricAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, keys):
		self.sum   = 0.0
		self.count = 0.0
		self.key   = keys

	def update(self, x, weight, estimate):
		if x >= 0:
			self.sum  += x*weight
			self.count += weight

	def seq_update(self, x, weights, estimate):
		self.sum += np.dot(x, weights)
		self.count += np.sum(weights)

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)

	def combine(self, suff_stat):
		self.sum   += suff_stat[1]
		self.count += suff_stat[0]
		return self

	def value(self):
		return self.count, self.sum

	def from_value(self, x):
		self.count = x[0]
		self.sum = x[1]
		return self

	def key_merge(self, stats_dict):
		if self.key is not None:
			if self.key in stats_dict:
				vals = stats_dict[self.key]
				stats_dict[self.key] = (vals[0] + self.count, vals[1] + self.sum)
			else:
				stats_dict[self.key] = (self.count, self.sum)

	def key_replace(self, stats_dict):
		if self.key is not None:
			if self.key in stats_dict:
				vals = stats_dict[self.key]
				self.count = vals[0]
				self.sum   = vals[1]


class GeometricEstimator(ParameterEstimator):

	def __init__(self, pseudo_count=None, suff_stat=None, name=None, keys=None):

		self.pseudo_count  = pseudo_count
		self.suff_stat     = suff_stat
		self.keys          = keys
		self.name          = name

	def accumulatorFactory(self):
		obj = type('', (object,), {'make': lambda o: GeometricAccumulator(self.keys)})()
		return(obj)

	def estimate(self, nobs, suff_stat):

		if self.pseudo_count is not None and self.suff_stat is not None:
			p = (suff_stat[0] + self.pseudo_count*self.suff_stat[0])/(suff_stat[1] + self.pseudo_count*self.suff_stat[1])
		elif self.pseudo_count is not None and self.suff_stat is None:
			p = (suff_stat[0] + self.pseudo_count)/(suff_stat[1] + self.pseudo_count)
		else:
			p = suff_stat[0]/suff_stat[1]

		return GeometricDistribution(p, name=self.name)
