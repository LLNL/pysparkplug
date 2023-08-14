from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
	ParameterEstimator, ProbabilityDistribution
from numpy.random import RandomState
import numpy as np


class IgnoredDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, dist):
		self.dist = dist
		self.has_pdf = isinstance(dist, ProbabilityDistribution)

	def __str__(self):
		return 'IgnoredDistribution(%s)'%(str(self.dist))

	def density(self, x):
		return np.exp(self.log_density(x))

	def log_density(self, x):
		if self.dist is not None:
			return self.dist.log_density(x)
		else:
			return 0.0

	def seq_log_density(self, x):
		if self.dist is None:
			return np.zeros(len(x))
		else:
			return self.dist.seq_log_density(x)

	def seq_encode(self, x):
		if isinstance(self.dist, SequenceEncodableProbabilityDistribution):
			return self.dist.seq_encode(x)
		else:
			return x

	def sampler(self, seed=None):
		return IgnoredSampler(self, seed)

	def estimator(self, pseudo_count=None):
		return IgnoredEstimator(dist=self.dist)


class IgnoredSampler(object):

	def __init__(self, dist, seed=None):
		if dist.dist is not None:
			self.dist_sampler = dist.dist.sampler(seed)
		else:
			self.dist_sampler = None

	def sample(self, size=None):
		if self.dist_sampler is None:
			if size is None:
				return None
			else:
				return [None]*size
		else:
			return self.dist_sampler.sample(size=size)


class IgnoredAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self):
		pass

	def update(self, x, weight, estimate):
		pass

	def seq_update(self, x, weights, estimate):
		pass

	def initialize(self, x, weight, rng):
		pass

	def combine(self, suff_stat):
		return self

	def value(self):
		return None

	def from_value(self, x):
		return self

	def key_merge(self, stats_dict):
		pass

	def key_replace(self, stats_dict):
		pass


class IgnoredEstimator(ParameterEstimator):

	def __init__(self, dist=None, pseudo_count=None, suff_stat=None, keys=None):

		self.dist         = dist
		self.pseudo_count = pseudo_count
		self.suff_stat    = suff_stat
		self.keys         = keys

	def accumulatorFactory(self):
		obj = type('', (object,), {'make': lambda o: IgnoredAccumulator()})()
		return(obj)

	def estimate(self, nobs, suff_stat):
		return IgnoredDistribution(self.dist)
