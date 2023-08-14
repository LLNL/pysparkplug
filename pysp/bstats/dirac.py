from typing import Optional, Any
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
	ParameterEstimator, ProbabilityDistribution
from numpy.random import RandomState
from pysp.bstats.nulldist import NullDistribution
import numpy as np

null_dist = NullDistribution()

class DiracDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, value: Any):
		self.value = value

	def __str__(self):
		return 'DiracDistribution(%s)'%(str(self.value))

	def get_prior(self):
		return self.dist.get_prior()

	def get_parameters(self):
		return self.value

	def density(self, x):
		return np.exp(self.log_density(x))

	def log_density(self, x):
		return self.dist.log_density(x)

	def seq_log_density(self, x):
		return self.dist.seq_log_density(x)

	def seq_encode(self, x):
		return np.asarray(x, dtype=object)

	def sampler(self, seed=None):
		return DiracSampler(self, seed)

	def estimator(self):
		return DiracEstimator()


class DiracSampler(object):

	def __init__(self, dist, seed=None):
		self.dist_sampler = dist.dist.sampler(seed)

	def sample(self, size=None):
		return self.dist_sampler.sample(size=size)


class DiracAccumulator(SequenceEncodableStatisticAccumulator):

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


class DiracEstimator(ParameterEstimator):

	def __init__(self, value, prior: ProbabilityDistribution = null_dist, keys=None):

		self.value  = value
		self.prior  = prior
		self.keys   = keys

	def accumulator_factory(self):
		obj = type('', (object,), {'make': lambda o: DiracAccumulator()})()
		return(obj)

	def get_prior(self):
		return self.dist.get_prior()

	def set_prior(self):
		self.dist.set_prior()

	def estimate(self, suff_stat):
		return DiracDistribution(self.value)
