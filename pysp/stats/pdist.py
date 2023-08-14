import math
import numpy as np
from pysp.arithmetic import maxrandint

class ProbabilityDistribution(object):

	def __init__(self):
		pass

	def __repr__(self):
		return self.__str__()

	def density(self, x) -> float:
		return math.exp(self.log_density(x))

	def log_density(self, x) -> float:
		pass

	def sampler(self, seed=None):
		pass

	def estimator(self, pseudo_count=None):
		pass


class DistributionSampler(object):

	def __init__(self, dist, seed=None):
		self.dist = dist
		self.rng  = np.random.RandomState(seed)

	def new_seed(self):
		return self.rng.randint(0, maxrandint)

	def sample(self, size=None):
		pass


class ConditionalSampler(object):

	def sample_given(self, x):
		pass


class StatisticAccumulator(object):

	def update(self, x, weight, estimate):
		pass

	def initialize(self, x, weight, rng):
		self.update(x, weight, estimate=None)

	def combine(self, suff_stat):
		pass

	def value(self):
		pass

	def from_value(self, x):
		pass

	def key_merge(self, stats_dict):
		pass

	def key_replace(self, stats_dict):
		pass

class StatisticAccumulatorFactory(object):

	def make(self):
		pass

class ParameterEstimator(object):

	def estimate(self, nobs, suff_stat):
		pass

	def accumulatorFactory(self):
		pass


class SequenceEncodableProbabilityDistribution(ProbabilityDistribution):

	def seq_log_density(self, x):
		return np.asarray([self.log_density(u) for u in x])

	def seq_log_density_lambda(self):
		return [self.seq_log_density]

	def seq_encode(self, x):
		return x


class SequenceEncodableStatisticAccumulator(StatisticAccumulator):

	def get_seq_lambda(self):
		pass

	def seq_update(self, x, weights, estimate):
		pass
