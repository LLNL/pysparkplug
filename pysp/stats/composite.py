import numpy as np
from pysp.arithmetic import maxrandint
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
	ParameterEstimator


class CompositeDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, dists):
		self.dists = dists
		self.count = len(dists)

	def __str__(self):
		return 'CompositeDistribution((%s))' % (','.join(map(str, self.dists)))

	def density(self, x):

		rv = self.dists[0].density(x[0])
	
		for i in range(1, self.count):
			rv *= self.dists[i].density(x[i])

		return rv

	def log_density(self, x):
		rv = self.dists[0].log_density(x[0])

		for i in range(1, self.count):
			rv += self.dists[i].log_density(x[i])

		return rv

	def seq_log_density(self, x):
		rv = self.dists[0].seq_log_density(x[0])
		for i in range(1, self.count):
			rv += self.dists[i].seq_log_density(x[i])

		return rv

	def seq_encode(self, x):
		return tuple([self.dists[i].seq_encode([u[i] for u in x]) for i in range(self.count)])

	def sampler(self, seed=None):
		return CompositeSampler(self, seed)

	def estimator(self, pseudo_count=None):
		return CompositeEstimator([d.estimator(pseudo_count=pseudo_count) for d in self.dists])


class CompositeSampler(object):

	def __init__(self, dist, seed=None):
		self.dist         = dist
		self.rng          = RandomState(seed)
		self.distSamplers = [d.sampler(seed=self.rng.randint(maxrandint)) for d in dist.dists]

	def sample(self, size=None):

		if size is None:
			return tuple([d.sample(size=size) for d in self.distSamplers])
		else:
			return list(zip(*[d.sample(size=size) for d in self.distSamplers]))


class CompositeAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, accumulators, keys=None):
		self.accumulators = accumulators
		self.count        = len(accumulators)
		self.key          = keys

	def update(self, x, weight, estimate):
		if estimate is not None:
			for i in range(0, self.count):
				self.accumulators[i].update(x[i], weight, estimate.dists[i])
		else:
			for i in range(0, self.count):
				self.accumulators[i].update(x[i], weight, None)

	def initialize(self, x, weight, rng):
		for i in range(0, self.count):
			self.accumulators[i].initialize(x[i], weight, rng)

	def get_seq_lambda(self):
		rv = []
		for i in range(self.count):
			rv.extend(self.accumulators[i].get_seq_lambda())
		return rv

	def seq_update(self, x, weights, estimate):
		for i in range(self.count):
			self.accumulators[i].seq_update(x[i], weights, estimate.dists[i] if estimate is not None else None)

	def combine(self, suff_stat):
		for i in range(0, self.count):
			self.accumulators[i].combine(suff_stat[i])
		return self

	def value(self):
		return tuple([x.value() for x in self.accumulators])

	def from_value(self, x):
		self.accumulators = [self.accumulators[i].from_value(x[i]) for i in range(len(x))]
		self.count = len(x)
		return self

	def key_merge(self, stats_dict):

		if self.key is not None:
			if self.key in stats_dict:
				stats_dict[self.key].combine(self.value())
			else:
				stats_dict[self.key] = self

		for u in self.accumulators:
			u.key_merge(stats_dict)

	def key_replace(self, stats_dict):

		if self.key is not None:
			if self.key in stats_dict:
				self.from_value(stats_dict[self.key].value())

		for u in self.accumulators:
			u.key_replace(stats_dict)

class CompositeAccumulatorFactory(object):

	def __init__(self, factories, keys):
		self.factories = factories
		self.keys = keys

	def make(self):
		return CompositeAccumulator([u.make() for u in self.factories], self.keys)

class CompositeEstimator(ParameterEstimator):

	def __init__(self, estimators, keys=None):

		self.estimators  = estimators
		self.count       = len(estimators)
		self.keys        = keys
		
	def accumulatorFactory(self):
		return CompositeAccumulatorFactory([u.accumulatorFactory() for u in self.estimators], self.keys)

	def estimate(self, nobs, suff_stat):
		return CompositeDistribution(tuple([est.estimate(nobs, ss) for est, ss in zip(self.estimators, suff_stat)]))
