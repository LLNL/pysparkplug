from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator, DistributionSampler, ConditionalSampler
from pysp.arithmetic import maxrandint
from numpy.random import RandomState
import numpy as np
import math

class ConditionalDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, dmap, default_dist=None, given_dist=None, name=None, keys=None):

		if isinstance(dmap, list):
			dmap = dict(zip(range(len(dmap)),dmap))

		self.dmap         = dmap
		self.default_dist = default_dist
		self.has_default  = default_dist is not None
		self.given_dist   = given_dist
		self.name         = name
		self.keys         = keys

	def __str__(self):
		s1 = repr(self.dmap)
		s2 = repr(self.default_dist)
		s3 = repr(self.given_dist)
		s4 = repr(self.name)
		s5 = repr(self.keys)
		return 'ConditionalDistribution(%s, default_dist=%s, given_dist=%s, name=%s, keys=%s)' % (s1, s2, s3, s4, s5)

	def density(self, x):
		return math.log(self.log_density(x))

	def log_density(self, x):

		if self.has_default:
			rv = self.dmap.get(x[0], self.default_dist).log_density(x[1])
		else:
			rv = self.dmap[x[0]].log_density(x[1])

		if self.given_dist is not None:
			rv += self.given_dist.log_density(x[0])

		return rv

	def seq_log_density(self, x):

		sz, cond_vals, idx_vals, eobs_vals, given_vals = x

		rv = np.zeros(sz)

		if self.has_default:
			for i in range(len(cond_vals)):
				rv[idx_vals[i]] = self.dmap.get(cond_vals[i], self.default_dist).seq_log_density(eobs_vals[i])
		else:
			for i in range(len(cond_vals)):
				if cond_vals[i] in self.dmap:
					rv[idx_vals[i]] = self.dmap[cond_vals[i]].seq_log_density(eobs_vals[i])
				else:
					rv[idx_vals[i]] = -np.inf

		if self.given_dist is not None:
			rv += self.given_dist.seq_log_density(given_vals)

		return rv


	def seq_encode(self, x):

		cond_enc = dict()

		given_vals = []

		for i in range(len(x)):
			xx = x[i]
			given_vals.append(xx[0])
			if xx[0] not in cond_enc:
				cond_enc[xx[0]] = [[xx[1]], [i]]
			else:
				cond_enc_loc = cond_enc[xx[0]]
				cond_enc_loc[0].append(xx[1])
				cond_enc_loc[1].append(i)

		cond_enc = list(cond_enc.items())

		cond_vals  = tuple([u[0] for u in cond_enc])
		eobs_vals  = tuple([self.dmap.get(u[0], self.default_dist).seq_encode(u[1][0]) for u in cond_enc])
		idx_vals   = tuple([np.asarray(u[1][1]) for u in cond_enc])

		if self.given_dist is not None:
			given_vals = self.given_dist.seq_encode(given_vals)
		else:
			given_vals = None

		return len(x), cond_vals, idx_vals, eobs_vals, given_vals


	def sampler(self, seed=None):
		return ConditionalDistributionSampler(self, seed=seed)

	def estimator(self, pseudo_count=None):
		pass


class ConditionalDistributionSampler(ConditionalSampler, DistributionSampler):

	def __init__(self, dist: ConditionalDistribution, seed=None):
		self.dist = dist
		rng = np.random.RandomState(seed)

		loc_seed = rng.randint(0, maxrandint)
		if dist.has_default:
			self.default_sampler = dist.default_dist.sampler(loc_seed)
		else:
			self.default_sampler = None

		loc_seed = rng.randint(0, maxrandint)
		if dist.given_dist is not None:
			self.given_sampler = dist.given_dist.sampler(loc_seed)
		else:
			self.given_sampler = None

		self.samplers = {k: u.sampler(rng.randint(0, maxrandint)) for k,u in self.dist.dmap.items()}

	def sample(self, size=None):

		if size is None:
			x0 = self.given_sampler.sample()
			if x0 in self.samplers:
				x1 = self.samplers[x0].sample()
			else:
				x1 = self.default_sampler.sample()
			return x0, x1

		else:
			return [self.sample() for i in range(size)]

	def sample_given(self, x):

		if x in self.samplers:
			return self.samplers[x].sample()
		elif self.default_sampler is not None:
			return self.default_sampler.sample()
		else:
			raise Exception('Conditional default distribution unspecified.')


class ConditionalDistributionAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, accumulator_map, default_accumulator, given_accumulator, keys):
		self.accumulator_map     = accumulator_map
		self.default_accumulator = default_accumulator
		self.given_accumulator   = given_accumulator
		self.key                 = keys

	def update(self, x, weight, estimate):

		if x[0] in self.accumulator_map:
			if estimate is None:
				self.accumulator_map[x[0]].update(x[1], weight, None)
			else:
				self.accumulator_map[x[0]].update(x[1], weight, estimate.dmap[x[0]])
		else:
			if self.default_accumulator is not None:
				if estimate is None:
					self.default_accumulator.update(x[1], weight, None)
				else:
					self.default_accumulator.update(x[1], weight, estimate.default_dist)

		if self.given_accumulator is not None:
			if estimate is None:
				self.given_accumulator.update(x[0], weight, None)
			else:
				self.given_accumulator.update(x[0], weight, estimate.given_dist)

	def initialize(self, x, weight, rng):

		if x[0] in self.accumulator_map:
			self.accumulator_map[x[0]].initialize(x[1], weight, rng)
		else:
			if self.default_accumulator is not None:
				self.default_accumulator.initialize(x[1], weight, rng)

		if self.given_accumulator is not None:
			self.given_accumulator.initialize(x[0], weight, rng)

	def seq_update(self, x, weights, estimate):

		sz, cond_vals, idx_vals, eobs_vals, given_vals = x

		for i in range(len(cond_vals)):
			if cond_vals[i] in self.accumulator_map:
				self.accumulator_map[cond_vals[i]].seq_update(eobs_vals[i], weights[idx_vals[i]], estimate.dmap[cond_vals[i]])
			else:
				if self.default_accumulator is not None:
					if estimate is None:
						self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], None)
					else:
						self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], estimate.default_dist)

		if self.given_accumulator is not None:
			if estimate is None:
				self.given_accumulator.seq_update(given_vals, weights, None)
			else:
				self.given_accumulator.seq_update(given_vals, weights, estimate.given_dist)

	def combine(self, suff_stat):

		for k,v in suff_stat[0].items():
			if k in self.accumulator_map:
				self.accumulator_map[k].combine(v)
			else:
				self.accumulator_map[k] = v

		if self.default_accumulator is not None and suff_stat[1] is not None:
			self.default_accumulator.combine(suff_stat[1])

		if self.given_accumulator is not None and suff_stat[2] is not None:
			self.given_accumulator.combine(suff_stat[2])

		return self

	def value(self):
		rv3 = None if self.given_accumulator is None else self.given_accumulator.value()
		rv2 = None if self.default_accumulator is None else self.default_accumulator.value()
		rv1 = {k: v.value() for k, v in self.accumulator_map.items()}
		return rv1, rv2, rv3

	def from_value(self, x):

		for k,v in x[0].items():
			self.accumulator_map[k].from_value(v)

		if self.default_accumulator is not None and x[1] is not None:
			self.default_accumulator.from_value(x[1])

		if self.given_accumulator is not None and x[2] is not None:
			self.given_accumulator.from_value(x[2])

		return self

	def key_merge(self, stats_dict):
		for k,v in self.accumulator_map.items():
			v.key_merge(stats_dict)

		if self.default_accumulator is not None:
			self.default_accumulator.key_merge(stats_dict)

		if self.given_accumulator is not None:
			self.given_accumulator.key_merge(stats_dict)

	def key_replace(self, stats_dict):
		for k,v in self.accumulator_map.items():
			v.key_replace(stats_dict)

		if self.default_accumulator is not None:
			self.default_accumulator.key_replace(stats_dict)

		if self.given_accumulator is not None:
			self.given_accumulator.key_replace(stats_dict)


class ConditionalDistributionAccumulatorFactory(object):

	def __init__(self, factory_map, default_factory, given_factory, keys):

		self.factory_map     = factory_map
		self.default_factory = default_factory
		self.given_factory   = given_factory
		self.keys            = keys

	def make(self):
		acc = {k: v.make() for k,v in self.factory_map.items()}
		def_acc = None if self.default_factory is None else self.default_factory.make()
		given_acc = None if self.given_factory is None else self.given_factory.make()

		return ConditionalDistributionAccumulator(acc, def_acc, given_acc, self.keys)


class ConditionalDistributionEstimator(ParameterEstimator):

	def __init__(self, estimator_map, default_estimator=None, given_estimator=None, name=None, keys=None):

		self.estimator_map = estimator_map
		self.default_estimator = default_estimator
		self.keys = keys
		self.given_estimator = given_estimator
		self.name = name

	def accumulatorFactory(self):

		emap_items = {k: v.accumulatorFactory() for k,v in self.estimator_map.items()}
		def_factory = None if self.default_estimator is None else self.default_estimator.accumulatorFactory()
		given_factory = None if self.given_estimator is None else self.given_estimator.accumulatorFactory()

		return ConditionalDistributionAccumulatorFactory(emap_items, def_factory, given_factory, self.keys)

	def estimate(self, nobs, suff_stat):

		if self.default_estimator is not None:
			default_dist = self.default_estimator.estimate(None, suff_stat[1])
		else:
			default_dist = None

		if self.given_estimator is not None:
			given_dist = self.given_estimator.estimate(None, suff_stat[2])
		else:
			given_dist = None

		dist_map = {k : self.estimator_map[k].estimate(None, v) for k,v in suff_stat[0].items()}


		return ConditionalDistribution(dist_map, default_dist=default_dist, given_dist=given_dist, name=self.name, keys=self.keys)

