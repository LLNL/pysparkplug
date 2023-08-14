from typing import Dict, Optional
from pysp.arithmetic import maxint
from pysp.bstats.pdist import ProbabilityDistribution, SequenceEncodableAccumulator, \
	ParameterEstimator
from numpy.random import RandomState
import numpy as np
from pysp.bstats.nulldist import null_dist


class ConditionalDistribution(ProbabilityDistribution):

	def __init__(self, dmap, cond_dist, default_dist=null_dist, pass_value=False):
		self.dmap         = dmap
		self.cond_dist    = cond_dist
		self.default_dist = default_dist
		self.pass_value   = pass_value

	def __str__(self):
		return 'ConditionalDistribution(%s, default_dist=%s)' % (str({k: str(v) for k,v in self.dmap.items()}), str(self.default_dist))

	def log_density(self, x):
		if self.pass_value:
			return self.dmap.get(x[0], self.default_dist).log_density(x)
		else:
			return self.dmap.get(x[0], self.default_dist).log_density(x[1])

	def seq_log_density(self, x):

		sz, cond_vals, idx_vals, eobs_vals = x

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

		return rv


	def seq_encode(self, x):

		cond_enc = dict()

		for i in range(len(x)):
			xx = x[i]
			vv = xx if self.pass_value else xx[1]
			if xx[0] not in cond_enc:
				cond_enc[xx[0]] = [[vv], [i]]
			else:
				cond_enc_loc = cond_enc[xx[0]]
				cond_enc_loc[0].append(vv)
				cond_enc_loc[1].append(i)

		cond_enc = list(cond_enc.items())

		cond_vals  = tuple([u[0] for u in cond_enc])
		eobs_vals  = tuple([self.dmap.get(u[0], self.default_dist).seq_encode(u[1][0]) for u in cond_enc])
		idx_vals   = tuple([np.asarray(u[1][1]) for u in cond_enc])

		return len(x), cond_vals, idx_vals, eobs_vals


	def sampler(self, seed=None):
		pass

	def estimator(self, pseudo_count=None):
		pass


class ConditionalDistributionSampler(object):
	def __init__(self, dist, seed=None):
		pass

	def sample(self, size=None):
		pass


class ConditionalDistributionEstimatorAccumulator(SequenceEncodableAccumulator):

	def __init__(self, accumulator_map, default_accumulator, keys=None):
		self.accumulator_map     = accumulator_map
		self.default_accumulator = default_accumulator
		self.key                 = keys

	def update(self, x, weight, estimate):
		if x[0] in self.accumulator_map:
			self.accumulator_map[x[0]].update(x[1], weight, estimate.dmap[x[0]])
		else:
			if self.default_accumulator is not None:
				self.default_accumulator.update(x[1], weight, estimate.default_dist)

	def initialize(self, x, weight, rng):
		if x[0] in self.accumulator_map:
			self.accumulator_map[x[0]].initialize(x[1], weight, rng)
		else:
			if self.default_accumulator is not None:
				self.default_accumulator.initialize(x[1], weight, rng)

	def seq_update(self, x, weights, estimate):
		sz, cond_vals, idx_vals, eobs_vals = x

		for i in range(len(cond_vals)):
			if cond_vals[i] in self.accumulator_map:
				self.accumulator_map[cond_vals[i]].seq_update(eobs_vals[i], weights[idx_vals[i]], estimate.dmap[cond_vals[i]])
			else:
				if self.default_accumulator is not None:
					self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], estimate.default_dist)

	def combine(self, suff_stat):

		for k,v in suff_stat[0].items():
			if k in self.accumulator_map:
				self.accumulator_map[k].combine(v)
			else:
				self.accumulator_map[k] = v

		if self.default_accumulator is not None and suff_stat[1] is not None:
			self.default_accumulator.combine(suff_stat[1])

		return self

	def value(self):
		rv2 = None if self.default_accumulator is None else self.default_accumulator.value()
		rv1 = {k: v.value() for k, v in self.accumulator_map.items()}
		return rv1, rv2

	def from_value(self, x):
		for k,v in x[0].items():
			self.accumulator_map[k].from_value(v)

		if self.default_accumulator is not None and x[1] is not None:
			self.default_accumulator.from_value(x[1])

		return self

	def key_merge(self, stats_dict):
		for k,v in self.accumulator_map.items():
			v.key_merge(stats_dict)

	def key_replace(self, stats_dict):
		for k,v in self.accumulator_map.items():
			v.key_replace(stats_dict)



class ConditionalDistributionEstimator(ParameterEstimator):
	def __init__(self, estimator_map, default_estimator=None, keys=None):
		self.estimator_map = estimator_map
		self.default_estimator = default_estimator
		self.keys = keys

	def accumulator_factory(self):
		emap_items = self.estimator_map.items()

		obj = type('', (object,), {'make': lambda o: ConditionalDistributionEstimatorAccumulator({k : v.accumulator_factory().make() for k,v in emap_items}, None if self.default_estimator is None else self.default_estimator.accumulator_factory().make(), self.keys)})()
		# def makeL():
		#	return(CompositeEstimatorAccumulator([x.accumulatorFactory().make() for x in self.estimators]))
		# obj = AccumulatorFactory(makeL)
		return (obj)

	def estimate(self, suff_stat):

		if self.default_estimator is not None:
			default_dist = self.default_estimator.estimate(suff_stat[1])
		else:
			default_dist = None

		dist_map = {k : self.estimator_map[k].estimate(v) for k,v in suff_stat[0].items()}


		return ConditionalDistribution(dist_map, default_dist)

