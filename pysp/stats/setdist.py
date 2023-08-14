from typing import Optional, Dict, Tuple, Any
from numpy.random import RandomState
from collections import defaultdict, OrderedDict
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np

class BernoulliSetDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, pmap: Dict[Any, float], min_prob: float = 1.0e-128, name: Optional[str] = None):

		self.name     = name
		self.pmap     = pmap
		self.required = set()
		self.nlog_sum = 0.0
		self.log_dmap = dict()


		if min_prob == 0:
			for k,v in pmap.items():
				if v == 1.0:
					self.log_dmap[k] = 0.0
					self.required.add(k)
				elif v == 0.0:
					self.log_dmap[k] = -np.inf
				else:
					vv = np.log1p(-v)
					self.log_dmap[k] = np.log(v) - vv
					self.nlog_sum += vv
			self.min_prob = 0.0
			self.num_required = len(self.required)

		else:
			min_pv = np.log(min_prob)
			min_nv = np.log1p(-min_prob)

			for k,v in pmap.items():
				if v == 1.0:
					self.log_dmap[k] = min_nv - min_pv
					self.nlog_sum   += min_pv
				elif v == 0.0:
					self.log_dmap[k] = min_pv - min_nv
					self.nlog_sum   += min_nv
				else:
					vv = np.log1p(-v)
					self.log_dmap[k] = np.log(v) - vv
					self.nlog_sum += vv

			self.min_prob = min_prob
			self.num_required = 0

	def __str__(self):

		s1 = repr(sorted(self.pmap.items(), key=lambda t: t[0]))
		s2 = repr(self.min_prob)
		s3 = repr(self.name)
		return 'BernoulliSetDistribution(dict(%s), min_prob=%s, name=%s)'%(s1, s2, s3)

	def density(self, x):
		return np.exp(self.log_density(x))

	def log_density(self, x):

		if not self.required.issubset(x):
			return -np.inf
		rv = 0.0
		for v in x:
			rv += self.log_dmap[v]
		return self.nlog_sum + rv

	def seq_log_density(self, x):

		sz, idx, val_map_inv, xs = x

		dlog_loc = np.asarray([self.log_dmap[u] for u in val_map_inv], dtype=np.float64)

		rv = np.bincount(idx, weights=dlog_loc[xs], minlength=sz)
		rv += self.nlog_sum

		if self.num_required != 0:
			required_loc = np.isin(val_map_inv, self.required)
			req_cnt = np.bincount(idx, weights=required_loc[xs], minlength=sz)
			rv[req_cnt != self.num_required] = -np.inf

		return rv


	def seq_encode(self, x):

		idx = []
		xs  = []

		for i in range(len(x)):
			idx.extend([i] * len(x[i]))
			xs.extend(x[i])

		val_map, xs = np.unique(xs, return_inverse=True)

		idx = np.asarray(idx, dtype=np.int32)
		xs  = np.asarray(xs, dtype=np.int32)

		return len(x), idx, val_map, xs


	def sampler(self, seed=None):
		return BernoulliSetSampler(self, seed)

	def estimator(self, pseudo_count=None):

		if pseudo_count is None:
			return BernoulliSetEstimator(min_prob=self.min_prob, name=self.name)
		else:
			return BernoulliSetEstimator(min_prob=self.min_prob, pseudo_count=pseudo_count, suff_stat=self.pmap, name=self.name)

class BernoulliSetSampler(object):

	def __init__(self, dist: BernoulliSetDistribution, seed=None):
		self.rng  = RandomState(seed)
		self.dist = dist

	def sample(self, size=None):

		if size is not None:
			retval = [[] for i in range(size)]
			for k,v in self.dist.pmap.items():
				for i in np.flatnonzero(self.rng.rand(size) <= v):
					retval[i].append(k)
			return retval

		else:
			retval = []
			for k,v in self.dist.pmap.items():
				if self.rng.rand() <= v:
					retval.append(k)
			return retval

class BernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, keys=None):
		self.pmap = defaultdict(float)
		self.tot_sum = 0.0
		self.key = keys

	def update(self, x, weight, estimate):
		for u in x:
			self.pmap[u] += weight
		self.tot_sum += weight

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)

	def seq_update(self, x, weights, estimate):

		sz, idx, val_map_inv, xs = x
		agg_cnt = np.bincount(xs, weights[idx])

		for i,v in enumerate(agg_cnt):
			self.pmap[val_map_inv[i]] += v

		self.tot_sum += weights.sum()

	def combine(self, suff_stat):
		for k,v in suff_stat[0].items():
			self.pmap[k] += v
		self.tot_sum += suff_stat[1]
		return self

	def value(self):
		return dict(self.pmap), self.tot_sum

	def from_value(self, x):
		self.pmap = x[0]
		self.tot_sum = x[1]
		return self


	def key_merge(self, stats_dict):
		if self.key is not None:
			if self.key in stats_dict:
				stats_dict[self.key].combine(self.value())
			else:
				stats_dict[self.key] = self


	def key_replace(self, stats_dict):
		if self.key is not None:
			if self.key in stats_dict:
				self.from_value(stats_dict[self.key].value())


class BernoulliSetAccumulatorFactory(object):

	def __init__(self, keys):
		self.keys = keys

	def make(self):
		return BernoulliSetAccumulator(self.keys)


class BernoulliSetEstimator(ParameterEstimator):

	def __init__(self, min_prob: float = 1.0e-128, pseudo_count=None, suff_stat=None, name=None, keys=None):

		self.pseudo_count  = pseudo_count
		self.suff_stat     = suff_stat
		self.keys          = keys
		self.name          = name
		self.min_prob      = min_prob

	def accumulatorFactory(self):
		return BernoulliSetAccumulatorFactory(self.keys)

	def estimate(self, nobs, suff_stat):

		if self.pseudo_count is not None and self.suff_stat is not None:
			keys = set(suff_stat[0].keys())
			keys.update(self.suff_stat.keys())

			pmap = {k: (self.suff_stat.get(k,0.0)*self.pseudo_count + suff_stat[0].get(k, 0.0))/(self.pseudo_count + suff_stat[1]) for k in keys}

		elif self.pseudo_count is not None and self.suff_stat is None:
			p   = self.pseudo_count
			cnt = float(p + suff_stat[1])
			pmap = {k: (v + (p/2.0))/cnt for k,v in suff_stat[0].items()}

		else:

			if suff_stat[1] != 0:
				pmap = {k: v/suff_stat[1] for k,v in suff_stat[0].items()}
			else:
				pmap = {k : 0.5 for k in suff_stat[0].keys()}

		return BernoulliSetDistribution(pmap, min_prob=self.min_prob, name=self.name)


