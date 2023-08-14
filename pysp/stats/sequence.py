import numpy as np
from numpy.random import RandomState
from pysp.arithmetic import maxrandint
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator

class SequenceDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, dist, len_dist=None, len_normalized=False, name=None):
		self.dist = dist
		self.len_dist = len_dist
		self.len_normalized = len_normalized
		self.name = name

	def __str__(self):
		s1 = str(self.dist)
		s2 = str(self.len_dist)
		s3 = repr(self.len_normalized)
		s4 = repr(self.name)
		return 'SequenceDistribution(%s, len_dist=%s, len_normalized=%s, name=%s)'%(s1, s2, s3, s4)

	def density(self, x):
		rv = 1.0
		for i in range(len(x)):
			rv *= self.dist.density(x[i])

		if self.len_normalized and len(x) > 0:
			rv = np.power(rv, 1.0/len(x))

		return rv

	def log_density(self, x):
		rv = 0.0
		for i in range(len(x)):
			rv += self.dist.log_density(x[i])

		if self.len_normalized and len(x) > 0:
			rv /= len(x)

		if self.len_dist is not None:
			rv += self.len_dist.log_density(len(x))

		return rv

	def seq_ld_lambda(self):
		rv = self.dist.seq_ld_lambda()
		if self.len_dist is not None:
			rv.extend(self.len_dist.seq_ld_lambda())
		return rv

	def seq_log_density(self, x):

		idx, icnt, inz, enc_seq, enc_nseq = x

		if np.all(icnt==0):
			ll_sum = np.zeros(len(icnt), dtype=float)

		else:
			ll = self.dist.seq_log_density(enc_seq)
			ll_sum = np.bincount(idx, weights=ll, minlength=len(icnt))

			if self.len_normalized:
				ll_sum *= icnt

		if self.len_dist is not None and enc_nseq is not None:
			nll = self.len_dist.seq_log_density(enc_nseq)
			ll_sum += nll

		return ll_sum


	def seq_encode(self, x):

		tx   = []
		nx   = []
		tidx = []
		for i in range(len(x)):
			nx.append(len(x[i]))
			for j in range(len(x[i])):
				tidx.append(i)
				tx.append(x[i][j])

		rv1 = np.asarray(tidx, dtype=int)
		rv2 = np.asarray(nx, dtype=float)
		rv3 = (rv2 != 0)

		rv2[rv3] = 1.0/rv2[rv3]
		#rv2[rv3] = 1.0

		rv4 = self.dist.seq_encode(tx)

		if self.len_dist is not None:
			rv5 = self.len_dist.seq_encode(nx)
		else:
			rv5 = None

		return rv1, rv2, rv3, rv4, rv5

	def sampler(self, seed=None):
		return SequenceSampler(self, seed)

	def estimator(self, pseudo_count=None):

		if self.len_dist is not None:
			len_est = self.len_dist.estimator(pseudo_count=pseudo_count)
		else:
			len_est = None

		return SequenceEstimator(self.dist.estimator(pseudo_count=pseudo_count), len_estimator=len_est, len_normalized=self.len_normalized, name=self.name)


class SequenceSampler(object):
	def __init__(self, dist, seed=None):
		self.dist        = dist
		self.rng         = RandomState(seed)
		self.distSampler = self.dist.dist.sampler(seed=self.rng.randint(0, maxrandint))
		self.lenSampler  = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

	def sample(self, size=None):

		if size is None:
			n = self.lenSampler.sample()
			return [self.distSampler.sample() for i in range(n)]
		else:
			return [self.sample() for i in range(size)]


class SequenceAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, accumulator, len_normalized=False, len_accumulator=None, keys=None):
		self.accumulator = accumulator
		self.len_accumulator = len_accumulator
		self.key = keys
		self.len_normalized = len_normalized

	def update(self, x, weight, estimate):

		if estimate is None:
			w = weight / len(x) if (self.len_normalized and len(x) > 0) else weight

			for i in range(len(x)):
				self.accumulator.update(x[i], w, None)

			if self.len_accumulator is not None:
				self.len_accumulator.update(len(x), weight, None)

		else:
			w = weight / len(x) if (self.len_normalized and len(x) > 0) else weight

			for i in range(len(x)):
				self.accumulator.update(x[i], w, estimate.dist)

			if self.len_accumulator is not None:
				self.len_accumulator.update(len(x), weight, estimate.len_dist)

	def initialize(self, x, weight, rng):

		if len(x) > 0:
			w = weight/len(x) if self.len_normalized else weight
			for xx in x:
				self.accumulator.initialize(xx, w, rng)

		if self.len_accumulator is not None:
			self.len_accumulator.initialize(len(x), weight, rng)

	def combine(self, suff_stat):
		self.accumulator.combine(suff_stat[0])
		if self.len_accumulator is not None:
			self.len_accumulator.combine(suff_stat[1])
		return self

	def value(self):
		if self.len_accumulator is not None:
			return self.accumulator.value(), self.len_accumulator.value()
		else:
			return self.accumulator.value(), None

	def from_value(self, x):
		self.accumulator.from_value(x[0])
		if self.len_accumulator is not None:
			self.len_accumulator.from_value(x[1])
		return self

	def get_seq_lambda(self):
		rv = self.accumulator.get_seq_lambda()
		if self.len_accumulator is not None:
			rv.extend(self.len_accumulator.get_seq_lambda())
		return rv

	def seq_update(self, x, weights, estimate):
		idx, icnt, inz, enc_seq, enc_nseq = x

		w = weights[idx]*icnt[idx] if self.len_normalized else weights[idx]

		self.accumulator.seq_update(enc_seq, w, estimate.dist if estimate is not None else None)

		if self.len_accumulator is not None:
			self.len_accumulator.seq_update(enc_nseq, weights, estimate.len_dist if estimate is not None else None)

	def key_merge(self, stats_dict):

		if self.key is not None:
			if self.key in stats_dict:
				stats_dict[self.key].combine(self.value())
			else:
				stats_dict[self.key] = self

		self.accumulator.key_merge(stats_dict)

		if self.len_accumulator is not None:
			self.len_accumulator.key_merge(stats_dict)


	def key_replace(self, stats_dict):

		if self.key is not None:
			if self.key in stats_dict:
				self.from_value(stats_dict[self.key].value())

		self.accumulator.key_replace(stats_dict)

		if self.len_accumulator is not None:
			self.len_accumulator.key_replace(stats_dict)

class SequenceAccumulatorFactory(object):

	def __init__(self, dist_factory, len_normalized, len_factory, keys):
		self.dist_factory = dist_factory
		self.len_factory = len_factory
		self.len_normalized = len_normalized
		self.keys = keys

	def make(self):
		len_acc = None if self.len_factory is None else self.len_factory.make()
		return SequenceAccumulator(self.dist_factory.make(), self.len_normalized, len_acc, self.keys)

class SequenceEstimator(ParameterEstimator):

	def __init__(self, estimator, len_estimator=None, len_dist=None, len_normalized=False, name=None, keys=None):
		self.estimator = estimator
		self.len_estimator = len_estimator
		self.len_dist = len_dist
		self.keys = keys
		self.len_normalized=len_normalized
		self.name = name

	def accumulatorFactory(self):
		len_factory = None if self.len_estimator is None else self.len_estimator.accumulatorFactory()
		dist_factory = self.estimator.accumulatorFactory()
		return SequenceAccumulatorFactory(dist_factory, self.len_normalized, len_factory, self.keys)

	def estimate(self, nobs, suff_stat):

		if self.len_estimator is None:
			return SequenceDistribution(self.estimator.estimate(nobs, suff_stat[0]), len_dist=self.len_dist, len_normalized=self.len_normalized, name=self.name)
		else:
			return SequenceDistribution(self.estimator.estimate(nobs, suff_stat[0]), len_dist=self.len_estimator.estimate(nobs, suff_stat[1]), len_normalized=self.len_normalized, name=self.name)

