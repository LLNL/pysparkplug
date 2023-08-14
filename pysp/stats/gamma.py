import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from pysp.utils.special import gammaln, digamma, trigamma


class GammaDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, k, theta, name=None):
		self.k         = k
		self.theta     = theta
		self.log_const = -(gammaln(k) + k*log(theta))
		self.name      = name

	def __str__(self):
		return 'GammaDistribution(%s, %s, name=%s)' % (repr(self.k), repr(self.theta), repr(self.name))

	def density(self, x):
		return exp(self.log_const + (self.k-one)*log(x) - x/self.theta)

	def log_density(self, x):
		return self.log_const + (self.k-one)*log(x) - x/self.theta

	def seq_log_density(self, x):
		rv = x[0]*(-1.0/(self.theta))
		if self.k != 1.0:
			rv += x[1]*(self.k - 1.0)
		rv += self.log_const
		return rv

	def seq_encode(self, x):
		rv1 = np.asarray(x)
		rv2 = np.log(rv1)
		return rv1,rv2

	def sampler(self, seed=None):
		return GammaSampler(self, seed)

	def estimator(self, pseudo_count=None):

		if pseudo_count is None:
			return GammaEstimator(name=self.name)
		else:
			suff_stat = (self.k * self.theta, exp(digamma(self.k) + log(self.theta)))
			pseudo_count = (pseudo_count, pseudo_count)
			return GammaEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat, name=self.name)


class GammaSampler(object):

	def __init__(self, dist, seed=None):
		self.rng  = RandomState(seed)
		self.dist = dist

	def sample(self, size=None):
		return self.rng.gamma(shape=self.dist.k, scale=self.dist.theta, size=size)


class GammaAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, keys):
		self.nobs        = zero
		self.sum         = zero
		self.sum_of_logs = zero
		self.key         = keys

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)

	def update(self, x, weight, estimate):
		self.nobs        += weight
		self.sum         += x*weight
		self.sum_of_logs += log(x)*weight

	def seq_update(self, x, weights, estimate):
		self.sum += np.dot(x[0], weights)
		self.sum_of_logs += np.dot(x[1], weights)
		self.nobs += np.sum(weights)

	def combine(self, suff_stat):
		self.nobs        += suff_stat[0]
		self.sum         += suff_stat[1]
		self.sum_of_logs += suff_stat[2]

		return self

	def value(self):
		return self.nobs, self.sum, self.sum_of_logs

	def from_value(self, x):
		self.nobs = x[0]
		self.sum = x[1]
		self.sum_of_logs = x[2]
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



class GammaEstimator(ParameterEstimator):

	def __init__(self, pseudo_count=(0.0, 0.0), suff_stat=(1.0, 0.0), threshold=1.0e-8, name=None, keys=None):

		self.pseudo_count = pseudo_count
		self.suff_stat    = suff_stat
		self.threshold    = threshold
		self.keys         = keys
		self.name         = name

	def accumulatorFactory(self):
		obj = type('', (object,), {'make': lambda o: GammaAccumulator(self.keys)})()
		return (obj)

	def estimate(self, nobs, suff_stat):

		pc1,pc2 = self.pseudo_count
		ss1,ss2 = self.suff_stat

		if suff_stat[0] == 0:
			return GammaDistribution(1.0, 1.0)

		adj_sum   = suff_stat[1] + ss1*pc1
		adj_cnt   = suff_stat[0] + pc1
		adj_mean  = adj_sum/adj_cnt

		adj_lsum  = suff_stat[2] + ss2*pc2
		adj_lcnt  = suff_stat[0] + pc2
		adj_lmean = adj_lsum/adj_lcnt

		k = self.estimate_shape(adj_mean, adj_lmean, self.threshold)

		return GammaDistribution(k, adj_sum/(k*adj_lcnt), name=self.name)

	@staticmethod
	def estimate_shape(avg_sum, avg_sum_of_logs, threshold):
		s = log(avg_sum) - avg_sum_of_logs
		old_k = inf
		k = (3 - s + sqrt((s-3)*(s-3) + 24*s))/(12*s)
		while abs(old_k-k) > threshold:
			old_k = k
			k    -= (log(k) - digamma(k) - s)/(one/k - trigamma(k))
		return k
