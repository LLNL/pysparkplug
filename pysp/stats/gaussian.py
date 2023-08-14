from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import numpy as np


class GaussianDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, mu, sigma2, name=None):
		self.mu       = mu
		self.sigma2   = 1.0 if (sigma2 <= 0 or isnan(sigma2) or isinf(sigma2)) else sigma2
		self.logConst = -0.5*log(2.0*pi*self.sigma2)
		self.const    = 1.0/sqrt(2.0*pi*self.sigma2)
		self.name = name

	def __str__(self):
		return 'GaussianDistribution(%s, %s, name=%s)'%(repr(self.mu), repr(self.sigma2), repr(self.name))

	def density(self, x):
		return self.const * exp(-0.5*(x-self.mu)*(x-self.mu)/self.sigma2)

	def log_density(self, x):
		return self.logConst - 0.5*(x-self.mu)*(x-self.mu)/self.sigma2

	def seq_ld_lambda(self):
		return [self.seq_log_density]

	def seq_log_density(self, x):
		rv = x - self.mu
		rv *= rv
		rv *= -0.5/self.sigma2
		rv += self.logConst
		return rv

	def seq_encode(self, x):
		rv = np.asarray(x)
		return rv

	def sampler(self, seed=None):
		return GaussianSampler(self, seed)

	def estimator(self, pseudo_count=None):

		if pseudo_count is None:
			return GaussianEstimator(name=self.name)
		else:
			pseudo_count = (pseudo_count,pseudo_count)
			suff_stat = (self.mu, self.sigma2)
			return GaussianEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat, name=self.name)


class GaussianSampler(object):

	def __init__(self, dist, seed=None):
		self.rng  = RandomState(seed)
		self.dist = dist

	def sample(self, size=None):
		return self.rng.normal(loc=self.dist.mu, scale=sqrt(self.dist.sigma2), size=size)


class GaussianAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, keys=(None, None)):
		self.sum  = 0.0
		self.sum2 = 0.0
		self.count = 0.0
		self.count2 = 0.0
		self.sum_key = keys[0]
		self.sum2_key = keys[1]

	def update(self, x, weight, estimate):
		xWeight   = x*weight
		self.sum  += xWeight
		self.sum2 += x*xWeight
		self.count += weight
		self.count2 += weight

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)


	def seq_update(self, x, weights, estimate):
		self.sum += np.dot(x, weights)
		self.sum2 += np.dot(x*x, weights)
		w_sum = weights.sum()
		self.count += w_sum
		self.count2 += w_sum


	def combine(self, suff_stat):
		self.sum  += suff_stat[0]
		self.sum2 += suff_stat[1]
		self.count += suff_stat[2]
		self.count2 += suff_stat[3]

		return self

	def value(self):
		return self.sum, self.sum2, self.count, self.count2

	def from_value(self, x):
		self.sum = x[0]
		self.sum2 = x[1]
		self.count = x[2]
		self.count2 = x[3]
		return self

	def key_merge(self, stats_dict):
		if self.sum_key is not None:
			if self.sum_key in stats_dict:
				vals = stats_dict[self.sum_key]
				stats_dict[self.sum_key] = (vals[0] + self.count, vals[1] + self.sum)
			else:
				stats_dict[self.sum_key] = (self.count, self.sum)

		if self.sum2_key is not None:
			if self.sum2_key in stats_dict:
				pass


	def key_replace(self, stats_dict):
		if self.sum_key is not None:
			if self.sum_key in stats_dict:
				vals = stats_dict[self.sum_key]
				self.count = vals[0]
				self.sum = vals[1]

		if self.sum2_key is not None:
			if self.sum2_key in stats_dict:
				pass


class GaussianAccumulatorFactory():

	def __init__(self, name, keys):
		self.keys = keys

	def make(self):
		return GaussianAccumulator(keys=self.keys)

class GaussianEstimator(ParameterEstimator):

	def __init__(self, pseudo_count=(None,None), suff_stat=(None,None), name=None, keys=(None, None)):

		self.pseudo_count  = pseudo_count
		self.suff_stat    = suff_stat
		self.keys         = keys
		self.name         = name

		if keys[1] is not None:
			raise RuntimeWarning('Support for keying the variance is currently not available.')

	def accumulatorFactory(self):
		return GaussianAccumulatorFactory(self.name, self.keys)

	def estimate(self, nobs, suff_stat):

		nobs_loc1 = suff_stat[2]
		nobs_loc2 = suff_stat[3]

		if nobs_loc1 == 0:
			mu = 0.0
		elif self.pseudo_count[0] is not None:
			mu     = (suff_stat[0] + self.pseudo_count[0]*self.suff_stat[0])/(nobs_loc1 + self.pseudo_count[0])
		else:
			mu     = suff_stat[0]/nobs_loc1

		if nobs_loc2 == 0:
			sigma2 = 0
		elif self.pseudo_count[1] is not None:
			sigma2 = (suff_stat[1] - mu*mu*nobs_loc2 + self.pseudo_count[1]*self.suff_stat[1])/(nobs_loc2 + self.pseudo_count[1])
		else:
			sigma2 = suff_stat[1]/nobs_loc2 - mu*mu

		return GaussianDistribution(mu, sigma2, name=self.name)

