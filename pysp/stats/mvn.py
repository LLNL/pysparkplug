from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import pysp.utils.vector as vec
import numpy as np
import scipy.linalg

class MultivariateGaussianDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, mu, covar, name=None):

		self.dim      = len(mu)
		self.mu       = np.asarray(mu, dtype=float)
		self.covar    = np.asarray(covar, dtype=float)
		self.covar    = np.reshape(self.covar, (len(self.mu), len(self.mu)))
		self.chol     = scipy.linalg.cho_factor(self.covar)
		self.name     = name

		if self.chol is None:
			raise RuntimeError('Cannot obtain Choleskey factorization for covariance matrix.')
		else:
			self.useLSTSQ  = False
			self.cholConst = -0.5*(len(self.mu)*np.log(2.0*pi) + 2.0*np.log(vec.diag(self.chol[0])).sum())

	def __str__(self):
		s1 = repr(list(self.mu))
		s2 = repr([list(u) for u in self.covar])
		s3 = repr(self.name)
		return 'MultivariateGaussianDistribution(%s, %s, name=%s)'%(s1, s2, s3)

	def density(self, x):
		return exp(self.log_density(x))

	def log_density(self, x):

		if self.useLSTSQ:
			raise RuntimeError('Least-squares log-likelihood evaluation not supported.')
		else:
			diff = self.mu - x
			soln = scipy.linalg.cho_solve(self.chol, diff.T).T

			try:
				if np.ndim(x) == 2:
					rv = self.cholConst - 0.5*((diff*soln).sum(axis=1))
				else:
					rv = self.cholConst - 0.5*((diff*soln).sum())
			except Exception as e:
				raise e

			return rv

	def seq_encode(self, x):
		return np.reshape(np.asarray(x), (-1, self.dim))

	def seq_log_density(self, x):

		if self.useLSTSQ:
			return np.ones(x.shape[0])
		else:
			return self.log_density(x)

	def sampler(self, seed=None):
		return MultivariateGaussianSampler(self, seed)

	def estimator(self, pseudo_count=None):

		if pseudo_count is None:
			return MultivariateGaussianEstimator(name=self.name)
		else:
			pseudo_count = (pseudo_count, pseudo_count)
			return MultivariateGaussianEstimator(pseudo_count=pseudo_count, suff_stat=(self.mu, self.covar), name=self.name)


class MultivariateGaussianSampler(object):

	def __init__(self, dist, seed=None):
		self.rng  = RandomState(seed)
		self.dist = dist

	def sample(self, size=None):
		return self.rng.multivariate_normal(mean=self.dist.mu, cov=self.dist.covar, size=size)


class MultivariateGaussianAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, dim=None):

		self.dim     = dim
		self.count   = 0.0

		if dim is not None:
			self.sum  = vec.zeros(dim)
			self.sum2 = vec.zeros((dim,dim))
		else:
			self.sum = None
			self.sum2 = None

	def update(self, x, weight, estimate):

		if self.dim is None:
			self.dim  = len(x)
			self.sum  = vec.zeros(self.dim)
			self.sum2 = vec.zeros((self.dim, self.dim))

		xWeight    = x*weight
		self.sum  += xWeight
		self.sum2 += vec.outer(x, xWeight)
		self.count += weight

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)

	def seq_update(self, x, weights, estimate):

		if self.dim is None:
			self.dim  = x.shape[1]
			self.sum  = vec.zeros(self.dim)
			self.sum2 = vec.zeros((self.dim, self.dim))

		xWeight    = np.multiply(x.T, weights)
		self.count += weights.sum()
		self.sum   += xWeight.sum(axis=1)
		self.sum2  += np.einsum('ji,ik->jk', xWeight, x)

	def combine(self, suff_stat):

		if suff_stat[0] is not None and self.sum is not None:
			self.sum  += suff_stat[0]
			self.sum2 += suff_stat[1]
			self.count += suff_stat[2]

		elif suff_stat[0] is not None and self.sum is None:
			self.sum  = suff_stat[0]
			self.sum2 = suff_stat[1]
			self.count = suff_stat[2]

		return self

	def value(self):
		return self.sum, self.sum2, self.count

	def from_value(self, x):
		self.sum = x[0]
		self.sum2 = x[1]
		self.count = x[2]


class MultivariateGaussianEstimator(object):

	def __init__(self, dim=None, pseudo_count=(None, None), suff_stat = (None, None), name=None):

		dim_loc = dim if dim is not None else ((None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1])))) if suff_stat[0] is None else len(suff_stat[0]))

		self.dim            = dim_loc
		self.is_diag        = False
		self.pseudo_count   = pseudo_count
		self.priorMu        = None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
		self.priorCovar     = None if suff_stat[1] is None else np.reshape(suff_stat[1], (dim_loc, dim_loc))
		self.name           = name

	def accumulatorFactory(self):
		dim = self.dim
		obj = type('', (object,), {'make': lambda o: MultivariateGaussianAccumulator(dim=dim)})()
		return(obj)

	def estimate(self, nobs, suff_stat):

		nobs = suff_stat[2]
		pc1, pc2 = self.pseudo_count

		if pc1 is not None and self.priorMu is not None:
			mu = (suff_stat[0] + pc1*self.priorMu)/(nobs + pc1)
		else:
			mu = suff_stat[0] / nobs

		if pc2 is not None and self.priorCovar is not None:
			covar = (suff_stat[1] + (pc2 * self.priorCovar) - vec.outer(mu, mu*nobs))/(nobs + pc2)
		else:
			covar = (suff_stat[1]/nobs) - vec.outer(mu, mu)

		return MultivariateGaussianDistribution(mu, covar, name=self.name)
