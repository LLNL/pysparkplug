from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
import pysp.utils.vector as vec
import numpy as np
import scipy.linalg
import scipy.special
from scipy.special import gammaln
import sys

def lniv_z(v, ln_z):
	return v*(ln_z - np.log(2.0)) - scipy.special.gammaln(np.exp(ln_z)+1.0)

def lniv_h(v, ln_z):
	pt = ln_z*0.0
	cc = 1.0
	rv = 1.0
	for i in range(10):
		num = np.log(4.0*v*v - (2*i+1)**2)
		den = np.log((i+1)*8) + ln_z
		cc  *= 1.0
		pt  += (num - den)
		rv  += cc*np.exp(pt)
	rv = np.log(rv)
	rv += (np.exp(ln_z) - 0.5 * (np.log(2.0 * pi) + ln_z))
	return rv


def lniv(v,ln_z):

	z  = exp(ln_z)

	if np.isfinite(ln_z):
		rv0 = scipy.special.ive(v,z)
		if rv0 == 0:
			rv = lniv_z(v, ln_z)
		elif np.isposinf(rv0):
			rv = lniv_h(v, ln_z)
		else:
			rv = np.log(rv0) + z
	else:
		rv = 0

	return rv

class VonMisesFisherDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, mu, kappa, name=None):

		dim = len(mu)
		mu  = np.asarray(mu).copy()

		if kappa > 0:
			log_kappa = np.log(kappa)
			cc = log_kappa*((dim/2.0)-1) - lniv((dim / 2.0) - 1.0, log_kappa)

			log_kappa0 = -10000
			# This is a hack to identify the limiting constant as k -> 0
			cc0 = log_kappa0 * ((dim / 2.0) - 1) - lniv((dim / 2.0) - 1.0, log_kappa0)
			cc  = (cc - cc0)+gammaln(dim/2.0)
		else:
			cc = gammaln(dim/2.0)


		self.name = name
		self.dim = dim
		self.mu  = mu
		self.kappa = kappa
		self.log_const = cc - np.log(2.0*pi)*(dim/2.0)

	def __str__(self):
		s1 = repr(list(self.mu))
		s2 = repr(self.kappa)
		s3 = repr(self.name)
		return 'VonMisesFisherDistribution(%s, %s, name=%s)'%(s1, s2, s3)

	def density(self, x):
		return exp(self.log_density(x))

	def log_density(self, x):
		z = np.asarray(x).copy()
		return np.dot(z, self.mu)*self.kappa + self.log_const

	def seq_encode(self, x):
		rv = np.asarray(x).copy()
		return rv

	def seq_log_density(self, x):
		return np.dot(x, self.mu) * self.kappa + self.log_const

	def sampler(self, seed=None):
		return VonMisesFisherSampler(self, seed)

	def estimator(self, pseudo_count=None):

		if pseudo_count is None:
			return VonMisesFisherEstimator(name=self.name)
		else:
			return VonMisesFisherEstimator(name=self.name)


class VonMisesFisherSampler(object):

	def __init__(self, dist, seed=None):
		self.rng  = RandomState(seed)
		self.dist = dist

	def sample(self, size=None):

		rng1 = np.random.RandomState(self.rng.randint(maxrandint))
		rng2 = np.random.RandomState(self.rng.randint(maxrandint))
		rng3 = np.random.RandomState(self.rng.randint(maxrandint))

		d  = self.dist.dim
		mu = self.dist.mu
		k  = self.dist.kappa

		t1 = np.sqrt(4.0*k*k + (d-1.0)*(d-1.0))
		#b = (d-1.0)/(t1 + 2*k)
		b = (t1 - 2*k)/(d-1.0)
		x0 = (1.0-b)/(1.0+b)

		m  = (d - 1.0)/2.0
		c  = k*x0 + (d - 1.0)*np.log(1 - x0*x0)

		sz = 1 if size is None else size
		rv = np.zeros((sz, d))

		QQ = np.zeros((d,d), dtype=float)
		QQ[0,:] = mu
		_, s, vh = scipy.linalg.svd(QQ)
		QQ = vh[np.abs(s) < 0.1, :].T

		for i in range(sz):

			t = c-1
			u = 1

			while (t-c) < np.log(u):
				z = rng1.beta(m,m)
				u = rng2.rand()
				w = (1.0 - (1.0+b)*z)/(1.0 - (1 - b)*z)
				t = k*w + (d-1)*np.log(1.0 - x0*w)

			v = rng3.randn(d-1)
			v = np.dot(QQ, v)
			v /= np.sqrt(np.dot(v,v))
			rv[i,:] = np.sqrt(1-w*w)*v + w*mu


		if size is None:
			return rv[0,:]
		else:
			return rv


class VonMisesFisherAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, dim=None):

		self.dim     = dim
		self.count   = 0.0

		if dim is not None:
			self.ssum  = vec.zeros(dim)
		else:
			self.ssum = None


	def update(self, x, weight, estimate):

		if self.dim is None:
			self.dim  = len(x)
			self.ssum  = vec.zeros(self.dim)

		self.ssum  += x*weight
		self.count += weight

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)

	def seq_update(self, x, weights, estimate):

		if self.dim is None:
			self.dim  = x.shape[1]
			self.ssum  = vec.zeros(self.dim)

		good_w = np.bitwise_and(np.isfinite(weights), weights >= 0)
		if np.all(good_w):
			xWeight = np.multiply(x.T, weights)
		else:
			xWeight = np.multiply(x[good_w, :].T, weights[good_w])

		self.count += weights.sum()
		self.ssum   += xWeight.sum(axis=1)


	def combine(self, suff_stat):

		if suff_stat[1] is not None and self.ssum is not None:
			self.ssum  += suff_stat[1]
			self.count += suff_stat[0]

		elif suff_stat[1] is not None and self.ssum is None:
			self.ssum  = suff_stat[1]
			self.count = suff_stat[0]

		return self

	def value(self):
		return self.count, self.ssum

	def from_value(self, x):
		self.ssum = x[1]
		self.count = x[0]


class VonMisesFisherEstimator(object):

	def __init__(self, dim=None, pseudo_count=None, name=None):

		self.dim            = dim
		self.pseudo_count    = pseudo_count
		self.name           = name

	def accumulatorFactory(self):
		dim = self.dim
		obj = type('', (object,), {'make': lambda o: VonMisesFisherAccumulator(dim=dim)})()
		return(obj)

	def estimate(self, nobs, suff_stat):

		count, ssum = suff_stat
		dim = len(ssum)

		def _newton(p, r, k):
			k = max(sys.float_info.min, k)
			#apk = scipy.special.iv(p/2.0, k)/scipy.special.iv((p/2.0)-1.0, k)
			apk = np.exp(lniv(p/2.0, np.log(k))-lniv((p/2.0)-1.0, np.log(k)))

			rv = k - (apk - r) / (1.0 - apk * apk - ((p - 1.0) / k) * apk)
			rv = max(sys.float_info.min, rv)
			return rv

		ssum_norm = np.sqrt(np.dot(ssum, ssum))

		if ssum_norm > 0 and count > 0:
			rhat = ssum_norm / count
			mu = ssum / ssum_norm

			if rhat != 1.0:
				k = rhat*(dim-(rhat*rhat))/(1.0 - (rhat*rhat))
			else:
				k = 1.1 * (dim - 1.1**2) / (1.0 - (1.1**2))
			dk = 1

			for i in range(3):
				#while dk > 1.0e-12:
				#old_k = k
				k  = _newton(dim, rhat, k)
				#dk = np.abs(k-old_k)/old_k

		else:
			mu = np.ones(dim)/np.sqrt(dim)
			k  = 0.0

		return VonMisesFisherDistribution(mu, k, name=self.name)
