from numpy.random import RandomState
from pysp.arithmetic import maxint, log, exp, one
from pysp.stats.pdist import ProbabilityDistribution

class WeightedDistribution(ProbabilityDistribution):
	def __init__(self, dist, logWeighted=False):
		self.dist = dist
		self.logWeighted = logWeighted

	def __str__(self):
		return 'WeightedDistribution(%s, logWeighted=%s)'%(str(self.dist), str(self.logWeighted))

	def density(self, x):
		if self.logWeighted:
			return self.dist.density(x[0])*exp(x[1])
		else:
			return self.dist.density(x[0])*x[1]

	def logDensity(self, x):
		if self.logWeighted:
			return self.dist.logDensity(x[0]) + x[1]
		else:
			return self.dist.logDensity(x[0]) + log(x[1])

	def weight(self, x):
		return self.dist.weight(x[0])*x[1]

	def sampler(self, seed=None):
		return WeightedSampler(self, seed)

	def estimator(self, pseudo_count=None):
		return WeightedEstimator(self.dist.estimator(pseudo_count=pseudo_count))


class WeightedSampler(object):
	def __init__(self, dist, seed=None):
		self.dist        = dist
		self.rng         = RandomState(seed)
		self.distSampler = self.dist.dist.sampler(seed=self.rng.randint(maxint))

	def sample(self, size=None):

		if size is None:
			return (self.distSampler.sample(), one)
		else:
			return [self.sample() for i in range(size)]


class WeightedEstimatorAccumulator(object):
	def __init__(self, accumulator):
		self.accumulator = accumulator

	def update(self, x, weight):
		self.accumulator.update(x[0], weight*x[1])

	def combine(self, suff_stat):
		self.accumulator.combine(suff_stat)
		return self

	def value(self):
		return self.accumulator.value()


class WeightedEstimator(object):
	def __init__(self, estimator, logWeighted=False):
		self.estimator   = estimator
		self.logWeighted = logWeighted

	def accumulatorFactory(self):
		obj = type('', (object,), {'make': lambda o: WeightedEstimatorAccumulator(self.estimator.accumulatorFactory().make())})()

		return (obj)

	def estimate(self, nobs, suff_stat):
		return WeightedDistribution(self.estimator.estimate(nobs, suff_stat))

