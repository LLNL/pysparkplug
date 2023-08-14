from numpy.random import RandomState
from pysp.arithmetic import *
from pysp.stats.pdist import ProbabilityDistribution

class SelectDistribution(ProbabilityDistribution):

	def __init__(self, dists, choice_function):
		self.dists            = dists
		self.choice_function  = choice_function
		self.count            = len(dists)

	def __str__(self):
		return 'SelectDistribution(' + ','.join([str(u) for u in self.dists]) + ')'

	def density(self, x):
		idx = self.choice_function(x)
		return self.dists[idx].density(x)

	def logDensity(self, x):
		idx = self.choice_function(x)
		return self.dists[idx].logDensity(x)

	def sampler(self, seed=None):
		return SelectSampler(self, seed)

	def estimator(self, pseudo_count=None):
		return SelectEstimator([d.estimator(pseudo_count=pseudo_count) for d in self.dists], self.choice_function)


class SelectSampler(object):
	def __init__(self, dist, seed=None):
		self.dist = dist
		self.rng  = RandomState(seed)
		self.distSamplers = [d.sampler(RandomState(self.rng.randint(maxint))) for d in dist.dists]

	def sample(self, size=None):

		if size is None:
			return tuple([d.sample(size=size) for d in self.distSamplers])
		else:
			return zip(*[d.sample(size=size) for d in self.distSamplers])


class SelectEstimatorAccumulator(object):
	def __init__(self, accumulators, choice_function):
		self.accumulators    = accumulators
		self.choice_function = choice_function
		self.weights         = [zero]*len(accumulators)
		self.count           = len(accumulators)

	def update(self, x, weight):
		#cf  = pickle.loads(self.choice_function)
		idx = self.choice_function(x)
		self.accumulators[idx].update(x, weight)
		self.weights[idx] += weight


	def combine(self, suff_stat):
		for i in range(0, self.count):
			self.weights[i] += suff_stat[i][0]
			self.accumulators[i].combine(suff_stat[i][1])

		return (self)

	def value(self):
		return zip(self.weights, [x.value() for x in self.accumulators])

class SelectEstimatorAccumulatorFactory(object):

	def __init__(self, estimators, choice_function):
		self.estimators = estimators
		self.choice_function = choice_function

	def make(self):
		return SelectEstimatorAccumulator([x.accumulatorFactory().make() for x in self.estimators], self.choice_function)

class SelectEstimator(object):
	def __init__(self, estimators, choice_function):
		self.estimators      = estimators
		self.choice_function = choice_function
		self.count           = len(estimators)

	def accumulatorFactory(self):
		return SelectEstimatorAccumulatorFactory(self.estimators, self.choice_function)

	def estimate(self, nobs, suff_stat):
		return (SelectDistribution([est.estimate(ss[0], ss[1]) for est, ss in zip(self.estimators, suff_stat)], self.choice_function))

