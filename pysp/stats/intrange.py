from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np

class IntegerCategoricalDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, minVal, pVec, name=None):
		self.pVec     = np.asarray(pVec)
		self.minVal   = minVal
		self.maxVal   = minVal + self.pVec.shape[0] - 1
		self.logPVec  = np.log(self.pVec)
		self.num_vals = self.pVec.shape[0]
		self.name     = name

	def __str__(self):
		s1 = str(self.minVal)
		s2 = repr(list(self.pVec))
		s3 = repr(self.name)
		return 'IntegerCategoricalDistribution(%s, %s, name=%s)' % (s1, s2, s3)

	def density(self, x):
		return zero if x < self.minVal or x > self.maxVal else self.pVec[x-self.minVal]

	def log_density(self, x):
		#return self.logPVec[x-self.minVal]
		return -inf if (x < self.minVal or x > self.maxVal) else self.logPVec[x-self.minVal]

	def seq_log_density(self, x):

		v  = x - self.minVal
		u  = np.bitwise_and(v >= 0, v < self.num_vals)
		rv = np.zeros(len(x))
		rv.fill(-np.inf)
		rv[u] = self.logPVec[v[u]]

		return rv

	def seq_encode(self, x):
		return np.asarray(x, dtype=int)

	def weight(self, x):
		return one

	def sampler(self, seed=None):
		return IntegerCategoricalSampler(self, seed)

	def estimator(self, pseudo_count=None):

		if pseudo_count is None:
			return IntegerCategoricalEstimator(name=self.name)
		else:
			return IntegerCategoricalEstimator(pseudo_count=pseudo_count, suff_stat=(self.minVal, self.pVec), name=self.name)


class IntegerCategoricalSampler(object):

	def __init__(self, dist, seed=None):
		self.rng  = RandomState(seed)
		self.dist = dist

	def sample(self, size=None):

		if size is None:
			return self.rng.choice(range(self.dist.minVal, self.dist.maxVal+1), p=self.dist.pVec)
		else:
			return list(self.rng.choice(range(self.dist.minVal, self.dist.maxVal + 1), p=self.dist.pVec, size=size))


class IntegerCategoricalAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, minVal=None, maxVal=None, keys=None):

		self.minVal = minVal
		self.maxVal = maxVal

		if minVal is not None and maxVal is not None:
			self.countVec = vec.zeros(maxVal-minVal+1)
		else:
			self.countVec = None

		self.key = keys

	def update(self, x, weight, estimate):

		if self.countVec is None:
			self.minVal   = x
			self.maxVal   = x
			self.countVec = np.asarray([weight])

		elif self.maxVal < x:
			tempVec = self.countVec
			self.maxVal   = x
			self.countVec = np.zeros(self.maxVal - self.minVal + 1)
			self.countVec[:len(tempVec)] = tempVec
			self.countVec[x-self.minVal] += weight
		elif self.minVal > x:
			tempVec  = self.countVec
			tempDiff = self.minVal - x
			self.minVal   = x
			self.countVec = np.zeros(self.maxVal - self.minVal + 1)
			self.countVec[tempDiff:] = tempVec
			self.countVec[x-self.minVal] += weight
		else:
			self.countVec[x-self.minVal] += weight


	def seq_update(self, x, weights, estimate):

		min_x = x.min()
		max_x = x.max()

		loc_cnt = np.bincount(x-min_x, weights=weights)

		if self.countVec is None:
			self.countVec = np.zeros(max_x-min_x+1)
			self.minVal = min_x
			self.maxVal = max_x

		if self.minVal > min_x or self.maxVal < max_x:
			prev_min    = self.minVal
			self.minVal = min(min_x, self.minVal)
			self.maxVal = max(max_x, self.maxVal)
			temp        = self.countVec
			prev_diff   = prev_min - self.minVal
			self.countVec = np.zeros(self.maxVal - self.minVal + 1)
			self.countVec[prev_diff:(prev_diff + len(temp))] = temp

		min_diff = min_x - self.minVal
		self.countVec[min_diff:(min_diff+len(loc_cnt))] += loc_cnt

	def combine(self, suff_stat):

		if self.countVec is None and suff_stat[1] is not None:
			self.minVal   = suff_stat[0]
			self.maxVal   = suff_stat[0] + len(suff_stat[1]) - 1
			self.countVec = suff_stat[1]

		elif self.countVec is not None and suff_stat[1] is not None:

			if self.minVal == suff_stat[0] and len(self.countVec) == len(suff_stat[1]):
				self.countVec += suff_stat[1]

			else:
				minVal = min(self.minVal, suff_stat[0])
				maxVal = max(self.maxVal, suff_stat[0] + len(suff_stat[1]) - 1)

				countVec = vec.zeros(maxVal-minVal+1)

				i0 = self.minVal - minVal
				i1 = self.maxVal - minVal + 1
				countVec[i0:i1] = self.countVec

				i0 = suff_stat[0] - minVal
				i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - minVal + 1
				countVec[i0:i1] += suff_stat[1]

				self.minVal   = minVal
				self.maxVal   = maxVal
				self.countVec = countVec

		return self

	def value(self):
		return self.minVal, self.countVec

	def from_value(self, x):
		self.minVal   = x[0]
		self.maxVal   = x[0] + len(x[1]) - 1
		self.countVec = x[1]


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


class IntegerCategoricalAccumulatorFactory(object):

	def __init__(self, minVal, maxVal, keys):
		self.minVal = minVal
		self.maxVal = maxVal
		self.keys = keys

	def make(self):
		return IntegerCategoricalAccumulator(self.minVal, self.maxVal, self.keys)


class IntegerCategoricalEstimator(object):
	def __init__(self, minVal=None, maxVal=None, pseudo_count=None, suff_stat=None, name=None, keys=None):

		self.pseudo_count = pseudo_count
		self.minVal      = minVal
		self.maxVal      = maxVal
		self.suff_stat    = suff_stat
		self.keys        = keys
		self.name        = name

	def accumulatorFactory(self):

		minVal = None
		maxVal = None

		if self.suff_stat is not None:
			minVal = self.suff_stat[0]
			maxVal = minVal + len(self.suff_stat[1]) - 1
		elif self.minVal is not None and self.maxVal is not None:
			minVal = self.minVal
			maxVal = self.maxVal

		return IntegerCategoricalAccumulatorFactory(minVal, maxVal, self.keys)

	def estimate(self, nobs, suff_stat):

		if self.pseudo_count is not None and self.suff_stat is None:

			pseudo_countPerLevel = self.pseudo_count / float(len(suff_stat[1]))
			adjustedNobs        = suff_stat[1].sum() + self.pseudo_count

			return IntegerCategoricalDistribution(suff_stat[0], (suff_stat[1]+pseudo_countPerLevel)/adjustedNobs, name=self.name)

		elif self.pseudo_count is not None and self.minVal is not None and self.maxVal is not None:

			minVal = min(self.minVal, suff_stat[0])
			maxVal = max(self.maxVal, suff_stat[0] + len(suff_stat[1]) - 1)

			countVec = vec.zeros(maxVal - minVal + 1)

			i0 = suff_stat[0] - minVal
			i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - minVal + 1
			countVec[i0:i1] += suff_stat[1]

			pseudo_countPerLevel = self.pseudo_count / float(len(countVec))
			adjustedNobs        = suff_stat[1].sum() + self.pseudo_count

			return IntegerCategoricalDistribution(minVal, (countVec+pseudo_countPerLevel)/adjustedNobs, name=self.name)

		elif self.pseudo_count is not None and self.suff_stat is not None:

			sMaxVal = self.suff_stat[0] + len(self.suff_stat[1]) - 1
			sMinVal = self.suff_stat[0]

			minVal = min(sMinVal, suff_stat[0])
			maxVal = max(sMaxVal, suff_stat[0] + len(suff_stat[1]) - 1)

			countVec = vec.zeros(maxVal - minVal + 1)

			i0 = sMinVal - minVal
			i1 = sMaxVal - minVal + 1
			countVec[i0:i1] = self.suff_stat[1]*self.pseudo_count

			i0 = suff_stat[0] - minVal
			i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - minVal + 1
			countVec[i0:i1] += suff_stat[1]

			return IntegerCategoricalDistribution(minVal, countVec/(countVec.sum()), name=self.name)


		else:
			return IntegerCategoricalDistribution(suff_stat[0], suff_stat[1]/(suff_stat[1].sum()), name=self.name)

