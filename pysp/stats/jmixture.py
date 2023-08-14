from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
import pysp.utils.vector as vec
from pysp.arithmetic import maxrandint


class JointMixtureDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, components1, components2, w1, w2, taus12, taus21):
		self.components1 = components1
		self.components2 = components2
		self.w1 = vec.make(w1)
		self.w2 = vec.make(w2)
		self.num_components1 = len(components1)
		self.num_components2 = len(components2)
		self.taus12 = np.reshape(taus12, (self.num_components1, self.num_components2))
		self.taus21 = np.reshape(taus21, (self.num_components1, self.num_components2))
		self.logW1 = log(self.w1)
		self.logW2 = log(self.w2)
		self.logTaus12 = log(self.taus12)
		self.logTaus21 = log(self.taus21)

	def __str__(self):
		s1 = ','.join([str(u) for u in self.components1])
		s2 = ','.join([str(u) for u in self.components2])
		s3 = ','.join(map(str, self.w1))
		s4 = ','.join(map(str, self.w2))
		s5 = ','.join(map(str,self.taus12.flatten()))
		s6 = ','.join(map(str, self.taus21.flatten()))

		return 'JointMixtureDistribution([%s], [%s], [%s], [%s], [%s], [%s])' % (s1, s2, s3, s4, s5, s6)

	def density(self, x):
		return exp(self.logDensity(x))

	def log_density(self, x):

		ll1 = np.zeros((1, self.num_components1))
		ll2 = np.zeros((1, self.num_components2))

		for i in range(self.num_components1):
			ll1[0,i] = self.components1[i].log_density(x[0]) + self.logW1[i]
		for i in range(self.num_components2):
			ll2[0,i] += self.components2[i].log_density(x[1])

		max1 = ll1.max()
		ll1 -= max1
		np.exp(ll1, out=ll1)

		max2 = ll2.max()
		ll2 -= max2
		np.exp(ll2, out=ll2)

		ll12 = np.dot(ll1, self.taus12)
		ll2 *= ll12

		rv = np.log(ll2.sum()) + max1 + max2

		return rv

	def seq_log_density(self, x):

		sz, enc_data1, enc_data2 = x
		ll_mat1 = np.zeros((sz, self.num_components1))
		ll_mat2 = np.zeros((sz, self.num_components2))

		for i in range(self.num_components1):
			ll_mat1[:, i] = self.components1[i].seq_log_density(enc_data1)
			ll_mat1[:, i] += self.logW1[i]

		for i in range(self.num_components2):
			ll_mat2[:, i] = self.components2[i].seq_log_density(enc_data2)

		ll_max1  = ll_mat1.max(axis = 1, keepdims=True)
		ll_mat1 -= ll_max1
		np.exp(ll_mat1, out=ll_mat1)

		ll_max2  = ll_mat2.max(axis = 1, keepdims=True)
		ll_mat2 -= ll_max2
		np.exp(ll_mat2, out=ll_mat2)

		ll_mat12  = np.dot(ll_mat1, self.taus12)
		ll_mat2  *= ll_mat12

		rv = np.log(ll_mat2.sum(axis=1)) + ll_max1[:,0] + ll_max2[:,0]

		return rv

	def seq_encode(self, x):
		rv0 = len(x)
		rv1 = self.components1[0].seq_encode([u[0] for u in x])
		rv2 = self.components2[0].seq_encode([u[1] for u in x])

		return rv0,rv1,rv2

	def posterior(self, x):
		return None

	def sampler(self, seed=None):
		return JointMixtureSampler(self, seed)


class JointMixtureSampler(object):
	def __init__(self, dist, seed=None):
		self.rng = RandomState(seed)
		self.dist = dist
		self.compSamplers1 = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in self.dist.components1]
		self.compSamplers2 = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in self.dist.components2]

	def sample(self, size=None):

		if size is None:
			compState1 = self.rng.choice(range(0, self.dist.num_components1), replace=True, p=self.dist.w1)
			f1 = self.compSamplers1[compState1].sample()
			compState2 = self.rng.choice(range(0, self.dist.num_components2), replace=True, p=self.dist.taus12[compState1,:])
			f2 = self.compSamplers2[compState2].sample()

			return (f1, f2)
		else:
			return [self.sample() for i in range(size)]



class JointMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, accumulators1, accumulators2):
		self.accumulators1 = accumulators1
		self.accumulators2 = accumulators2
		self.num_components1 = len(accumulators1)
		self.num_components2 = len(accumulators2)
		self.comp_counts1 = vec.zeros(self.num_components1)
		self.comp_counts2 = vec.zeros(self.num_components2)
		self.joint_counts = vec.zeros((self.num_components1, self.num_components2))


	def update(self, x, weight, estimate):
		pass

	def initialize(self, x, weight, rng):

		idx1 = rng.choice(self.num_components1)
		idx2 = rng.choice(self.num_components2)

		self.joint_counts[idx1,idx2] += 1.0

		for i in range(self.num_components1):
			w = 1.0 if i == idx1 else 0.0
			self.accumulators1[idx1].initialize(x[0], w, rng)
			self.comp_counts1[idx1] += w
		for i in range(self.num_components2):
			w = 1.0 if i == idx2 else 0.0
			self.accumulators2[idx2].initialize(x[1], w, rng)
			self.comp_counts2[idx2] += w

	def seq_update(self, x, weights, estimate):

		sz, enc_data1, enc_data2 = x
		ll_mat1 = np.zeros((sz, self.num_components1, 1))
		ll_mat2 = np.zeros((sz, 1, self.num_components2))
		log_w = estimate.logW1

		for i in range(estimate.num_components1):
			ll_mat1[:,i,0] = estimate.components1[i].seq_log_density(enc_data1)
			ll_mat1[:,i,0] += log_w[i]

		ll_max1 = ll_mat1.max(axis = 1, keepdims=True)
		ll_mat1 -= ll_max1
		np.exp(ll_mat1, out=ll_mat1)


		for i in range(estimate.num_components2):
			ll_mat2[:,0,i] = estimate.components2[i].seq_log_density(enc_data2)

		ll_max2 = ll_mat2.max(axis = 2, keepdims=True)
		ll_mat2 -= ll_max2
		np.exp(ll_mat2, out=ll_mat2)

		ll_joint = ll_mat1 * ll_mat2
		ll_joint *= estimate.taus12

		gamma_2 = np.sum(ll_joint, axis=1, keepdims=True)
		sf = np.sum(gamma_2, axis=2, keepdims=True)
		ww = np.reshape(weights, [-1,1,1])

		gamma_1 = np.sum(ll_joint, axis=2, keepdims=True)
		gamma_1 *= ww/sf
		gamma_2 *= ww/sf

		ll_joint *= ww/sf

		self.comp_counts1 += np.sum(gamma_1, axis=0).flatten()
		self.comp_counts2 += np.sum(gamma_2, axis=0).flatten()
		self.joint_counts += ll_joint.sum(axis=0)

		for i in range(self.num_components1):
			self.accumulators1[i].seq_update(enc_data1, gamma_1[:, i, 0], estimate.components1[i])


		for i in range(self.num_components2):
			self.accumulators2[i].seq_update(enc_data2, gamma_2[:, 0, i], estimate.components2[i])




	def combine(self, suff_stat):

		cc1,cc2,jc,s1,s2 = suff_stat

		self.joint_counts += jc
		self.comp_counts1 += cc1
		for i in range(self.num_components1):
			self.accumulators1[i].combine(s1[i])
		self.comp_counts2 += cc2
		for i in range(self.num_components2):
			self.accumulators2[i].combine(s2[i])

		return self

	def value(self):
		return self.comp_counts1, self.comp_counts2, self.joint_counts, tuple([u.value() for u in self.accumulators1]), tuple([u.value() for u in self.accumulators2])

	def from_value(self, x):

		cc1, cc2, jc, s1, s2 = x

		self.comp_counts1 = cc1
		self.comp_counts2 = cc2
		self.joint_counts = jc

		for i in range(self.num_components1):
			self.accumulators1[i].from_value(s1[i])
		for i in range(self.num_components2):
			self.accumulators2[i].from_value(s2[i])

		return self



	def key_merge(self, stats_dict):

		'''
		if self.weight_key is not None:
			if self.weight_key in stats_dict:
				stats_dict[self.weight_key] += self.comp_counts
			else:
				stats_dict[self.weight_key] = self.comp_counts

		if self.comp_key is not None:
			if self.comp_key in stats_dict:
				acc = stats_dict[self.comp_key]
				for i in range(len(acc)):
				 	acc[i] = acc[i].combine(self.accumulators[i].value())
			else:
				stats_dict[self.comp_key] = self.accumulators
		'''
		for u in self.accumulators1:
			u.key_merge(stats_dict)
		for u in self.accumulators2:
			u.key_merge(stats_dict)

	def key_replace(self, stats_dict):

		'''
		if self.weight_key is not None:
			if self.weight_key in stats_dict:
				self.comp_counts = stats_dict[self.weight_key]

		if self.comp_key is not None:
			if self.comp_key in stats_dict:
				acc = stats_dict[self.comp_key]
				self.accumulators = acc
		'''
		for u in self.accumulators1:
			u.key_replace(stats_dict)
		for u in self.accumulators2:
			u.key_replace(stats_dict)

class JointMixtureEstimatorAccumulatorFactory(object):
	def __init__(self, factories1, factories2):
		self.factories1 = factories1
		self.factories2 = factories2

	def make(self):
		f1 = [self.factories1[i].make() for i in range(len(self.factories1))]
		f2 = [self.factories2[i].make() for i in range(len(self.factories2))]
		return JointMixtureEstimatorAccumulator(f1, f2)


class JointMixtureEstimator(ParameterEstimator):

	def __init__(self, estimators1, estimators2, suff_stat=None, pseudo_count=None):
		self.num_components1 = len(estimators1)
		self.num_components2 = len(estimators2)
		self.estimators1 = estimators1
		self.estimators2 = estimators2
		self.pseudo_count = pseudo_count
		self.suff_stat = suff_stat

	def accumulatorFactory(self):
		est_factories1 = [u.accumulatorFactory() for u in self.estimators1]
		est_factories2 = [u.accumulatorFactory() for u in self.estimators2]
		return JointMixtureEstimatorAccumulatorFactory(est_factories1, est_factories2)

	def estimate(self, nobs, suff_stat):

		num_components1 = self.num_components1
		num_components2 = self.num_components2
		counts1, counts2, joint_counts, comp_suff_stats1, comp_suff_stats2 = suff_stat

		components1 = [self.estimators1[i].estimate(counts1[i], comp_suff_stats1[i]) for i in range(num_components1)]
		components2 = [self.estimators2[i].estimate(counts2[i], comp_suff_stats2[i]) for i in range(num_components2)]

		if self.pseudo_count is not None and self.suff_stat is None:
			p1 = self.pseudo_count[0] / float(self.num_components1)
			p2 = self.pseudo_count[1] / float(self.num_components2)
			p3 = self.pseudo_count[2] / float(self.num_components2 * self.num_components1)

			w1 = (counts1 + p1) / (counts1.sum() + p1)
			w2 = (counts2 + p2) / (counts2.sum() + p2)
			taus = joint_counts + p3

			taus12_sum = np.sum(taus, axis=1, keepdims=True)
			taus12_sum[taus12_sum == 0] = 1.0
			taus12 = taus / taus12_sum

			taus21_sum = np.sum(taus, axis=0, keepdims=True)
			taus21_sum[taus21_sum == 0] = 1.0
			taus21 = taus / taus21_sum

		else:
			w1 = counts1 / counts1.sum()
			w2 = counts2 / counts2.sum()
			taus = joint_counts

			taus12_sum = np.sum(taus, axis=1, keepdims=True)
			taus12_sum[taus12_sum == 0] = 1.0
			taus12 = taus / taus12_sum

			taus21_sum = np.sum(taus, axis=0, keepdims=True)
			taus21_sum[taus21_sum == 0] = 1.0
			taus21 = taus / taus21_sum

		return JointMixtureDistribution(components1, components2, w1, w2, taus12, taus21)

