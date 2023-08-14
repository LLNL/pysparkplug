from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
from scipy.sparse import dok_matrix
import collections
from pysp.arithmetic import maxrandint


class MarkovChainDistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, init_prob_map, transitionMap, len_dist=None, default_value=0.0, name=None):

		self.name              = name
		self.init_prob_map     = init_prob_map
		self.transitionMap     = transitionMap
		self.len_dist          = len_dist

		self.allVals           = set(init_prob_map.keys()).union(set([v for u in transitionMap.values() for v in u.keys()])).union(transitionMap.keys())
		self.loginit_prob_map           = {u[0] : -np.inf if u[1] == 0 else log(u[1]) for u in init_prob_map.items()}


		self.logTransitionMap  = dict((key, dict((u[0], log(u[1])) for u in transitionMap[key].items())) for key in transitionMap.keys())

		self.default_value  = default_value
		self.log_dv         = -np.inf if default_value == 0 else log(self.default_value)
		self.log_dtv        = -np.inf if default_value == 0 else (log(default_value) - np.log(len(self.allVals)+1))
		self.log1p_dv       = log(one + self.default_value)

		num_keys = len(self.allVals)

		keys     = list(self.allVals)
		sidx     = np.argsort(keys)
		keys     = [keys[i] for i in sidx]

		self.key_map     = {keys[i]: i+1 for i in range(num_keys)}
		self.inv_key_map = keys
		self.num_keys    = num_keys

		self.init_log_pvec  = np.zeros(num_keys + 1)
		self.trans_log_pvec = dok_matrix((num_keys + 1, num_keys + 1))

		for k1,v1 in init_prob_map.items():
			self.init_log_pvec[self.key_map.get(k1,0)] = -np.inf if v1 == 0 else np.log(v1)

		for k1,v1 in transitionMap.items():
			k1_idx = self.key_map.get(k1,0)
			for k2,v2 in v1.items():
				self.trans_log_pvec[k1_idx, self.key_map.get(k2,0)] = -np.inf if v2 == 0 else np.log(v2)

		self.init_log_pvec[0] = self.log_dv
		self.trans_log_pvec[:, 0] = self.log_dv
		self.trans_log_pvec[0, :] = self.log_dv - np.log(num_keys+1)

	def __str__(self):

		s1 = repr(dict(sorted(self.init_prob_map.items(), key=lambda u:u[0])))
		temp = sorted(self.transitionMap.items(), key=lambda u:u[0])
		s2 = repr(dict([(k,dict(sorted(v.items(), key=lambda u:u[0]))) for k,v in temp]))
		s3 = str(self.len_dist)
		s4 = repr(self.default_value)
		s5 = repr(self.name)

		return 'MarkovChainDistribution(%s, %s, len_dist=%s, default_value=%s, name=%s)' % (s1, s2, s3, s4, s5)

	def density(self, x):
		return np.exp(self.log_density(x))

	def log_density(self, x):

		if len(x) == 0:
			rv = 0.0
		else:
			rv = self.loginit_prob_map.get(x[0], self.log_dv) - self.log1p_dv

			for i in range(1, len(x)):
				if x[i - 1] in self.logTransitionMap:
					rv += self.logTransitionMap[x[i - 1]].get(x[i], self.log_dv) - self.log1p_dv
				else:
					rv += self.log_dtv - self.log1p_dv

		if self.len_dist is not None:
			rv += self.len_dist.log_density(len(x))

		return rv

	def seq_encode(self, x):

		init_entries = []
		pair_entries = []
		entries_idx0 = []
		entries_idx1 = []
		obs_cnt = []
		key_map = dict()

		for i in range(len(x)):
			entry = x[i]
			obs_cnt.append(len(entry))

			if len(entry) == 0:
				continue

			if entry[0] not in key_map:
				key_map[entry[0]] = len(key_map)

			prev_idx = key_map[entry[0]]
			init_entries.append(prev_idx)
			entries_idx0.append(i)

			for j in range(1, len(entry)):

				if entry[j] not in key_map:
					key_map[entry[j]] = len(key_map)
				next_idx = key_map[entry[j]]

				pair_entries.append([prev_idx, next_idx])
				entries_idx1.append(i)
				prev_idx = next_idx

		obs_cnt      = np.asarray(obs_cnt)
		init_entries = np.asarray(init_entries)
		pair_entries = np.asarray(pair_entries)
		entries_idx0 = np.asarray(entries_idx0)
		entries_idx1 = np.asarray(entries_idx1)

		inv_key_map = [None]*len(key_map)
		for k,v in key_map.items():
			inv_key_map[v] = k
		inv_key_map = np.asarray(inv_key_map)

		len_enc = None
		if self.len_dist is not None:
			len_enc = self.len_dist.seq_encode(obs_cnt)

		return len(x), entries_idx0, entries_idx1, init_entries, pair_entries[:,0], pair_entries[:,1], inv_key_map, len_enc

	def seq_log_density(self, x):

		sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x

		loc_key_map = np.asarray([self.key_map.get(u,0) for u in inv_key_map])

		temp = self.trans_log_pvec[loc_key_map[prev_x], loc_key_map[next_x]].toarray().flatten() - self.log1p_dv
		rv = np.bincount(idx1, weights=temp, minlength=sz)
		rv[idx0] += self.init_log_pvec[loc_key_map[init_x]] - self.log1p_dv

		if self.len_dist is not None and len_enc is not None:
			rv += self.len_dist.seq_log_density(len_enc)

		return rv

	def sampler(self, seed=None):
		return MarkovChainSampler(self, seed)

	def estimator(self, pseudo_count=None):

		if self.len_dist is None:
			len_est = None
		else:
			len_est = self.len_dist.estimator(pseudo_count=pseudo_count)

		if pseudo_count is None:
			return MarkovChainEstimator(len_estimator=len_est, name=self.name)
		else:
			return MarkovChainEstimator(pseudo_count=pseudo_count, len_estimator=len_est, name=self.name)

class MarkovChainSampler(object):

	def __init__(self, dist, seed):

		self.rng = RandomState(seed)


		loc_trans = list(dist.init_prob_map.items())
		loc_probs = [v[1] for v in loc_trans]
		loc_keys  = [v[0] for v in loc_trans]

		self.init_prob = (loc_keys, loc_probs)

		self.trans_prob = dict()
		for k,v in dist.transitionMap.items():
			loc_trans = list(v.items())
			loc_probs = [v[1] for v in loc_trans]
			loc_keys = [v[0] for v in loc_trans]	
			self.trans_prob[k] = (loc_keys, loc_probs)

		#self.keys       = dist.inv_key_map
		#self.num_keys   = len(self.keys)

		if dist.len_dist is not None:
			self.len_sampler = dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
		else:
			self.len_sampler = None

	def sample(self, size=None):

		if size is not None:
			return [self.sample() for i in range(size)]

		else:
			cnt = self.len_sampler.sample()
			rv  = [None]*cnt

			if cnt >= 1:
				rv[0] = self.rng.choice(self.init_prob[0], p=self.init_prob[1])

			for i in range(1,cnt):
				curr_k, curr_p = self.trans_prob[rv[i-1]]
				rv[i] = self.rng.choice(curr_k, p=curr_p)

			return rv


	def sample_seq(self, size=None, v0=None):


		if size is not None:

			rv = [None]*size

			prev_val = v0

			if size > 0 and prev_val is None:
				rv[0] = self.rng.choice(self.init_prob[0], p=self.init_prob[1])
				prev_val = rv[0]

			for i in range(1, size):

				if prev_val not in self.trans_prob:
					break

				levels, probs = self.trans_prob[prev_val]
				rv[i] = self.rng.choice(levels, p=probs)
				prev_val = rv[i]

			return rv

		else:

			prev_val = v0

			if prev_val is None:
				rv = self.rng.choice(self.init_prob[0], p=self.init_prob[1])
			else:
				levels, probs = self.trans_prob[prev_val]
				rv = self.rng.choice(levels, p=probs)

			return rv


class MarkovChainEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, len_accumulator):
		self.init_count_map = dict()
		self.trans_count_map = dict()
		self.len_accumulator = len_accumulator

	def update(self, x, weight, estimate):

		if x is not None and self.len_accumulator is not None:
			l_est = None
			if estimate is not None and estimate.len_dist is not None:
				l_est = estimate.len_dist

			self.len_accumulator.update(len(x), weight, l_est)

		if x is not None and len(x) != 0:
			x0 = x[0]
			self.init_count_map[x0] = self.init_count_map.get(x0, zero) + weight

			for u in x[1:]:
				if x0 not in self.trans_count_map:
					self.trans_count_map[x0] = dict()

				self.trans_count_map[x0][u] = self.trans_count_map[x0].get(u, zero) + weight
				x0 = u

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)

	def seq_update(self, x, weights, estimate):

		sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = x

		key_sz = len(inv_key_map)

		init_count  = np.bincount(init_x, weights=weights[idx0])

		for i in range(len(init_count)):
			v  = init_count[i]
			if v != 0:
				self.init_count_map[inv_key_map[i]] = self.init_count_map.get(inv_key_map[i], 0.0) + v

		'''
		trans_count = np.bincount(prev_x*key_sz + next_x, weights=weights[idx1], minlength=key_sz*key_sz)
		trans_count = np.reshape(trans_count, (key_sz, key_sz))
		trans_count_nz1, trans_count_nz2 = np.nonzero(trans_count)

		for i in range(len(trans_count_nz1)):
			j1 = trans_count_nz1[i]
			j2 = trans_count_nz2[i]
			k1 = inv_key_map[j1]
			k2 = inv_key_map[j2]

			if k1 not in self.trans_count_map:
				self.trans_count_map[k1] = {k2 : trans_count[j1,j2]}
			else:
				m = self.trans_count_map[k1]
				m[k2] = m.get(k2,0.0) + trans_count[j1,j2]
		'''

		# ------------- slow and sparse...

		for i in range(len(prev_x)):
			k1 = inv_key_map[prev_x[i]]
			k2 = inv_key_map[next_x[i]]
			ww = weights[idx1[i]]

			if k1 not in self.trans_count_map:
				self.trans_count_map[k1] = {k2: ww}
			else:
				m = self.trans_count_map[k1]
				m[k2] = m.get(k2,0.0) + ww

		# ------------- slow and sparse...

		if self.len_accumulator is not None:
			l_est = None
			if estimate is not None and estimate.len_dist is not None:
				l_est = estimate.len_dist

			self.len_accumulator.seq_update(len_enc, weights, l_est)

	def combine(self, suff_stat):
		for item in suff_stat[0].items():
			self.init_count_map[item[0]] = self.init_count_map.get(item[0], 0.0) + item[1]

		for item in suff_stat[1].items():
			if item[0] not in self.trans_count_map:
				self.trans_count_map[item[0]] = dict()

			itemMap = self.trans_count_map[item[0]]
			for elem in item[1].items():
				itemMap[elem[0]] = itemMap.get(elem[0], 0.0) + elem[1]

		if self.len_accumulator is not None and suff_stat[2] is not None:
			self.len_accumulator = self.len_accumulator.combine(suff_stat[2])
			
		return self

	def value(self):
		if self.len_accumulator is not None:
			return self.init_count_map, self.trans_count_map, self.len_accumulator.value()
		else:
			return self.init_count_map, self.trans_count_map, None

	def from_value(self, x):
		self.init_count_map  = x[0]
		self.trans_count_map = x[1]

		if self.len_accumulator is not None and x[2] is not None:
			self.len_accumulator = self.len_accumulator.from_value(x[2])

		return self





class MarkovChainEstimator(ParameterEstimator):
	def __init__(self, pseudo_count=None, levels=None, len_estimator=None, name=None):

		self.name = name
		self.pseudo_count = pseudo_count
		self.levels = levels
		self.len_estimator = len_estimator


	def accumulatorFactory(self):
		if self.len_estimator is not None:
			len_acc = self.len_estimator.accumulatorFactory()
			obj = type('', (object,), {'make': lambda self: MarkovChainEstimatorAccumulator(len_accumulator=len_acc.make())})()
		else:
			obj = type('', (object,),{'make': lambda self: MarkovChainEstimatorAccumulator(len_accumulator=None)})()
		return obj

	def estimate(self, nobs, suff_stat):

		if self.pseudo_count is not None:
			return self.estimate1(nobs, suff_stat)
		else:
			return self.estimate0(nobs, suff_stat)

	def estimate0(self, nobs, suff_stat):

		temp_sum = sum(suff_stat[0].values())
		init_prob_map = {k: v / temp_sum for k, v in suff_stat[0].items()}

		trans_map = dict()

		for key, tmap in suff_stat[1].items():
			temp_sum = sum(tmap.values())
			if temp_sum > 0:
				trans_map[key] = {k: v / temp_sum for k, v in tmap.items()}


		if self.len_estimator is not None and suff_stat[2] is not None:
			len_dist = self.len_estimator.estimate(nobs, suff_stat[2])
		else:
			len_dist = None

		return MarkovChainDistribution(init_prob_map, trans_map, len_dist=len_dist, name=self.name)


	def estimate1(self, nobs, suff_stat):

		trans_map = dict()
		init_prob_map = dict()
		def_val = 0.0

		all_keys = set(suff_stat[0].keys())
		for u in suff_stat[1].values():
			all_keys.update(u.keys())
		if self.levels is not None:
			all_keys.update(self.levels)

		temp_sum = sum(suff_stat[0].values())
		pcnt0 = self.pseudo_count if self.pseudo_count is not None else 0.0
		pcnt1 = pcnt0 / len(all_keys)

		if (temp_sum + pcnt0) > 0:
			init_prob_map = {k: (suff_stat[0].get(k,0.0) + pcnt1) / (temp_sum + pcnt0) for k in all_keys}

		asum = temp_sum
		for key, tmap in suff_stat[1].items():
			temp_sum = sum(tmap.values())
			asum += temp_sum
			if (temp_sum + pcnt0) > 0:
				trans_map[key] = {k: (tmap.get(k, 0.0) + pcnt1) / (temp_sum + pcnt0) for k in all_keys}
				#trans_map[key] = {k: v / temp_sum for k, v in tmap.items()}


		if self.len_estimator is not None and suff_stat[2] is not None:
			len_dist = self.len_estimator.estimate(nobs, suff_stat[2])
		else:
			len_dist = None

		if asum > 0:
			def_val = self.pseudo_count/asum

		return MarkovChainDistribution(init_prob_map, trans_map, len_dist=len_dist, default_value=def_val, name=self.name)

	def estimate1_old(self, nobs, suff_stat):

		all_keys = set(suff_stat[0].keys())
		for s in suff_stat[1].values():
			all_keys.update(s.keys())

		if self.levels is not None:
			all_keys.update(self.levels)

		tempSum = sum(suff_stat[0].values())
		nobs_loc = tempSum + self.pseudo_count
		pcnt = self.pseudo_count
		pcnt1 = self.pseudo_count / len(all_keys)

		#init_prob_map = dict((u[0], u[1] / tempSum) for u in suff_stat[0].items())
		init_prob_map = {kk : (suff_stat[0].get(kk,0.0) + pcnt1) / (tempSum + pcnt) for kk in all_keys}
		transMap = dict()
		totalSum = zero


		for key in all_keys:
			tmap  = suff_stat[1].get(key, dict())

			tempSum = sum(tmap.values()) + pcnt
			if tempSum == zero:
				tempSum = one

			#2.7transMap[key] = {k: (v + self.pseudo_count)/ tempSum for k, v in tmap.iteritems()}
			transMap[key] = {kk : (tmap.get(kk,0.0) + pcnt1) / tempSum for kk in all_keys}
			totalSum += tempSum

		totalSum = totalSum/max(len(suff_stat[1]), 1)

		if totalSum == 0:
			totalSum = 1.0/nobs_loc


		if self.len_estimator is not None and suff_stat[2] is not None:
			len_dist = self.len_estimator.estimate(nobs, suff_stat[2])
		else:
			len_dist = None

		return MarkovChainDistribution(init_prob_map, transMap, default_value=self.pseudo_count/nobs_loc, len_dist=len_dist, name=self.name)

	def estimate2(self, nobs, suff_stat):

		tempSum = sum(suff_stat[0].values())
		#2.7init_prob_map = {k: v / tempSum for k, v in suff_stat[0].iteritems()}
		init_prob_map = dict((u[0], u[1] / tempSum) for u in suff_stat[0].items())
		transMap = dict()
		dMap = dict()

		for key, tmap in suff_stat[1].iteritems():
			tempSum = sum(tmap.values())
			#2.7transMap[key] = {k: v / tempSum for k, v in tmap.iteritems()}
			transMap[key] = dict((u[0], u[1] / tempSum) for u in tmap.items())

			for key2, val2 in tmap.iteritems():
				dMap[key2] = dMap.get(key2, zero) + val2

		dMapSum = sum(dMap.values())
		#2.7dMap = {k: v/dMapSum for k,v in dMap.iteritems()}
		dMap = dict((u[0], u[1] / dMapSum) for u in dMap.items())

		return MarkovChainDistribution(dMap, transMap)

