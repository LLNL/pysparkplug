import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, gammaln
import sys
import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.dirichlet import DirichletDistribution
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
from pysp.utils.special import digammainv
from pysp.arithmetic import maxrandint


#import pysp.c_ext

class LDADistribution(SequenceEncodableProbabilityDistribution):

	def __init__(self, topics, alpha, len_dist=None, gamma_threshold=1.0e-8):
		self.topics       = topics
		self.nTopics      = len(topics)
		self.alpha        = np.asarray(alpha)
		self.len_dist     = len_dist
		self.gamma_threshold = gamma_threshold

	def __str__(self):
		return 'LDADistribution([%s], [%s])'%(','.join([str(u) for u in self.topics]), ','.join(map(str,self.alpha)))

	def log_density(self, x):
		return self.seq_log_density(self.seq_encode([x]))[0]

	def seq_log_density(self, x):

		num_topics = self.nTopics
		alpha = self.alpha
		num_documents, idx, counts, _, enc_data = x

		idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
		idx_full *= num_topics
		idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

		log_density_gamma, document_gammas, per_topic_log_densities = seq_posterior(self, x)

		# This block keeps the gammas positive
		log_density_gamma[np.bitwise_or(np.isnan(log_density_gamma), np.isinf(log_density_gamma))] = sys.float_info.min
		log_density_gamma[log_density_gamma <= 0] = sys.float_info.min
		document_gammas[np.bitwise_or(np.isnan(document_gammas), np.isinf(document_gammas))] = sys.float_info.min

		elob0 = digamma(document_gammas) - digamma(np.sum(document_gammas, axis=1, keepdims=True))
		elob1 = elob0[idx, :]
		elob2 = log_density_gamma * (elob1 + per_topic_log_densities - np.log(log_density_gamma))
		elob3 = np.sum(elob0 * ((alpha - 1.0) - (document_gammas - 1.0)), axis=1)
		elob4 = np.bincount(idx_full.flat, weights=elob2.flat)
		elob5 = np.sum(np.reshape(elob4, (-1, num_topics)), axis=1)
		elob6 = np.sum(gammaln(document_gammas), axis=1) - gammaln(document_gammas.sum(axis=1))
		elob7 = gammaln(alpha.sum()) - gammaln(alpha).sum()

		elob = elob3 + elob5 + elob6 + elob7

		return elob


	def seq_encode(self, x):

		num_documents = len(x)

		tx   = []
		ctx  = []
		nx   = []
		tidx = []
		for i in range(len(x)):
			nx.append(len(x[i]))
			for j in range(len(x[i])):
				tidx.append(i)
				tx.append(x[i][j][0])
				ctx.append(x[i][j][1])

		idx      = np.asarray(tidx)
		counts   = np.asarray(ctx)
		gammas   = None
		enc_data = self.topics[0].seq_encode(tx)

		return num_documents, idx, counts, gammas, enc_data


	def seq_component_log_density(self, x):

		num_topics = self.nTopics
		alpha = self.alpha
		num_documents, idx, counts, _, enc_data = x

		ll_mat = np.zeros((len(idx), self.nTopics))
		ll_mat.fill(-np.inf)

		rv = np.zeros((num_documents, self.nTopics))
		rv.fill(-np.inf)

		for i in range(num_topics):
			ll_mat[:, i]  = self.topics[i].seq_log_density(enc_data)
			rv[:, i] = np.bincount(idx, weights=ll_mat[:,i]*counts, minlength=num_documents)


		return rv

	def seq_posterior(self, x):
		num_topics = self.nTopics
		alpha = self.alpha
		num_documents, idx, counts, _, enc_data = x

		log_density_gamma, document_gammas, per_topic_log_densities = seq_posterior(self, x)

		document_gammas /= document_gammas.sum(axis=1, keepdims=True)

		return document_gammas


	def sampler(self, seed=None):
		return LDASampler(self, seed)







class LDASampler(object):

	def __init__(self, dist, seed=None):
		self.rng              = RandomState(seed)
		self.dist             = dist
		self.nTopics          = dist.nTopics
		self.compSamplers     = [self.dist.topics[i].sampler(seed=self.rng.randint(0, maxrandint)) for i in range(dist.nTopics)]
		self.dirichletSampler = DirichletDistribution(dist.alpha).sampler(self.rng.randint(0, maxrandint))
		self.len_dist = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

	def sample(self, size=None):

		if size is None:
			n = self.len_dist.sample()
			nTopics   = self.nTopics
			weights   = self.dirichletSampler.sample()
			#topics    = self.rng.choice(range(0, nTopics), size=n, replace=True, p=weights)
			#rv        = [None]*n
			#for i in range(n):
			#	rv[i] = self.compSamplers[topics[i]].sample()
			#
			topic_counts = self.rng.multinomial(n, pvals=weights)
			topics = []
			rv = []
			for i in np.flatnonzero(topic_counts):
				topics.extend([i]*topic_counts[i])
				rv.extend(self.compSamplers[i].sample(size=topic_counts[i]))

			return rv

		else:
			return [self.sample() for i in range(size)]



class LDAEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

	def __init__(self, accumulators, keys=(None, None), prev_alpha=None):
		self.accumulators = accumulators
		self.num_topics   = len(accumulators)
		self.sum_of_logs  = np.zeros(self.num_topics)
		self.doc_counts   = 0.0
		self.topic_counts = np.zeros(self.num_topics)
		self.prev_alpha   = prev_alpha

		self.alpha_key    = keys[0]
		self.topics_key   = keys[1]

	def update(self, x, weight, estimate):
		pass

	def initialize(self, x, weight, rng):

		if self.prev_alpha is None:
			self.prev_alpha = np.ones(self.num_topics)

		theta = rng.dirichlet(self.prev_alpha)

		idx_list = rng.choice(self.num_topics, size=len(x), replace=True, p=theta)

		self.sum_of_logs += np.log(theta)
		self.doc_counts  += weight

		for i in range(len(x)):
			idx = idx_list[i]
			ww_v = -np.log(rng.rand(self.num_topics))
			ww_v[idx] += 1
			ww_v *= weight*x[i][1]/ww_v.sum()
			for j in range(self.num_topics):
				#w = weight*x[i][1] if idx == j else 0.0
				w = ww_v[j]
				self.topic_counts[j] += w
				self.accumulators[j].initialize(x[i][0], w, rng)

	def seq_update(self, x, weights, estimate):

		num_documents, idx, counts, old_gammas, enc_data = x
		log_density_gamma, final_gammas, per_topic_log_densities = seq_posterior(estimate, x)


		for i in range(self.num_topics):
			self.accumulators[i].seq_update(enc_data, log_density_gamma[:, i]*weights[idx]*counts, estimate.topics[i])

		mlpf = digamma(final_gammas) - digamma(np.sum(final_gammas, axis=1, keepdims=True))

		self.sum_of_logs  += np.dot(weights, mlpf)
		self.doc_counts   += weights.sum()
		self.topic_counts += np.sum(log_density_gamma, axis=0)
		self.prev_alpha    = estimate.alpha

		#return num_documents, idx, counts, final_gammas, enc_data


	def combine(self, suff_stat):

		prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats = suff_stat

		if self.prev_alpha is None:
			self.prev_alpha = prev_alpha

		self.sum_of_logs  += sum_of_logs
		self.doc_counts   += doc_counts
		self.topic_counts += topic_counts

		for i in range(self.num_topics):
			self.accumulators[i].combine(topic_suff_stats[i])

		return self

	def value(self):
		return self.prev_alpha, self.sum_of_logs, self.doc_counts, self.topic_counts, [u.value() for u in self.accumulators]

	def from_value(self, x):

		prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats = x

		self.prev_alpha   = prev_alpha
		self.sum_of_logs  = sum_of_logs
		self.doc_counts   = doc_counts
		self.topic_counts = topic_counts
		self.accumulators = [self.accumulators[i].from_value(topic_suff_stats[i]) for i in range(len(self.num_topics))]


		return self

	def key_merge(self, stats_dict):

		if self.alpha_key is not None:
			if self.alpha_key in stats_dict:

				p_sol, p_doc, p_pa = stats_dict[self.alpha_key]

				prev_alpha = self.prev_alpha if self.prev_alpha is not None else p_pa
				stats_dict[self.alpha_key] = (self.sum_of_logs + p_sol, self.doc_counts + p_doc, prev_alpha)

			else:
				stats_dict[self.alpha_key] = (self.sum_of_logs, self.doc_counts, self.prev_alpha)

		if self.topics_key is not None:
			if self.topics_key in stats_dict:
				acc = stats_dict[self.topics_key]
				for i in range(len(acc)):
					acc[i] = acc[i].combine(self.accumulators[i].value())
			else:
				stats_dict[self.topics_key] = self.accumulators

		for u in self.accumulators:
			u.key_merge(stats_dict)

	def key_replace(self, stats_dict):


		if self.alpha_key is not None:
			if self.alpha_key in stats_dict:
				p_sol, p_doc, p_pa = stats_dict[self.alpha_key]
				self.prev_alpha = p_pa
				self.sum_of_logs = p_sol
				self.doc_counts = p_doc

		if self.topics_key is not None:
			if self.topics_key in stats_dict:
				acc = stats_dict[self.topics_key]
				self.accumulators = acc

		for u in self.accumulators:
			u.key_replace(stats_dict)


class LDAEstimatorAccumulatorFactory(object):
	def __init__(self, factories, dim, keys, prev_alpha):
		self.factories = factories
		self.dim = dim
		self.keys = keys
		self.prev_alpha = prev_alpha

	def make(self):
		return LDAEstimatorAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys, self.prev_alpha)


class LDAEstimator(ParameterEstimator):

	def __init__(self, estimators, suff_stat=None, pseudo_count=None, keys=(None, None), fixed_alpha=None, gamma_threshold=1.0e-8, alpha_threshold=1.0e-8):

		self.num_topics  = len(estimators)
		self.estimators  = estimators
		self.pseudo_count = pseudo_count
		self.suff_stat   = suff_stat
		self.keys        = keys
		self.gamma_threshold = gamma_threshold
		self.alpha_threshold = alpha_threshold
		self.fixed_alpha = fixed_alpha

	def accumulatorFactory(self):
		est_factories = [u.accumulatorFactory() for u in self.estimators]
		return LDAEstimatorAccumulatorFactory(est_factories, self.num_topics, self.keys, self.fixed_alpha)

	def estimate(self, nobs, suff_stat):

		prev_alpha, sum_of_logs, doc_counts, topic_counts, topic_suff_stats = suff_stat


		num_topics = self.num_topics
		topics     = [self.estimators[i].estimate(topic_counts[i], topic_suff_stats[i]) for i in range(num_topics)]

		if doc_counts == 0:
			sys.stderr.write('Warning: LDA Estimation performed with zero documents.\n')
			LDADistribution(topics, prev_alpha, gamma_threshold=self.gamma_threshold)

		if self.fixed_alpha is None:

			if self.pseudo_count is not None:
				mean_of_logs = (sum_of_logs + np.log(self.pseudo_count[1]))/(doc_counts + self.pseudo_count[0])

			#new_alpha, _ = find_alpha(prev_alpha, sum_of_logs/doc_counts, gamma_threshold*np.sqrt(float(doc_counts)))
			new_alpha, _ = updateAlpha(prev_alpha, sum_of_logs/doc_counts, self.alpha_threshold)
		else:
			new_alpha = np.asarray(self.fixed_alpha).copy()

		return LDADistribution(topics, new_alpha, gamma_threshold=self.gamma_threshold)







def updateAlpha(currentAlpha, meanLogP, alphaThreshold):

	alpha = currentAlpha.copy()
	asum = alpha.sum()
	res = np.inf
	its_cnt = 0
	while res > alphaThreshold:
		dasum = digamma(asum)
		oldAlpha = alpha
		alpha = digammainv(meanLogP + dasum)
		asum = alpha.sum()
		res = np.abs(alpha - oldAlpha).sum() /asum
		its_cnt += 1

	return alpha, its_cnt


def mpe_update(X, y, min_size=2):

	if(X is None):
		X = np.reshape(y, (1,-1))
		return X, y
	elif(X.shape[0] < min_size):
		X = np.concatenate((X, np.reshape(y, (1, -1))), axis=0)
		return X, y


	dy = y-X[-1,:]
	U  = (X[1:,:]-X[:-1,:]).T
	X2 = X[1:,:].T
	c = np.dot(np.linalg.pinv(U), dy)
	c *= -1
	s = (np.dot(X2, c) + y)/(c.sum() + 1)

	X = np.concatenate((X, np.reshape(y, (1,-1))), axis=0)

	return X, s


def mpe(x0, f, eps):

	x1 = f(x0)
	x2 = f(x1)
	x3 = f(x2)
	X = np.asarray([x0, x1, x2, x3])
	s0 = x3
	s = s0
	res = np.abs(x3 - x2).sum()
	its_cnt = 2

	while res > eps:
		y = f(X[-1, :])
		dy = y-X[-1,:]
		U  = (X[1:,:]-X[:-1,:]).T
		X2 = X[1:,:].T
		c = np.dot(np.linalg.pinv(U), dy)
		c *= -1
		s = (np.dot(X2, c) + y)/(c.sum() + 1)

		res = np.abs(s-s0).sum()
		s0 = s
		X = np.concatenate((X, np.reshape(y, (1,-1))), axis=0)
		its_cnt += 1

	return s, its_cnt

def alpha_seq_lambda(meanLogP):

	def next_alpha(currentAlpha):
		return digammainv(meanLogP + digamma(currentAlpha.sum()))

	return next_alpha


def find_alpha(current_alpha, mlp, thresh):
	f = alpha_seq_lambda(mlp)
	return mpe(current_alpha, f, thresh)




def seq_posterior2(estimate, x):

	alpha  = estimate.alpha
	topics = estimate.topics
	gamma_threshold = estimate.gamma_threshold

	num_documents, idx, counts, gammas, enc_data = x

	num_topics  = len(topics)
	num_samples = len(idx)

	per_topic_log_densities0 = np.asarray([topics[i].seq_log_density(enc_data) for i in range(num_topics)]).transpose()

	per_topic_log_densities = per_topic_log_densities0.copy()
	max_val = per_topic_log_densities.max(axis=1, keepdims=True)
	per_topic_log_densities -= max_val
	per_topic_log_densities = np.exp(per_topic_log_densities)

	idx_full  = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
	idx_full *= num_topics
	idx_full += np.reshape(np.arange(num_topics), (1, num_topics))
	alpha_loc = np.repeat(np.reshape(alpha, (1, num_topics)), num_documents, axis=0)

	if gammas is None:
		document_gammas = alpha_loc + np.reshape(np.bincount(idx_full.flat), (num_documents, num_topics))/float(num_topics)
	else:
		document_gammas = gammas.copy()

	document_gammas = document_gammas.astype(np.float64)
	idx = idx.astype(np.intp)
	alpha_loc = alpha_loc.astype(np.float64)
	per_topic_log_densities = per_topic_log_densities.astype(np.float64)
	ccc = counts.astype(np.float64)

	rv0 = np.zeros(num_documents, dtype=bool)
	rv1 = np.zeros(document_gammas.shape, dtype=np.float64)
	rv2 = np.zeros(document_gammas.shape, dtype=np.float64)
	rv3 = np.zeros(per_topic_log_densities.shape, dtype=np.float64)
	rv4 = np.arange(0, num_samples, dtype=np.intp)
	rv5 = np.zeros(num_documents, dtype=np.float64)

	aa, bb = pysp.c_ext.lda_update(idx, document_gammas, rv1, rv2, alpha_loc, per_topic_log_densities, rv3, ccc, rv0, rv4, rv5, -1, gamma_threshold)

	final_gammas = bb + alpha_loc
	log_density_gamma = aa

	return log_density_gamma, final_gammas, per_topic_log_densities0

def seq_posterior(estimate, x):

	alpha = estimate.alpha
	topics = estimate.topics
	gamma_threshold = estimate.gamma_threshold


	num_documents, idx, counts, gammas, enc_data = x


	num_topics  = len(topics)
	num_samples = len(idx)

	per_topic_log_densities   = np.asarray([topics[i].seq_log_density(enc_data) for i in range(num_topics)]).transpose()
	per_topic_log_densities2  = per_topic_log_densities.copy()
	per_topic_log_densities2 -= np.max(per_topic_log_densities2, axis=1, keepdims=True)
	np.exp(per_topic_log_densities2, out=per_topic_log_densities2)
	per_topic_log_densities3 = per_topic_log_densities2.copy()

	idx_full  = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
	idx_full *= num_topics
	idx_full += np.reshape(np.arange(num_topics), (1, num_topics))
	alpha_loc = np.reshape(alpha, (1, num_topics))


	if gammas is None:
		document_gammas = alpha_loc + np.reshape(np.bincount(idx_full.flat), (num_documents, num_topics))/float(num_topics)
	else:
		document_gammas = gammas.copy()

	document_gammas2 = np.zeros((num_documents, num_topics), dtype=float)
	document_gammas3 = np.zeros((num_documents, num_topics), dtype=float)


	gamma_sum = np.zeros((num_documents, 1), dtype=float)
	gamma_asum = np.zeros((num_documents, 1), dtype=float)


	posterior_sum_ll = np.zeros((num_samples, 1), dtype=float)

	log_density_gamma = np.zeros(per_topic_log_densities.shape, dtype=float)
	document_gamma_diff_loc = np.zeros((num_documents, num_topics), dtype=float)
	log_density_gamma_loc = log_density_gamma.view()
	posterior_sum_ll_loc = 	posterior_sum_ll.view()
	gamma_asum_loc = gamma_asum.view()
	gamma_sum_loc = gamma_sum.view()

	ndoc = num_documents

	rel_idx = idx.copy()
	rel_counts = counts.copy()
	rel_counts = np.reshape(rel_counts, (-1, 1))

	rem_gammas_idx = np.arange(num_documents, dtype=int)
	final_gammas = np.zeros((num_documents, num_topics), dtype=float)
	final_gammas_idx = np.zeros(num_documents, dtype=int)
	finished_count = 0
	itr_cnt = 0
	gamma_itr_cnt = np.zeros(num_documents, dtype=int)


	#

	digamma(document_gammas, out=document_gammas2)
	temp = np.max(document_gammas2, axis=1, keepdims=True)
	np.exp(document_gammas2 - temp, out=document_gammas3)

	np.multiply(per_topic_log_densities2, document_gammas3[rel_idx, :], out=log_density_gamma_loc)
	np.sum(log_density_gamma_loc, axis=1, keepdims=True, out=posterior_sum_ll_loc)
	log_density_gamma_loc /= posterior_sum_ll_loc

	old_stuff = None


	while ndoc > 0:

		itr_cnt += 1

		digamma(document_gammas, out=document_gammas2)
		temp = np.max(document_gammas2, axis=1, keepdims=True)
		document_gammas2 -= temp
		np.exp(document_gammas2, out=document_gammas3)

		np.multiply(per_topic_log_densities2, document_gammas3[rel_idx, :], out=log_density_gamma_loc)
		np.sum(log_density_gamma_loc, axis=1, keepdims=True, out=posterior_sum_ll_loc)
		posterior_sum_ll_loc /= rel_counts
		log_density_gamma_loc /= posterior_sum_ll_loc

		gamma_updates = np.bincount(idx_full.flat, weights=log_density_gamma_loc.flat)
		gamma_updates = np.reshape(gamma_updates, (-1, num_topics))
		gamma_updates += alpha_loc


		np.subtract(document_gammas, gamma_updates, out=document_gamma_diff_loc)
		np.abs(document_gamma_diff_loc, out=document_gamma_diff_loc)
		np.sum(document_gamma_diff_loc, axis=1, keepdims=True, out=gamma_asum_loc)
		np.sum(gamma_updates, axis=1, keepdims=True, out=gamma_sum_loc)
		gamma_asum_loc /= gamma_sum_loc


		document_gammas = gamma_updates



		has_finished = np.nonzero(gamma_asum_loc.flat <= gamma_threshold)[0]

		if has_finished.size != 0:

			final_gammas[finished_count:(finished_count+len(has_finished)), :] = document_gammas[has_finished, :]
			final_gammas_idx[finished_count:(finished_count + len(has_finished))] = rem_gammas_idx[has_finished]
			gamma_itr_cnt[finished_count:(finished_count + len(has_finished))] = itr_cnt

			is_rem_bool = gamma_asum_loc.flat > gamma_threshold

			is_rem_idx = np.nonzero(is_rem_bool)[0]
			rem_gammas_idx = rem_gammas_idx[is_rem_bool]
			finished_count += has_finished.size

			temp = np.zeros(ndoc, dtype=bool)
			temp[is_rem_bool] = True
			temp2 = np.arange(ndoc, dtype=int)
			temp2[temp] = np.arange(is_rem_idx.size, dtype=int)

			keep = temp[rel_idx]
			rel_idx = temp2[rel_idx[temp[rel_idx]]]

			idx_full = np.repeat(np.reshape(rel_idx, (-1, 1)), num_topics, axis=1)
			idx_full *= num_topics
			idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

			per_topic_log_densities2 = per_topic_log_densities2[keep, :]
			rel_counts = rel_counts[keep]
			nrec = per_topic_log_densities2.shape[0]
			ndoc = is_rem_idx.size

			log_density_gamma_loc = log_density_gamma[:nrec, :]
			posterior_sum_ll_loc = posterior_sum_ll[:nrec, :]
			gamma_sum_loc = gamma_sum[:ndoc, :]
			gamma_asum_loc = gamma_asum[:ndoc, :]
			document_gamma_diff_loc = document_gamma_diff_loc[:ndoc, :]

			document_gammas = document_gammas[is_rem_idx, :]
			document_gammas2 = document_gammas2[:ndoc, :]
			document_gammas3 = document_gammas3[:ndoc, :]




	#
	# Accumulate per-bag-sample
	#

	sidx = np.argsort(final_gammas_idx)
	final_gammas = final_gammas[sidx, :]
	gamma_itr_cnt = gamma_itr_cnt[sidx]

	digamma_gammas = digamma(final_gammas)
	temp2 = np.max(digamma_gammas, axis=1, keepdims=True)
	temp3 = np.exp(digamma_gammas-temp2)

	#per_topic_log_densities2  = per_topic_log_densities.copy()
	#per_topic_log_densities2 -= np.max(per_topic_log_densities2, axis=1, keepdims=True)
	#np.exp(per_topic_log_densities2, out=per_topic_log_densities2)

	np.multiply(per_topic_log_densities3, temp3[idx, :], out=log_density_gamma)
	np.sum(log_density_gamma, axis=1, keepdims=True, out=posterior_sum_ll)
	posterior_sum_ll /= np.reshape(counts, (-1, 1))
	log_density_gamma /= posterior_sum_ll

	effNs = log_density_gamma.sum(axis=0)


	idx_full = np.repeat(np.reshape(idx, (-1, 1)), num_topics, axis=1)
	idx_full *= num_topics
	idx_full += np.reshape(np.arange(num_topics), (1, num_topics))

	gamma_updates = np.bincount(idx_full.flat, weights=log_density_gamma.flat)
	gamma_updates = np.reshape(gamma_updates, (-1, num_topics))
	gamma_updates += alpha_loc
	final_gammas = gamma_updates

	mlpf = digamma(final_gammas) - digamma(np.sum(final_gammas, axis=1, keepdims=True))

	return log_density_gamma, final_gammas, per_topic_log_densities
