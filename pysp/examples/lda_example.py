import os
import sys
import time
import numpy as np
from pysp.stats import *
from pysp.utils.estimation import best_of
import pysp.utils.optsutil as ops


def make_fake_data(num_topics, num_docs, snr, palpha, seed):

	wordPerDoc = 100
	numWords   = 10
	numTopics  = num_topics

	rng = np.random.RandomState(seed)

	alpha1 = palpha*np.ones(num_topics)
	alpha1[np.arange(num_topics) >= (num_topics/2)] = 0.0001

	alpha2 = palpha*np.ones(num_topics)
	alpha2[np.arange(num_topics) < (num_topics/2)] = 0.0001


	topics  = [{k: snr*rng.rand() + (1.0 if (i*numWords/numTopics) <= k and ((i+1)*numWords/numTopics) > k else 0.0) for k in range(numWords)} for i in range(numTopics)]
	topics  = [CategoricalDistribution({str(k): v/float(sum(u.values())) for k,v in u.items()}) for u in topics]


	# Create the data
	dist1 = LDADistribution(topics, alpha1, len_dist=CategoricalDistribution({wordPerDoc: 1.0}))
	dist2 = LDADistribution(topics, alpha2, len_dist=CategoricalDistribution({wordPerDoc: 1.0}))
	dist  = MixtureDistribution([dist1, dist2], [0.5, 0.5])

	data  = dist.sampler(seed=1).sample(size=num_docs)

	words = sorted(set([u for v in data for u in v]))

	return data, words, dist


if __name__ == '__main__':

	num_topics = 10
	print_cnt = 10
	rng = np.random.RandomState(2)
	#out = open('/Users/boquet1/PycharmProjects/lda_debug.log', 'wt')
	out = sys.stdout

	data, words, dist = make_fake_data(num_topics, 50, 0.0001, 1, 1)

	avg_size = np.mean([len(u) for u in data])

	out.write('#words = %d / #docs = %d / avg w/doc = %f\n' % (len(words), len(data), avg_size))

	data_cnt = [list(ops.countByValue(u).items()) for u in data]

	estimator1 = LDAEstimator([CategoricalEstimator(pseudo_count=0.001, suff_stat={w : 1.0/len(words) for w in words})]*num_topics, keys=(None, 'topics'), gamma_threshold=0.001)
	estimator = MixtureEstimator([estimator1]*2, pseudo_count=1.0)

	#mm_ll, mm = best_of(data_cnt, data_cnt, estimator, 1, 30, 0.02, 1.0e-6, rng)


	imm = initialize(data_cnt, estimator, rng, 0.1)

	enc_data   = seq_encode(data_cnt, imm)
	prev_model = imm


	dcnt, lob_sum = seq_log_density_sum(enc_data, imm)
	old_elob = lob_sum / dcnt
	d_elob = np.inf
	kk = -1

	while d_elob > 1.0e-8:
		kk += 1
		mm = seq_estimate(enc_data, estimator, prev_estimate=prev_model)

		dcnt, lob_sum = seq_log_density_sum(enc_data, mm)
		elob = lob_sum/dcnt


		prev_model = mm
		out.write('Iteration %d\tE[LOB]=%e\tdelta E[LOB]=%e\n'%(kk+1, elob, elob-old_elob))

		old_elob = elob

		if (kk+1) % print_cnt == 0:

			out.write('Weights = %s\n'%(str(','.join(map(str, mm.w)))))
			out.write('Alpha_2 = %s\n'%(str(','.join(map(str, mm.components[0].alpha)))))
			out.write('Alpha_1 = %s\n'%(str(','.join(map(str, mm.components[1].alpha)))))
			topics = mm.components[0].topics

			for i in range(num_topics):
				sidx = np.argsort(-topics[i].log_prob_vec)
				top_words = ', '.join(['%s (%f)'%(topics[i].inv_key_map[j - 1], np.exp(topics[i].log_prob_vec[j])) for j in sidx[:10]])
				out.write('Topic %d: %s\n'%(i, top_words))

		out.flush()
