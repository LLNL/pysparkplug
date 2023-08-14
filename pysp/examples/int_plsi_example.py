#import os
#os.environ['NUMBA_DISABLE_JIT'] =  '1'

from pysp.stats import *
import numpy as np
from pysp.utils.estimation import empirical_kl_divergence, optimize

if __name__ == '__main__':

    num_docs = 200
    num_words = 20
    num_states = 5
    doc_len = 20

    rng = np.random.RandomState()
    state_word_mat = -np.log(rng.rand(num_words, num_states))
    doc_state_mat  = -np.log(rng.rand(num_docs, num_states))

    state_word_mat /= state_word_mat.sum(axis=0, keepdims=True)
    doc_state_mat  /= doc_state_mat.sum(axis=1, keepdims=True)

    len_dist = CategoricalDistribution({doc_len: 1.0})

    dist = IntegerPLSIDistribution(state_word_mat, doc_state_mat, np.ones(num_docs) / num_docs, len_dist)

    data = dist.sampler(1).sample(10000)
    est = IntegerPLSIEstimator(num_words, num_states, num_docs, len_estimator=CategoricalEstimator(), pseudo_count=(0.01, 0.01, 0.01))
    model = optimize(data, est, rng=np.random.RandomState(1))
    enc_data = seq_encode(data, model)

    for i in range(100):
        model = optimize(data, est, prev_estimate=model, print_iter=10, max_its=10)
        print(empirical_kl_divergence(dist, model, enc_data))

