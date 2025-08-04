"""Generate data and fit a hierarchical mixture model. 
This is a mixture sequence mixture distribution."""
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
import numpy as np
from dml.stats import *
from dml.utils.estimation import optimize


if __name__ == '__main__':

    # Define distribution
    topic1 = CategoricalDistribution({'a': 0.50, 'b': 0.25, 'c': 0.25})
    topic2 = CategoricalDistribution({'a': 0.25, 'b': 0.50, 'c': 0.25})
    topic3 = CategoricalDistribution({'a': 0.25, 'b': 0.25, 'c': 0.50})

    w = [0.25, 0.25, 0.25, 0.25]
    taus = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.3, 0.4, 0.3]]

    len_dist = CategoricalDistribution({8:0.1, 9:0.2, 10:0.7})
    dist = HierarchicalMixtureDistribution([topic1, topic2, topic3], w, taus, len_dist=len_dist)

    # Sample from each mixture component
    t1_sample = topic1.sampler(1).sample(10)
    t2_sample = topic2.sampler(2).sample(10)
    t3_sample = topic3.sampler(3).sample(10)
    # Print the posterior for the component at each value
    print(dist.posterior(t1_sample))
    print(dist.posterior(t2_sample))
    print(dist.posterior(t3_sample))

    # Sample data from hierarchical mixture model
    data = dist.sampler(1).sample(2000)

    # Estimate model parameters
    num_topics = 3
    num_mixtures = 4
    est0 = CategoricalEstimator()
    est1 = CategoricalEstimator()
    est = HierarchicalMixtureEstimator([est0]*num_topics, num_mixtures, len_estimator=est1)
    # Fit model
    model = optimize(data, est, max_its=10000, print_iter=500, rng=np.random.RandomState(2))
    print(str(model))
    # Evaluate the likelihood of one obs
    ll0 = model.log_density(data[0])
    print(f'Likelihood of model fit at {data[0]}: {ll0}')
    # Encode data for vectorized likelihood eval
    enc_data = seq_encode(data[:5], model=model)[0][1]
    ll = model.seq_log_density(enc_data)
    for x, y in zip(data[:5], ll):
        print(f'Obs: {x}, Likelihood: {y}')
    # Vectorized eval of posterior
    post = model.seq_posterior(enc_data)
    for x, y in zip(data[:5], post):
        print(f'Obs {x}, Posterior {y}')

