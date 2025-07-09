"""Example for DiagonalGaussianMixtureDistribution. Define distribution, 
generate data, estimate, and evaluate likelihoods.

This is a fast implementation of Mixture([DiagonalGaussianDistribution()]*K).

""" 
import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'
from numpy.random import RandomState
from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    n = int(1e4)
    rng = RandomState(1)
    # Define the model
    mu = [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]]
    covar = [[1.0, 1.0]]*3
    w = [0.50, 0.40, 0.10]
    dist = DiagonalGaussianMixtureDistribution(mu=mu, covar=covar, w=w)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Print out a few samples
    print(data[:5])
    # Define estimator
    est = DiagonalGaussianMixtureEstimator(
            num_components=3,
            dim=2,
            tied=True) 
    # Estimate model
    model = optimize(data, est, max_its=100, rng=rng, print_iter=1)
    print(str(model))
    # Eval likelihood on a an observation 
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likleihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    print(f'Likelihood of estimated model on data: {ll}')
    # Fast evaluation of the posterior
    post = model.seq_posterior(enc_data)
    print(f'Posterior of each observed point: {post}')
