"""IntegerCategoricalDistribution example on generated data.""" 
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
import numpy as np
from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    n = int(1e4)
    rng = np.random.RandomState(1)
    # Define the model
    p=np.ones(10) / 10.0
    dist=IntegerCategoricalDistribution(min_val=0, p_vec=p)
    # Generate data from sampler
    sampler = dist.sampler(seed=1)
    data = sampler.sample(n)
    # Define estimator
    est=IntegerCategoricalEstimator()
    # Estimate model
    model = optimize(
            data=data,
            estimator=est,
            rng=rng,
            max_its=1000,
            print_iter=100)
    print(model)
    # Eval likelihood on a an observation 
    ll0 = model.log_density(data[0])
    print(f'Likelihood of estimated model eval at {data[0]}: {ll0}')
    # Encode data for vectorized calls
    enc_data = seq_encode(data, model=model)[0][1]
    # Eval likleihood at all data points (fast)
    ll = model.seq_log_density(enc_data)
    for x, y in zip(data[:5], ll[:5]):
        print(f'Obs: {str(x)}, Likelihood: {y}.')

