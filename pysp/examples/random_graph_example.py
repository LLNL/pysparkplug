import numpy as np
from pysp.stats import *
from pysp.utils.estimation import optimize, partition_data


if __name__ == '__main__':

    P = np.zeros((4,4))
    P[0, :] = [0.0, 0.8, 0.1, 0.8]
    P[1, :] = [0.8, 0.0, 0.8, 0.1]
    P[2, :] = [0.1, 0.8, 0.0, 0.8]
    P[3, :] = [0.8, 0.1, 0.8, 0.0]

    dist1 = IntegerBernoulliSetDistribution(np.log(P.flatten()))

    P = np.zeros((4,4))
    P[0, :] = [0.0, 0.1, 0.8, 0.1]
    P[1, :] = [0.1, 0.0, 0.1, 0.8]
    P[2, :] = [0.8, 0.1, 0.0, 0.1]
    P[3, :] = [0.1, 0.8, 0.1, 0.0]

    dist2 = IntegerBernoulliSetDistribution(np.log(P.flatten()))
    dist = MixtureDistribution([dist1, dist2], [0.5, 0.5])

    data = dist.sampler(1).sample(1000)

    est = MixtureEstimator([IntegerBernoulliSetEstimator(16)]*2)

    model = optimize(data, est, max_its=100, delta=None, rng=np.random.RandomState(1))

    print(list(np.exp(model.components[0].log_pvec)))
    print(list(np.exp(model.components[1].log_pvec)))
