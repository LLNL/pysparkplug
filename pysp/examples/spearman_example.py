from pysp.stats.spearman_rho import SpearmanRankingDistribution, SpearmanRankingEstimator
import numpy as np

if __name__ == '__main__':

    dist = SpearmanRankingDistribution([2,3,0,1])

    data = dist.sampler(1).sample(100)

    est = SpearmanRankingEstimator(4)
    acc = est.accumulatorFactory().make()

    enc_data = dist.seq_encode(data)
    acc.seq_update(enc_data, np.ones(len(data)), None)

    est_model = est.estimate(None, acc.value())

    print(str(est_model))