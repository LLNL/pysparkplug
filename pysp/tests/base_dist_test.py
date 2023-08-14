import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'

from pysp.stats import *
from pysp.utils.estimation import empirical_kl_divergence
import numpy as np
import unittest

class BaseDistributionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        dists = []

        dists.append(BernoulliSetDistribution({'a': 0.8, 'b': 0.1, 'c': 0.7}, name='a'))
        dists.append(BernoulliSetDistribution({'a': 0.8, 'b': 0.1, 'c': 0.0}, name='a'))
        dists.append(BernoulliSetDistribution({'a': 1.0, 'b': 0.1, 'c': 0.7}, name='a'))
        dists.append(BernoulliSetDistribution({'a': 0.8, 'b': 0.2, 'c': 0.6}, min_prob=1.0e-128, name='a'))
        dists.append(BinomialDistribution(0.25, 5, name='a'))
        dists.append(CategoricalDistribution({'a': 0.4, 'b': 0.3, 'c': 0.2, 'd': 0.1}, name='a'))
        dists.append(CompositeDistribution((ExponentialDistribution(3.1), PoissonDistribution(3.2))))
        dists.append(DiagonalGaussianDistribution([1.8, 4.3, -1.5], [1.1, 4.8, 9.1], name='a'))
        dists.append(DirichletDistribution([1.1, 2.8, 4.5], name='a'))
        dists.append(ExponentialDistribution(10.8, name='a'))
        dists.append(GammaDistribution(3.3, 2.0, name='a'))
        dists.append(GaussianDistribution(1.0, 2.0))
        dists.append(GeometricDistribution(0.8, name='a'))
        dists.append(IgnoredDistribution(GeometricDistribution(0.8)))
        dists.append(IntegerBernoulliSetDistribution(np.log([0.9, 0.8, 0.7, 0.6, 0.5]), name='a'))
        dists.append(IntegerBernoulliSetDistribution(np.log([0.9, 0.8, 0.7, 0.6, 0.5]), np.log([0.1, 0.2, 0.3, 0.4, 0.5]), name='a'))
        dists.append(IntegerCategoricalDistribution(0, [0.1, 0.4, 0.3, 0.2], name='a'))
        dists.append(IntegerMultinomialDistribution(0, [0.1, 0.4, 0.3, 0.2], len_dist=CategoricalDistribution({4:1.0}), name='a'))

        dists.append(MarkovChainDistribution({'a': 0.1, 'b': 0.5, 'c': 0.4},
                                             {'a': {'a': 0.8, 'b': 0.1, 'c': 0.1},
                                              'b': {'a': 0.1, 'b': 0.8, 'c': 0.1},
                                              'c': {'a': 0.1, 'b': 0.1, 'c': 0.8}},
                                             len_dist=CategoricalDistribution({5:1.0}), name='a'))

        dists.append(MultinomialDistribution(IntegerCategoricalDistribution(0, [0.1, 0.4, 0.3, 0.2]), CategoricalDistribution({5:1.0}), name='a'))
        dists.append(MultivariateGaussianDistribution([1.0, 3.3, 2.2], [[3.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 3.0]], name='a'))

        dists.append(PoissonDistribution(4.7, name='a'))
        dists.append(OptionalDistribution(PoissonDistribution(4.7), p=0.1, name='a'))
        dists.append(OptionalDistribution(PoissonDistribution(4.7), p=0.1, missing_value='asdf', name='a'))
        dists.append(OptionalDistribution(BinomialDistribution(0.25, 5, name='a'), p=0.2, missing_value=float("nan"), name='a'))
        dists.append(SequenceDistribution(GeometricDistribution(0.8), len_dist=CategoricalDistribution({5:1.0}), name='a'))
        dists.append(SequenceDistribution(GeometricDistribution(0.8), len_dist=CategoricalDistribution({5:1.0}), len_normalized=True, name='a'))
        dists.append(VonMisesFisherDistribution([1.1,2.1,3.1,4.1,5.1], 2.0, name='a'))

        self.dists = dists


    def test_sampler_repeat(self):
        for dist in self.dists:
            res = sampler_repeat_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_string_match(self):
        for dist in self.dists:
            res = string_match_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_eval(self):
        for dist in self.dists:
            res = string_eval_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_log_density(self):
        for dist in self.dists:
            res = log_density_test(dist)
            if not res[0]:
                print(str(dist))
            self.assertTrue(res[0], str(res[1]))

    def test_estimation(self):
        for dist in self.dists:
            res = estimation_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_seq_estimation(self):
        for dist in self.dists:
            res = seq_estimation_test(dist)
            self.assertTrue(res[0], str(res[1]))

    def test_estimation_same_name(self):
        for dist in self.dists:
            res = estimation_same_name_test(dist)
            #if not res[0]:
            #    print(str(dist))
            self.assertTrue(res[0], str((dist,res[1])))

def sampler_repeat_test(dist):

    seeds = [1,2,3]
    sz = 20
    rv = []
    for seed in seeds:

        s  = dist.sampler(seed)
        d1 = s.sample(size=sz)
        s  = dist.sampler(seed)
        d2 = s.sample(size=sz)

        is_same = [u[0] == u[1] for u in zip(map(str, d1), map(str, d2))]

        rv.append(all(is_same))

    return all(rv), rv
def string_match_test(dist):
    sdist = eval(str(dist))
    return str(sdist) == str(dist), '__str__ is not idempotent.'
def string_eval_test(dist):

    seeds = [1, 2]
    sz = 5
    rv = []
    for seed in seeds:

        s  = dist.sampler(seed)
        data = s.sample(size=sz)

        sdist = eval(str(dist))

        enc_data = dist.seq_encode(data)
        seq_ll0  = dist.seq_log_density(enc_data)
        seq_ll1  = sdist.seq_log_density(enc_data)
        seq_dll  = np.zeros(sz, dtype=np.float64)

        for i in range(sz):

            if seq_ll0[i] == 0:
                seq_dll[i] = np.abs(seq_ll1[i])
            else:
                seq_dll[i] = np.abs(seq_ll0[i] - seq_ll1[i])/np.abs(seq_ll0[i])

        rv.append(np.max(seq_dll))

    return max(rv) < 1.0e-15, max(rv)
def log_density_test(dist):

    seeds = [1, 2, 3]
    sz = 20
    rv = []
    for seed in seeds:

        s  = dist.sampler(seed)
        data = s.sample(size=sz)

        seq_ll = dist.seq_log_density(dist.seq_encode(data))
        for i in range(sz):

            if seq_ll[i] == 0:
                seq_ll[i] = np.abs(dist.log_density(data[i]))
            else:
                seq_ll[i] = np.abs(seq_ll[i] - dist.log_density(data[i]))/np.abs(seq_ll[i])

        rv.append(max(seq_ll))

    return max(rv) < 1.0e-14, max(rv)

def estimation_test(dist):

    seeds = [1, 2, 3, 4]
    szs = [50, 500, 5000]
    rv  = []

    akld = []
    for seed in seeds:

        kld = []
        better = []
        for sz in szs:

            data  = dist.sampler(seed).sample(size=sz)
            est   = dist.estimator()
            enc_data = seq_encode(data, dist)
            est_dist = estimate(data, est, None)

            emp_kld, _, _ = empirical_kl_divergence(dist, est_dist, enc_data)

            if len(kld) > 0:
                better.append(kld[-1] >= emp_kld)

            kld.append(emp_kld)
        akld.append(kld)
        rv.append(all(better))

    akld_mean = np.mean(akld, axis=0)
    rv = np.all(akld_mean[1:] <= akld_mean[:-1])

    return rv, akld

def seq_estimation_test(dist):

    seeds = [1, 2, 3, 4]
    szs = [50, 500, 5000]
    rv  = []

    akld = []
    for seed in seeds:

        kld = []
        better = []
        for sz in szs:

            data  = dist.sampler(seed).sample(size=sz)
            est   = dist.estimator()
            enc_data = seq_encode(data, dist)
            est_dist = seq_estimate(enc_data, est, None)

            emp_kld, _, _ = empirical_kl_divergence(dist, est_dist, enc_data)

            if len(kld) > 0:
                better.append(kld[-1] >= emp_kld)

            kld.append(emp_kld)
        akld.append(kld)
        rv.append(all(better))

    akld_mean = np.mean(akld, axis=0)
    rv = np.all(akld_mean[1:] <= akld_mean[:-1])

    return rv, akld

def estimation_same_name_test(dist):

    if not hasattr(dist, 'name'):
        return True, ''

    seed = 1
    data = dist.sampler(seed).sample(100)
    est = dist.estimator()
    model = estimate(data, est, None)

    return model.name is dist.name, ''

def evaluate_dists(dists):

    tests = [sampler_repeat_test, log_density_test, estimation_test, string_match_test, string_eval_test]

    for dist in dists:
        print(str(dist))
        all_res1 = []
        all_res2 = []
        for test in tests:
            res = test(dist)
            all_res1.append(res[0])
            all_res2.append((test.__name__, res))
        passed = all(all_res1)
        if passed:
            print('Passed All Tests')
        else:
            for t in all_res2:
                if not t[1][0]:
                    print(t)
        print('-'*10)


if __name__ == '__main__':
    unittest.main()