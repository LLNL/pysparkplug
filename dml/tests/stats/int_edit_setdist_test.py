import os


from dml.tests.stats.stats_tests import * 
from dml.stats import *
from dml.stats.int_edit_setdist import *
from dml.stats.intsetdist import *
import numpy as np
import pytest 

class IntegerStepBernoulliEditDistributionTestCase(StatsTestClass):
    def setUp(self) -> None:
        rng = np.random.RandomState(10)
        init_dists = [
            IntegerBernoulliSetDistribution(np.log([0.3, 0.3, 0.3])),
            IntegerBernoulliSetDistribution(np.log([0.25]*4))
        ]
        log_edit_pmat = [
            np.log(rng.dirichlet(np.ones(3), size=2)).T,
            np.log(rng.dirichlet(np.ones(4), size=2)).T
        ]
        init_ests = [
            IntegerBernoulliSetEstimator(num_vals=3),
            IntegerBernoulliSetEstimator(num_vals=4)
        ]
        init_factory = [
            IntegerBernoulliSetAccumulatorFactory(num_vals=3),
            IntegerBernoulliSetAccumulatorFactory(num_vals=4)
        ]
        init_accumulator = [
            IntegerBernoulliSetAccumulator(num_vals=3),
            IntegerBernoulliSetAccumulator(num_vals=4)
        ]
        init_encoder= IntegerBernoulliSetDataEncoder()

        self.eval_dists = [
            IntegerBernoulliEditDistribution(log_edit_pmat=log_edit_pmat[0], init_dist=init_dists[0], name='name', keys='keys'),
            IntegerBernoulliEditDistribution(log_edit_pmat=log_edit_pmat[0], init_dist=init_dists[0], keys='keys'),
            IntegerBernoulliEditDistribution(log_edit_pmat=log_edit_pmat[1], init_dist=init_dists[1], name='name'),
            IntegerBernoulliEditDistribution(log_edit_pmat=log_edit_pmat[1], init_dist=init_dists[1])
        ]
        self._ests = [
            IntegerBernoulliEditEstimator(num_vals=3, init_estimator=init_ests[0], name='name', keys='keys'),
            IntegerBernoulliEditEstimator(num_vals=3, init_estimator=init_ests[0], keys='keys'),
            IntegerBernoulliEditEstimator(num_vals=4, init_estimator=init_ests[1], name='name'),
            IntegerBernoulliEditEstimator(num_vals=4, init_estimator=init_ests[1])
        ]
        self._factories = [
            IntegerBernoulliEditAccumulatorFactory(num_vals=3, init_factory=init_factory[0], name='name', keys='keys'),
            IntegerBernoulliEditAccumulatorFactory(num_vals=3, init_factory=init_factory[0], keys='keys'),
            IntegerBernoulliEditAccumulatorFactory(num_vals=4, init_factory=init_factory[1], name='name'),
            IntegerBernoulliEditAccumulatorFactory(num_vals=4, init_factory=init_factory[1])

        ]
        self._accumulators = [
            IntegerBernoulliEditAccumulator(num_vals=3, init_acc=init_accumulator[0], name='name', keys='keys'),
            IntegerBernoulliEditAccumulator(num_vals=3, init_acc=init_accumulator[0], keys='keys'),
            IntegerBernoulliEditAccumulator(num_vals=4, init_acc=init_accumulator[1], name='name'),
            IntegerBernoulliEditAccumulator(num_vals=4, init_acc=init_accumulator[1])
        ]
        self._encoders = [IntegerBernoulliEditDataEncoder(init_encoder=init_encoder)]*len(self.eval_dists)

        # Define members for tests
        self.dist_est = [(d, e) for d, e in zip(self.eval_dists, self._ests)]
        self.dist_encoder = [(d, e) for d, e in zip(self.eval_dists, self._encoders)]
        self.density_dist_encoder = self.dist_encoder
        self.sampler_dist = self.eval_dists[0]
        self.est_factory = [(e, f) for e, f in zip(self._ests, self._factories)]
        self.factory_acc = [(f, a) for f, a in zip(self._factories, self._accumulators)]
        self.acc_encoder = [(a, e) for a, e in zip(self._accumulators, self._encoders)]

        self.type_check_data = [None, np.ones((10, 10))]

    def test_seq_log_density_type(self):
        for x in self.type_check_data:
            with pytest.raises(Exception) as e:
                self.eval_dists[0].seq_log_density(x)
            assert str(e.value) == "IntegerBernoulliEditEncodedDataSequence required for seq_log_density()."