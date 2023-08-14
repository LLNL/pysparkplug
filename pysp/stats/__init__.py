__all__ = [
    "BinomialDistribution",
    "BinomialSampler",
    "BinomialEstimator",
    "CategoricalDistribution",
    "CategoricalEstimator",
    "CategoricalSampler",
    "CompositeDistribution",
    "CompositeEstimator",
    "CompositeSampler",
    "ConditionalDistribution",
    "ConditionalDistributionEstimator",
    "ConditionalDistributionSampler",
    "DiagonalGaussianDistribution",
    "DiagonalGaussianEstimator",
    "DiagonalGaussianSampler",
    "SelectDistribution",
    "SelectEstimator",
    "SelectSampler",
    "OptionalDistribution",
    "OptionalEstimator",
    "SequenceDistribution",
    "SequenceEstimator",
    "GaussianDistribution",
    "GaussianEstimator",
    "GaussianSampler",
    "GeometricDistribution",
    "GeometricEstimator",
    "GeometricSampler",
    "GammaDistribution",
    "GammaEstimator",
    "GammaSampler",
    "GaussianMixtureDistribution",
    "GaussianMixtureEstimator",
    "GaussianMixtureSampler",
    "ExponentialDistribution",
    "ExponentialEstimator",
    "ExponentialSampler",
    "DirichletDistribution",
    "DirichletEstimator",
    "DirichletSampler",
    "PoissonDistribution",
    "PoissonEstimator",
    "PoissonSampler",
    "MarkovChainDistribution",
    "MarkovChainEstimator",
    "MarkovChainSampler",
    "MarkovTransformDistribution",
    "MarkovTransformEstimator",
    "MarkovTransformSampler",
    "MixtureDistribution",
    "MixtureEstimator",
    "MixtureSampler",
    "JointMixtureDistribution",
    "JointMixtureEstimator",
    "JointMixtureSampler",
    "HierarchicalMixtureDistribution",
    "HierarchicalMixtureEstimator",
    "HierarchicalMixtureSampler",
    "HiddenAssociationDistribution",
    "HiddenAssociationEstimator",
    "HiddenAssociationSampler",
    "WeightedDistribution",
    "WeightedEstimator",
    "WeightedSampler",
    "IntegerBernoulliSetDistribution",
    "IntegerBernoulliSetEstimator",
    "IntegerBernoulliSetSampler",
    "IntegerBernoulliEditDistribution",
    "IntegerBernoulliEditEstimator",
    "IntegerBernoulliEditSampler",
    "IntegerCategoricalDistribution",
    "IntegerCategoricalEstimator",
    "IntegerCategoricalSampler",
    "IntegerHiddenAssociationDistribution",
    "IntegerHiddenAssociationEstimator",
    "IntegerHiddenAssociationSampler",
    "IntegerMarkovChainDistribution",
    "IntegerMarkovChainEstimator",
    "IntegerMarkovChainSampler",
    "IntegerMultinomialDistribution",
    "IntegerMultinomialEstimator",
    "IntegerMultinomialSampler",
    "IntegerPLSIDistribution",
    "IntegerPLSIEstimator",
    "IntegerPLSISampler",
    "LookbackHiddenMarkovDistribution",
    "LookbackHiddenMarkovEstimator",
    "LookbackHiddenMarkovSampler",
    "MultinomialDistribution",
    "MultinomialEstimator",
    "MultinomialSampler",
    "MultivariateGaussianDistribution",
    "MultivariateGaussianEstimator",
    "MultivariateGaussianSampler",
    "BernoulliSetDistribution",
    "BernoulliSetEstimator",
    "BernoulliSetSampler",
    "LDADistribution",
    "LDAEstimator",
    "LDASampler",
    "LLDADistribution",
    "LLDAEstimator",
    "LLDASampler",
    "HiddenMarkovModelDistribution",
    "HiddenMarkovEstimator",
    "HiddenMarkovSampler",
    "IndPiHiddenMarkovModelDistribution",
    "IndPiHiddenMarkovEstimator",
    "IndPiHiddenMarkovSampler",
    "IgnoredDistribution",
    "IgnoredEstimator",
    "IgnoredSampler",
    "VonMisesFisherDistribution",
    "VonMisesFisherEstimator",
    "VonMisesFisherSampler",
    "ICLTreeDistribution",
    "ICLTreeEstimator",
    "ICLTreeSampler",
    "SemiSupervisedMixtureDistribution",
    "SemiSupervisedMixtureEstimator",
    "SemiSupervisedMixtureSampler",
    "SparseMarkovAssociationDistribution",
    "SparseMarkovAssociationEstimator",
    "SparseMarkovAssociationSampler",
    "SpearmanRankingDistribution",
    "SpearmanRankingEstimator",
    "SpearmanRankingSampler",
    "IntegerStepBernoulliEditDistribution",
    "IntegerStepBernoulliEditEstimator",
    "IntegerStepBernoulliEditSampler",
    "estimate",
    "seq_estimate",
    "initialize",
    "seq_log_density_sum",
    "seq_encode",
    "seq_log_density",
]

# from pysp.arithmetic import *

from pysp.stats.lookback_hmm import (
    LookbackHiddenMarkovDistribution,
    LookbackHiddenMarkovEstimator,
    LookbackHiddenMarkovSampler,
)

import importlib

if importlib.util.find_spec("cnrg") is not None:
    from pysp.stats.grammar import GrammarDistribution, GrammarEstimator, GrammarSampler

    __all__.extend(["GrammarDistribution", "GrammarEstimator", "GrammarSampler"])

from pysp.stats.int_markovchain import (
    IntegerMarkovChainDistribution,
    IntegerMarkovChainEstimator,
    IntegerMarkovChainSampler,
)
from pysp.stats.hidden_association import (
    HiddenAssociationDistribution,
    HiddenAssociationEstimator,
    HiddenAssociationSampler,
)
from pysp.stats.spearman_rho import (
    SpearmanRankingDistribution,
    SpearmanRankingEstimator,
    SpearmanRankingSampler,
)

from pysp.stats.binomial import BinomialDistribution, BinomialSampler, BinomialEstimator
from pysp.stats.int_edit_setdist import (
    IntegerBernoulliEditDistribution,
    IntegerBernoulliEditEstimator,
    IntegerBernoulliEditSampler,
)
from pysp.stats.int_edit_stepsetdist import (
    IntegerStepBernoulliEditDistribution,
    IntegerStepBernoulliEditEstimator,
    IntegerStepBernoulliEditSampler,
)
from pysp.stats.catmultinomial import (
    MultinomialDistribution,
    MultinomialEstimator,
    MultinomialSampler,
)
from pysp.stats.intmultinomial import (
    IntegerMultinomialDistribution,
    IntegerMultinomialEstimator,
    IntegerMultinomialSampler,
)
from pysp.stats.markov_transform import (
    MarkovTransformDistribution,
    MarkovTransformEstimator,
    MarkovTransformSampler,
)
from pysp.stats.sparse_markov_transform import (
    SparseMarkovAssociationDistribution,
    SparseMarkovAssociationEstimator,
    SparseMarkovAssociationSampler,
)
from pysp.stats.int_hidden_association import (
    IntegerHiddenAssociationDistribution,
    IntegerHiddenAssociationEstimator,
    IntegerHiddenAssociationSampler,
)
from pysp.stats.dmvn import (
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
    DiagonalGaussianSampler,
)
from pysp.stats.mvnmixture import (
    GaussianMixtureDistribution,
    GaussianMixtureEstimator,
    GaussianMixtureSampler,
)
from pysp.stats.intsetdist import (
    IntegerBernoulliSetDistribution,
    IntegerBernoulliSetEstimator,
    IntegerBernoulliSetSampler,
)
from pysp.stats.int_plsi import (
    IntegerPLSIDistribution,
    IntegerPLSIEstimator,
    IntegerPLSISampler,
)
from pysp.stats.ignored import IgnoredDistribution, IgnoredEstimator, IgnoredSampler
from pysp.stats.categorical import (
    CategoricalDistribution,
    CategoricalEstimator,
    CategoricalSampler,
)
from pysp.stats.composite import (
    CompositeDistribution,
    CompositeEstimator,
    CompositeSampler,
)
from pysp.stats.conditional import (
    ConditionalDistribution,
    ConditionalDistributionEstimator,
    ConditionalDistributionSampler,
)
from pysp.stats.select import SelectDistribution, SelectEstimator, SelectSampler
from pysp.stats.optional import OptionalDistribution, OptionalEstimator
from pysp.stats.sequence import SequenceDistribution, SequenceEstimator
from pysp.stats.gaussian import GaussianDistribution, GaussianEstimator, GaussianSampler
from pysp.stats.gamma import GammaDistribution, GammaEstimator, GammaSampler
from pysp.stats.exponential import (
    ExponentialDistribution,
    ExponentialEstimator,
    ExponentialSampler,
)
from pysp.stats.mixture import MixtureDistribution, MixtureEstimator, MixtureSampler
from pysp.stats.hmixture import (
    HierarchicalMixtureDistribution,
    HierarchicalMixtureEstimator,
    HierarchicalMixtureSampler,
)
from pysp.stats.dirichlet import (
    DirichletDistribution,
    DirichletEstimator,
    DirichletSampler,
)
from pysp.stats.poisson import PoissonDistribution, PoissonEstimator, PoissonSampler
from pysp.stats.markovchain import (
    MarkovChainDistribution,
    MarkovChainEstimator,
    MarkovChainSampler,
)
from pysp.stats.weighted import WeightedDistribution, WeightedEstimator, WeightedSampler
from pysp.stats.intrange import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
    IntegerCategoricalSampler,
)
from pysp.stats.mvn import (
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
    MultivariateGaussianSampler,
)
from pysp.stats.lda import LDADistribution, LDASampler, LDAEstimator
from pysp.stats.llda import LLDADistribution, LLDASampler, LLDAEstimator
from pysp.stats.jmixture import (
    JointMixtureDistribution,
    JointMixtureEstimator,
    JointMixtureSampler,
)
from pysp.stats.hidden_markov import (
    HiddenMarkovModelDistribution,
    HiddenMarkovEstimator,
    HiddenMarkovSampler,
)
from pysp.stats.hidden_markov_ind_pi import (
    IndPiHiddenMarkovModelDistribution,
    IndPiHiddenMarkovEstimator,
    IndPiHiddenMarkovSampler,
)
from pysp.stats.vmf import (
    VonMisesFisherDistribution,
    VonMisesFisherEstimator,
    VonMisesFisherSampler,
)
from pysp.stats.icltree import ICLTreeDistribution, ICLTreeEstimator, ICLTreeSampler
from pysp.stats.geometric import (
    GeometricDistribution,
    GeometricEstimator,
    GeometricSampler,
)
from pysp.stats.ss_mixture import (
    SemiSupervisedMixtureDistribution,
    SemiSupervisedMixtureEstimator,
    SemiSupervisedMixtureSampler,
)

from pysp.stats.setdist import (
    BernoulliSetDistribution,
    BernoulliSetEstimator,
    BernoulliSetSampler,
)

import numpy as np
import pickle


def load_models(x):
    return eval(x)


def dump_models(x):
    return str(x)


def estimate(data, estimator, prev_estimate=None):
    if "pyspark.rdd" in str(type(data)):

        sc = data.context
        factory = estimator.accumulatorFactory()
        estimatorBroadcast = sc.broadcast(estimator)

        temp_estimate = pickle.dumps(prev_estimate, protocol=0)
        temp_estimateB = sc.broadcast(temp_estimate)

        def acc(splitIndex, itr):
            accumulatorForSplit = estimatorBroadcast.value.accumulatorFactory().make()
            countsForSplit = 0.0
            loc_prev_estimate = pickle.loads(temp_estimateB.value)

            for x in itr:
                countsForSplit = countsForSplit + 1.0
                accumulatorForSplit.update(x, 1.0, estimate=loc_prev_estimate)

            return iter([(countsForSplit, accumulatorForSplit.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobsForSplit, statsForSplit in temp.collect():
            nobs = nobs + nobsForSplit
            accumulator.combine(statsForSplit)

        return estimator.estimate(nobs, accumulator.value())

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulatorFactory().make()
        nobs = 0.0

        for x in idata:
            nobs += 1.0
            accumulator.update(x, 1.0, estimate=prev_estimate)

        return estimator.estimate(nobs, accumulator.value())


def seq_encode(data, model, num_chunks=1, chunk_size=None):
    if "pyspark.rdd" in str(type(data)):
        sc = data.context

        temp_model = pickle.dumps(model, protocol=0)
        modelBroadcast = sc.broadcast(temp_model)

        enc_data = (
            data.glom()
            .map(lambda x: list(x))
            .map(lambda x: (len(x), pickle.loads(modelBroadcast.value).seq_encode(x)))
        )

        return enc_data

    else:
        sz = len(data)
        if chunk_size is not None:
            num_chunks_loc = int(np.ceil(float(sz) / float(chunk_size)))
        else:
            num_chunks_loc = num_chunks

        rv = []
        for i in range(num_chunks_loc):
            data_loc = [data[i] for i in range(i, sz, num_chunks_loc)]
            enc_data = model.seq_encode(data_loc)
            rv.append((len(data_loc), enc_data))

        return rv


def seq_log_density_sum(enc_data, estimate):
    if "pyspark.rdd" in str(type(enc_data)):
        sc = enc_data.context
        estimate_broadcast = sc.broadcast(pickle.dumps(estimate, protocol=0))

        def acc(itr):

            rv = 0.0
            cnt = 0.0
            estimate_loc = pickle.loads(estimate_broadcast.value)
            for sz, x in itr:
                rv += estimate_loc.seq_log_density(x).sum()
                cnt += sz

            return [(cnt, rv)]

        return enc_data.mapPartitions(acc).reduce(
            lambda a, b: (a[0] + b[0], a[1] + b[1])
        )

    else:

        return sum([u[0] for u in enc_data]), sum(
            [estimate.seq_log_density(u[1]).sum() for u in enc_data]
        )


def seq_log_density(enc_data, estimate, is_list=False):
    if "pyspark.rdd" in str(type(enc_data)):
        sc = enc_data.context
        temp_estimate = pickle.dumps(estimate, protocol=0)
        estimateBroadcast = sc.broadcast(temp_estimate)

        def acc(itr):
            loc_estimate = pickle.loads(estimateBroadcast.value)
            if is_list:
                return [
                    np.asarray([ee.seq_log_density(x) for ee in loc_estimate])
                    for sz, x in itr
                ]
            else:
                return [loc_estimate.seq_log_density(x) for sz, x in itr]

        return enc_data.mapPartitions(acc).collect()

    else:

        if is_list:
            return [
                np.asarray([ee.seq_log_density(u[1]) for ee in estimate])
                for u in enc_data
            ]
        else:
            return [estimate.seq_log_density(u[1]) for u in enc_data]


def seq_estimate(enc_data, estimator, prev_estimate):
    if "pyspark.rdd" in str(type(enc_data)):
        sc = enc_data.context

        estimatorBroadcast = sc.broadcast(estimator)
        estimateBroadcast = sc.broadcast(pickle.dumps(prev_estimate, protocol=0))

        def acc(splitIndex, itr):
            accumulatorForSplit = estimatorBroadcast.value.accumulatorFactory().make()
            countsForSplit = 0.0
            local_estimate = pickle.loads(estimateBroadcast.value)

            for sz, x in itr:
                countsForSplit = countsForSplit + sz
                accumulatorForSplit.seq_update(x, np.ones(sz), local_estimate)

            rv = pickle.dumps((countsForSplit, accumulatorForSplit.value()), protocol=0)
            # return [(countsForSplit, accumulatorForSplit.value())]
            return [rv]

        def red(x, y):
            xx = pickle.loads(x)
            yy = pickle.loads(y)
            accumulator = estimatorBroadcast.value.accumulatorFactory().make()
            nobs = xx[0] + yy[0]
            vals = accumulator.from_value(xx[1]).combine(yy[1]).value()
            rv = pickle.dumps((nobs, vals))
            # return (nobs, vals)
            return rv

        temp = enc_data.mapPartitionsWithIndex(acc, True).cache()

        nobs = 0.0
        accumulator = estimator.accumulatorFactory().make()

        for stuff in temp.collect():
            nobsForSplit, statsForSplit = pickle.loads(stuff)
            nobs = nobs + nobsForSplit
            accumulator.combine(statsForSplit)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        estimateBroadcast.destroy()
        estimatorBroadcast.destroy()
        temp.unpersist()
        enc_data.localCheckpoint()

        return estimator.estimate(nobs, accumulator.value())

    else:

        accumulator = estimator.accumulatorFactory().make()
        nobs = 0.0

        data_update = []

        for sz, x in enc_data:
            nobs += sz
            accumulator.seq_update(x, np.ones(sz), prev_estimate)
            # x_update = accumulator.seq_update(x, np.ones(sz), prev_estimate)
            # data_update.append((sz, x_update))

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(None, accumulator.value())


def initialize(data, estimator, rng, p):
    if "pyspark.rdd" in str(type(data)):
        factory = estimator.accumulatorFactory()
        sc = data.context

        num_partitions = data.getNumPartitions()
        seeds = rng.randint(2 ** 31, size=num_partitions)

        estimatorBroadcast = sc.broadcast(estimator)
        seedsBroadcast = sc.broadcast(seeds)

        def acc(splitIndex, itr):
            accumulatorForSplit = estimatorBroadcast.value.accumulatorFactory().make()
            countsForSplit = 0.0
            rng_loc = np.random.RandomState(seedsBroadcast.value[splitIndex])

            for x in itr:
                w = 1.0 if rng_loc.rand() <= p else 0.0
                countsForSplit += w
                accumulatorForSplit.initialize(x, w, rng_loc)

            return iter([(countsForSplit, accumulatorForSplit.value())])

        temp = data.mapPartitionsWithIndex(acc, True)
        nobs = 0.0
        accumulator = factory.make()

        for nobsForSplit, statsForSplit in temp.collect():
            nobs = nobs + nobsForSplit
            accumulator.combine(statsForSplit)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())

    elif hasattr(data, "__iter__"):
        idata = iter(data)
        accumulator = estimator.accumulatorFactory().make()
        nobs = 0.0

        for x in idata:
            w = 1.0 if rng.rand() <= p else 0.0
            nobs += w
            accumulator.initialize(x, w, rng)

        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)

        return estimator.estimate(nobs, accumulator.value())
