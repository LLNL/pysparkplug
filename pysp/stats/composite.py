"""Create, estimate, and sample from a Composite distribution.

Defines the CompositeDistribution, CompositeSampler, CompositeAccumulatorFactory, CompositeAccumulator,
CompositeEstimator, and the CompositeDataEncoder classes for use with pysparkplug.

Data type: (Tuple[T_0, ... T_{n-1}]): The CompositeDistribution of size 'n' is a joint distribution for
independent observations of 'n'-tupled data. Each component 'k' of the CompositeDistribution has data type T_k that
must be compatible with data type T_k.

"""
import numpy as np
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from typing import Optional, List, Union, Any, Tuple, Sequence, TypeVar, Dict
from pysp.arithmetic import maxrandint


class CompositeDistribution(SequenceEncodableProbabilityDistribution):
    """CompositeDistribution for modeling tuples of heterogenous data.

    Attributes:
        dists (Sequence[SequenceEncodableProbabilityDistribution]): Distributions for each component.
        count (int): Number of components (i.e. len(dists)).
        name (Optional[str]): Name of object
        keys (Optional[str]): Key for marking shared parameters

    """

    def __init__(self, dists: Sequence[SequenceEncodableProbabilityDistribution], name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Create an instance of CompositeDistribution.

        Args:
            dists (Sequence[SequenceEncodableProbabilityDistribution]): Component distributions
            name (Optional[str]): Name of object
            keys (Optional[str]): Key for marking shared parameters
        """
        self.dists = tuple(dists)
        self.count = len(dists)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Returns str name of CompositeDistribution with each dist as well."""
        s0 = ','.join(map(str, self.dists))
        s1 = repr(self.name)
        s2 = repr(self.keys)
        return 'CompositeDistribution(dists=[%s], name=%s, keys=%s)' % (s0, s1, s2)

    def density(self, x: Tuple[Any, ...]) -> float:
        """Evaluates density of CompositeDistribution for single observation tuple x.

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k^{th} data type must be consistent with dists[k].

        Returns:
            Density as float.

        """
        rv = 0.0

        for i in range(1, self.count):
            rv *= self.dists[i].density(x[i])

        return rv

    def log_density(self, x: Tuple[Any, ...]) -> float:
        """Evaluates log-density of CompositeDistribution for single observation tuple x.

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k^{th} data type must be consistent with dists[k].

        Returns:
            Log-density as float.

        """
        rv = self.dists[0].log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].log_density(x[i])

        return rv

    def seq_log_density(self, x: 'CompositeEncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation of log density for CompositeEncodedDataSequence.

        Args:
            x (CompositeEncodedDataSequence): EncodedDataSequence for Composite Distribution.

        Returns:
            np.ndarray of log_density evaluated at all encoded data points.

        """
        if not isinstance(x, CompositeEncodedDataSequence):
            raise Exception('CompositeDistribution.seq_log_density() requires CompositeEncodedDataSequence.')
        else:
            rv = self.dists[0].seq_log_density(x.data[0])

            for i in range(1, self.count):
                rv += self.dists[i].seq_log_density(x.data[i])

            return rv

    def sampler(self, seed: Optional[int] = None) -> 'CompositeSampler':
        """Create CompositeSampler for sampling from CompositeDistribution instance.

        Args:
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Returns:
            CompositeSampler object.

        """
        return CompositeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'CompositeEstimator':
        """Create CompositeEstimator for estimating CompositeDistribution.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics in estimation.

        Returns:
            CompositeEstimator

        """
        return CompositeEstimator([d.estimator(pseudo_count=pseudo_count) for d in self.dists], name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'CompositeDataEncoder':
        encoders = tuple([d.dist_to_encoder() for d in self.dists])

        return CompositeDataEncoder(encoders=encoders)


class CompositeSampler(DistributionSampler):
    """CompositeSampler used to generate samples from CompositeDistribution.

    Attributes:
        dist (CompositeDistribution): CompositeDistribution to draw samples from.
        rng (RandomState): RandomState with seed set if provided.
        dist_samplers (List[DistributionSamplers]): List of DistributionSamplers for each component
            (len=len(dists)).
    """

    def __init__(self, dist: 'CompositeDistribution', seed: Optional[int] = None) -> None:
        """CompositeSampler object.

        Args:
            dist (CompositeDistribution): CompositeDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        """
        self.dist = dist
        self.rng = RandomState(seed)
        self.dist_samplers = [d.sampler(seed=self.rng.randint(maxrandint)) for d in dist.dists]

    def sample(self, size: Optional[int] = None) -> Union[List[Tuple[Any, ...]], Tuple[Any, ...]]:
        """Generate independent samples from a CompositeDistribution.

        If size is None, draw one sample and return as Tuple of length = len(dists). If size > 0,
        draw size samples and return a list of length size containing tuples of len(dists).

        Args:
            size (Optional[int]): If None, draw 1 sample. Else, draw size number of iid samples.

        Returns:
            A tuple of length = len(dists) or a list of length size containing tuples of length = len(dists).

        """
        if size is None:
            return tuple([d.sample(size=size) for d in self.dist_samplers])

        else:
            return list(zip(*[d.sample(size=size) for d in self.dist_samplers]))


class CompositeAccumulator(SequenceEncodableStatisticAccumulator):
    """CompositeAccumulator object used for aggregating sufficient statistics of each component of the
        CompositeDistribution.

    Attributes:
        accumulators (List[SequenceEncodableStatisticAccumulator]): List of SequenceEncodableStatisticAccumulator
            objects for accumulating sufficient statsitics for each component of the CompositeDistribution.
        count (int): Length of accumulators.
        keys (Optional[str]): All CompositeAccumulators with same keys will have suff-stats merged.
        name (Optional[str]): Name of the object.
        _init_rng (bool): Is True if _acc_rng has been set by a single function call to initialize.
        _acc_rng (List[RandomState]): List of RandomState objects generated from seeds set by rng in initialize.

    """

    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator], keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """CompositeAccumulator object used for aggregating suffcient statistics of each component of the
            CompositeDistribution.

        Args:
            accumulators (List[SequenceEncodableStatisticAccumulator]):
            keys (Optional[str]): All CompositeAccumulators with same keys will have suff-stats merged.
            name (Optional[str]): Name of the object.

        """
        self.accumulators = accumulators
        self.count = len(accumulators)
        self.key = keys
        self.name = name

        # variables for initialization
        self._init_rng = False
        self._acc_rng: Optional[List[RandomState]] = None

    def update(self, x: Tuple[Any, ...], weight: float, estimate: Optional['CompositeDistribution']) -> None:
        if estimate is not None:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, estimate.dists[i])

        else:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, None)

    def _rng_initialize(self, rng: RandomState) -> None:
        seeds = rng.randint(2 ** 31, size=self.count)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]

    def initialize(self, x: Tuple[Any, ...], weight: float, rng: np.random.RandomState) -> None:
        if not self._init_rng:
            self._rng_initialize(rng)

        for i in range(0, self.count):
            self.accumulators[i].initialize(x[i], weight, self._acc_rng[i])

    def seq_initialize(self, x: 'CompositeEncodedDataSequence', weights: np.ndarray, rng: np.random.RandomState) -> None:
        if not self._init_rng:
            self._rng_initialize(rng)

        for i in range(0, self.count):
            self.accumulators[i].seq_initialize(x.data[i], weights, self._acc_rng[i])

    def get_seq_lambda(self) -> List[Any]:
        rv = []
        for i in range(self.count):
            rv.extend(self.accumulators[i].get_seq_lambda())
        return rv

    def seq_update(self, x: 'CompositeEncodedDataSequence', weights: np.ndarray,
                   estimate: Optional['CompositeDistribution']) -> None:
        for i in range(self.count):
            self.accumulators[i].seq_update(x.data[i], weights, estimate.dists[i] if estimate is not None else None)

    def combine(self, suff_stat: Tuple[Any, ...]) -> 'CompositeAccumulator':

        for i in range(0, self.count):
            self.accumulators[i].combine(suff_stat[i])

        return self

    def value(self) -> Tuple[Any, ...]:

        return tuple([x.value() for x in self.accumulators])

    def from_value(self, x: Tuple[Any, ...]) -> 'CompositeAccumulator':

        self.accumulators = [self.accumulators[i].from_value(x[i]) for i in range(len(x))]
        self.count = len(x)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:

        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'CompositeDataEncoder':

        encoders = tuple([acc.acc_to_encoder() for acc in self.accumulators])

        return CompositeDataEncoder(encoders=encoders)


class CompositeAccumulatorFactory(StatisticAccumulatorFactory):
    """CompositeAccumulatorFactory object.

    Attributes:
        factories (List[StatisticAccumulatorFactory]): List of StatisticAccumulatorFactory objects for each
            component.
        keys (Optional[str]): Declare keys for merging sufficient statistics of CompositeAccumulator objects.
        name (Optional[str]): Name of the object.

    """

    def __init__(self, factories: Sequence[StatisticAccumulatorFactory], keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """CompositeAccumulatorFactory object.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): List of StatisticAccumulatorFactory objects for each
                component.
            keys (Optional[str]): Declare keys for merging sufficient statistics of CompositeAccumulator objects.
            name (Optional[str]): Name of the object.

        """
        self.factories = factories
        self.keys = keys
        self.name = name

    def make(self) -> 'CompositeAccumulator':
        return CompositeAccumulator([u.make() for u in self.factories], keys=self.keys, name=self.name)


class CompositeEstimator(ParameterEstimator):
    """CompositeEstimator object used to estimate CompositeDistribution.

    Attributes:
        estimators (List[ParameterEstimator]): List of ParameterEstimator objects for each component of
            CompositeEstimator.
        keys (Optional[str]): Keys used for merging sufficient statistics of CompositeEstimator objects.
        count (int): Number of components in CompositeEstimator.
        name (Optional[str]): Name of the object. 

    """

    def __init__(self, estimators: Sequence[ParameterEstimator], keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """CompositeEstimator object.

        Args:
            estimators (List[ParameterEstimator]): List of ParameterEstimator objects for each component of
                CompositeEstimator.
            keys (Optional[str]): Keys used for merging sufficient statistics of CompositeEstimator objects.
            name (Optional[str]): Name of the object.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("CompositeEstimator requires keys to be of type 'str'.")

        self.estimators = estimators
        self.count = len(estimators)
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'CompositeAccumulatorFactory':

        return CompositeAccumulatorFactory([u.accumulator_factory() for u in self.estimators], keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[Any, ...]) -> 'CompositeDistribution':
        """Estimate a CompositeDistribution from an aggregated sufficient statistics Tuple for a given number of
            observations (nobs).

        Args:
            nobs (Optional[float]): Weighted number of observations used to form suff_stat.
            suff_stat (Tuple[Any, ...]): Tuple of sufficient statistics for each ParameterEstimator of estimators.

        Returns:
            CompositeDistribution: Estimated from argument aggregated sufficient statistics (suff_stat), from a given
                number of observation (nobs).

        """
        return CompositeDistribution(tuple([est.estimate(nobs, ss) for est, ss in zip(self.estimators, suff_stat)]))


class CompositeDataEncoder(DataSequenceEncoder):
    """CompositeDataEncoder used for creating CompositeDataSequence.

    Data must be of form Sequence[Tuple[Any,...]]. Each encoder component must be compatible with each data
        component of the data.

    Attributes:
        encoders (Sequence[DataSequenceEncoder]): DataSequenceEncoders for each component of the
            CompositeDistribution.

    """

    def __init__(self, encoders: Sequence[DataSequenceEncoder]) -> None:
        """CompositeDataEncoder used for creating CompositeDataSequence.

        Args:
            encoders (Sequence[DataSequenceEncoder]): DataSequenceEncoders for each component of the
                CompositeDistribution.

        """
        self.encoders = encoders

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeDataEncoder):
            return False

        else:

            for i, encoder in enumerate(self.encoders):
                if not encoder == other.encoders[i]:
                    return False

        return True

    def __str__(self) -> str:
        s = 'CompositeDataEncoder(['

        for d in self.encoders[:-1]:
            s += str(d) + ','

        s += str(self.encoders[-1]) + '])'

        return s

    def seq_encode(self, x: Sequence[Tuple[Any, ...]]) -> 'CompositeEncodedDataSequence':
        """Encode Sequence of tuples of data for use with vectorized "seq_" functions.

        The input x must be a Sequence of Tuples of length equal to the length of encoders. Each component tuple
        observation of x, say x[i], must be component-wise compatible with encoders.

        Args:
            x (Sequence[Tuple[Any, ...]]): Sequence of tuples of length equal to len(encoders).

        Returns:
            CompositeEncodedDataSequence

        """
        enc_data = []

        for i, encoder in enumerate(self.encoders):
            enc_data.append(encoder.seq_encode([u[i] for u in x]))

        return CompositeEncodedDataSequence(data=tuple(enc_data))


class CompositeEncodedDataSequence(EncodedDataSequence):
    """CompositeDataSequence object.

    Data must be of form Sequence[Tuple[Any,...]]. Each encoder component must be compatible with each data
        component of the data.

    Attributes:
        data (Tuple[EncodedDataSequences, ...]): Tuple of EncodedDataSequences.

    """

    def __init__(self, data: Tuple[EncodedDataSequence, ...]):
        super().__init__(data)

    def __repr__(self) -> str:
        return f'CompositeEncodedDataSequence(data={self.data})'


