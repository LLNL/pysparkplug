"""Create, estimate, and sample from a Conditional distribution.

Defines the ConditionalDistribution, ConditionalDistributionSampler, ConditionalDistributionAccumulatorFactory,
ConditionalDistributionAccumulator, ConditionalDistributionEstimator, and the ConditionalDistributionDataEncoder
classes for use with pysparkplug.

Data type: (Tuple[T0, T1]): The ConditionalDistribution if given by density,
    P(X0,X1) = P_cond(X1|X0)*P_given(X0).

The ConditionalDistribution allows for user defined conditional distributions P_cond(X1|X0), and given distributions
P_given(X0).

"""
import numpy as np
import math
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, ConditionalSampler, \
    EncodedDataSequence
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullDataEncoder, NullAccumulatorFactory, \
    NullEstimator
from typing import Optional, List, Union, Any, Tuple, Sequence, TypeVar, Dict
from pysp.arithmetic import maxrandint
from numpy.random import RandomState

T0 = TypeVar('T0')
T1 = TypeVar('T1')

E0 = TypeVar('E0')
E1 = TypeVar('E1')
E = Tuple[int, Tuple[T0, ...], Tuple[np.ndarray, ...], Tuple[E0, ...], Optional[E1]]
SS0 = TypeVar('SS0')
SS1 = TypeVar('SS1')
SS2 = TypeVar('SS2')


class ConditionalDistribution(SequenceEncodableProbabilityDistribution):
    """ConditionalDistribution object for data types x=Tuple[T0, T1].

    Attributes:
        dmap (Dict[T0, SequenceEncodableProbabilityDistribution]): T0 is integer if dmap arg was list, else T0 is
            data type of the "given" or conditional.
        default_dist (SequenceEncodableProbabilityDistribution): Set to NullDistribution if None is passed as arg.
        given_dist (SequenceEncodableProbabilityDistribution): Set to NullDistribution if None is passed as arg.
        has_default (bool): True if default distribution is not NullDistribution, else False.
        has_given (bool): True if given_dist is not NullDistribution, else False.
        name (Optional[str]): Name assigned to object.
        keys (Optional[str]): All ConditionalDistribution objects with same keys value are the same distribution.

    """

    def __init__(self,
                 dmap: Union[Dict[Any, SequenceEncodableProbabilityDistribution],
                             List[SequenceEncodableProbabilityDistribution]],
                 default_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 given_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """ConditionalDistribution object.

        Args:
            dmap (Union[Dict[Any, SequenceEncodableProbabilityDistribution], List[SequenceEncodableProbabilityDistribution]]): Used to create dictionary of SequenceEncodableProbabilityDistribution objects.
            default_dist (Optional[SequenceEncodableProbabilityDistribution]): Defines the distribution for the case where x[0] is not a key in dmap
            given_dist (Optional[SequenceEncodableProbabilityDistribution]): p_mat(x[0]) is defined as the given distribution.
            name (Optional[str]): Name assigned to object.
            keys (Optional[str]): All ConditionalDistribution objects with same keys value are the same distribution.

        """
        if isinstance(dmap, list):
            dmap = dict(zip(range(len(dmap)), dmap))

        self.dmap = dmap
        self.default_dist = default_dist if default_dist is not None else NullDistribution()
        self.given_dist = given_dist if given_dist is not None else NullDistribution()

        self.has_default = not isinstance(self.default_dist, NullDistribution)
        self.has_given = not isinstance(self.given_dist, NullDistribution)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        s1 = repr(self.dmap)
        s2 = repr(self.default_dist)
        s3 = repr(self.given_dist)
        s4 = repr(self.name)
        s5 = repr(self.keys)

        return 'ConditionalDistribution(%s, default_dist=%s, given_dist=%s, name=%s, keys=%s)' % (s1, s2, s3, s4, s5)

    def density(self, x: Tuple[T0, T1]) -> float:
        """Evaluates density of ConditionalDistribution at Tuple x.

        Args:
            x (Tuple[T0, T1]): T0 data type much match keys of dmap, T1 much match value of dmap distribution for key
                value.

        Returns:
            float: Density of ConditionalDistribution at Tuple x

        """
        return math.exp(self.log_density(x))

    def log_density(self, x: Tuple[T0, T1]) -> float:
        """Evaluate log-density of ConditionalDistribution at Tuple x.

        Args:
            x (Tuple[T0, T1]): T0 data type much match keys of dmap, T1 much match value of dmap distribution for key
                value.

        Returns:
            float: Log-density of ConditionalDistribution at Tuple x.

        """
        if self.has_default:
            rv = self.dmap.get(x[0], self.default_dist).log_density(x[1])
        else:
            if x[0] in self.dmap:
                rv = self.dmap[x[0]].log_density(x[1])
            else:
                return -np.inf

        rv += self.given_dist.log_density(x[0])

        return rv

    def seq_log_density(self, x: 'ConditionalEncodedDataSequence') -> np.ndarray:
        if not isinstance(x, ConditionalEncodedDataSequence):
            raise Exception('Requires ConditionalEncodedDataSequence for ConditionalDistribution.seq_log_density()')
        sz, cond_vals, eobs_vals, idx_vals, given_enc = x.data
        rv = np.zeros(sz, dtype=float)

        for i in range(len(cond_vals)):
            if self.has_default:
                rv[idx_vals[i]] = self.dmap.get(cond_vals[i], self.default_dist).seq_log_density(eobs_vals[i])
            else:
                if cond_vals[i] in self.dmap:
                    rv[idx_vals[i]] += self.dmap[cond_vals[i]].seq_log_density(eobs_vals[i])

        if self.has_given:
            rv += self.given_dist.seq_log_density(given_enc)

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'ConditionalDistributionSampler':
        """Creates ConditionalDistributionSampler object for sampling from ConditionalDistribution instance.

        Args:
            seed (Optional[int]): Set seed for sampling from ConditionalDistributionSampler object.

        Returns:
            ConditionalDistributionSampler object.

        """
        return ConditionalDistributionSampler(self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> "ConditionalDistributionEstimator":
        """Creates ConditionalDistributionEstimator object from sufficient statistics of ConditionalDistribution object.

        Used to estimate a ConditionalDistribution from data observations.

        Args:
            pseudo_count (Optional[float]): Used to inflate the sufficient statistics of ConditionalDistribution.

        Returns:
            ConditionalDistributionEstimator object.

        """
        est_map = {k: v.estimator(pseudo_count) for k, v in self.dmap.items()}
        default_est = self.default_dist.estimator(pseudo_count)
        given_est = self.given_dist.estimator(pseudo_count)

        return ConditionalDistributionEstimator(estimator_map=est_map,
                                                default_estimator=default_est,
                                                given_estimator=given_est,
                                                name=self.name,
                                                keys=self.keys)

    def dist_to_encoder(self) -> 'ConditionalDistributionDataEncoder':
        encoder_map = {k: v.dist_to_encoder() for k, v in self.dmap.items()}
        default_encoder = NullDataEncoder() if not self.has_default else self.default_dist.dist_to_encoder()
        given_encoder = NullDataEncoder() if not self.has_given else self.given_dist.dist_to_encoder()

        return ConditionalDistributionDataEncoder(encoder_map=encoder_map,
                                                  default_encoder=default_encoder,
                                                  given_encoder=given_encoder)


class ConditionalDistributionSampler(ConditionalSampler, DistributionSampler):
    """ConditionalDistributionSampler object samples from ConditionalDistribution either directly or conditionally.

    Attributes:
        dist (ConditionalDistribution): ConditionalDistribution object to draw samples from.
        default_sampler (DistributionSampler): DistributionSampler object for sampling from default_dist of
            ConditionalDistribution.
        has_default_sampler (bool): True if default sampler is not NullDistribution, else False.
        given_sampler (DistributionSampler): DistributionSampler object for sampling from given_dist of
            ConditionalDistribution.
        has_given_sampler (bool): True if given sampler is not NullDistribution, else False.
        samplers (Dict[T0,DistributionSampler]): Dictionary of samplers for sampling from ConditionalDistribution,
            given a key of data type T0. Note returns List[T1] or T1.

    """

    def __init__(self, dist: ConditionalDistribution, seed: Optional[int] = None) -> None:
        """ConditionalDistributionSampler object.

        Args:
            dist (ConditionalDistribution): ConditionalDistribution object to draw samples from.
            seed (Optional[int]): Used to set the seed of random number generator used in sampling.

        """
        self.dist = dist
        rng = np.random.RandomState(seed)

        loc_seed = rng.randint(0, maxrandint)

        self.has_default_sampler = dist.has_default
        self.default_sampler = dist.default_dist.sampler(loc_seed)

        loc_seed = rng.randint(0, maxrandint)
        self.given_sampler = dist.given_dist.sampler(loc_seed)
        self.has_given_sampler = isinstance(dist.given_dist, NullDistribution)

        self.samplers = {k: u.sampler(rng.randint(0, maxrandint)) for k, u in self.dist.dmap.items()}

    def single_sample(self) -> Tuple[Any, Any]:
        """Generates a simple sample from the ConditionalDistribution.

        Returns Tuple of T0 and T1, where T1 is the data type of the conditional distribution, and T0 is the type of
        the given distribution.

        Returns:
            Tuple[Any, Any]: as defined from dmap and given_distribution types in dist (ConditionalDistribution instance).

        """
        x0 = self.given_sampler.sample()
        if x0 in self.samplers:
            x1 = self.samplers[x0].sample()
        else:
            x1 = self.default_sampler.sample()
        return x0, x1

    def sample(self, size: Optional[int] = None) -> Union[Tuple[Any, Any], List[Tuple[Any, Any]]]:
        """Sample 'size' independent samples from ConditionalDistribution.

        Sequence of 'size' calls to single_sample(). If size is None, size is taken to be 1.

        Data type returned is a Tuple[T0, T1], where T0 and T1 are the respective data types of the given_dist and
        dmap defined in the CompositeDistribution instance 'dist'.

        Args:
            size (Optional[int]): Number of independent samples to draw from ConditionalDistribution.

        Returns:
            A list of 'size' tuples of Tuple[T0, T1], or a single Tuple[T0, T1].

        """

        if size is None:
            return self.single_sample()
        else:
            return [self.single_sample() for i in range(size)]

    def sample_given(self, x: T0) -> Any:
        """Sample from conditional distribution of ConditionalDistribution object with given value x.

        Return data type T1 as defined for dictionary of ConditionalDistribution instance.

        Args:
            x (T0): Value of given/conditional value for ConditionalDistribution.

        Returns:
            Single sample from ConditionalDistribution object 'dist.dmap' given x.

        """
        if x in self.samplers:
            return self.samplers[x].sample()

        elif self.has_default_sampler:
            return self.default_sampler.sample()

        else:
            raise Exception('Conditional default distribution unspecified.')

class ConditionalDistributionAccumulator(SequenceEncodableStatisticAccumulator):
    """ConditionalDistributionAccumulator used for aggregating sufficient statistics of ConditionalDistribution.

    The sufficient statistics are defined through the accumulator_map dictionary, which is a dictionary with keys
    of data type T0 for the given type. Each value of the dict contains a SequenceEncodableStatisticAccumulator for
    accumulating respective sufficient statistics.

    The sufficient statistics for the default_distribution are stored in SequenceEncodableStatisticAccumulator
    obejct defualt_accumulator. If default_accumulator = None, default_accumulator is set to NullAccumualtor().

    The sufficient statistics for given_distribution are stored in given_accumulator. This is set to
    NullAccumulator() if no given_accumulator is specified.

    Attributes:
        accumulator_map (Dict[T0, SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of each
            conditional distribution for a given key value of data type T0.
        default_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of
            distribution for case where key not in accumulator_map.
        given_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of
            given distribution if provided.
        has_default (bool): True if default_accumulator is not NullAccumulator.
        has_given (bool): True if given_accumulator is not NullAccumulator.
        key (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.
        name (Optional[str]): Name for object
        _init_rng (bool): False unless a single call to initialize or seq_initialize has been made.
        _acc_rng (Optional[Dict[T0, RandomState]]): Used to seed RandomState calls of accumulator_map.
        _default_rng (Optional[RandomState]): Used to seed RandomState calls of defualt_accumulator initialize.
        _given_rng (Optional[RandomState]): Used to seed RandomState calls of given_accumulator initialize.

    """

    def __init__(self,
                 accumulator_map: Dict[T0, SequenceEncodableStatisticAccumulator],
                 default_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 given_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """ConditionalDistributionAccumulator object.

        Args:
            accumulator_map (Dict[T0, SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of each
                conditional distribution for a given key value of data type T0.
            default_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of
                distribution for case where key not in accumulator_map.
            given_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Stores sufficient statistics of
                given distribution if provided.
            name (Optional[str]): Name for object
            keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

        """

        self.accumulator_map = accumulator_map
        self.default_accumulator = default_accumulator if default_accumulator is not None else NullAccumulator()
        self.given_accumulator = given_accumulator if given_accumulator is not None else NullAccumulator()

        self.has_default = not isinstance(default_accumulator, NullAccumulator)
        self.has_given = not isinstance(given_accumulator, NullAccumulator)
        self.name = name 
        self.key = keys

        # seeds for intializers
        self._init_rng = False
        self._acc_rng: Optional[Dict[T0, RandomState]] = None
        self._default_rng: Optional[RandomState] = None
        self._given_rng: Optional[RandomState] = None

    def update(self, x: Tuple[T0, T1], weight: float, estimate: Optional['ConditionalDistribution']) -> None:

        if x[0] in self.accumulator_map:
            if estimate is None:
                self.accumulator_map[x[0]].update(x[1], weight, None)
            else:
                self.accumulator_map[x[0]].update(x[1], weight, estimate.dmap[x[0]])
        else:
            if self.has_default:
                if estimate is None:
                    self.default_accumulator.update(x[1], weight, None)
                else:
                    self.default_accumulator.update(x[1], weight, estimate.default_dist)

        if self.has_given:
            if estimate is None:
                self.given_accumulator.update(x[0], weight, None)
            else:
                self.given_accumulator.update(x[0], weight, estimate.given_dist)

    def _rng_initialize(self, rng: RandomState) -> None:

        self._acc_rng = dict()
        for acc_key in self.accumulator_map.keys():
            self._acc_rng[acc_key] = RandomState(seed=rng.randint(2**31))

        self._default_rng = RandomState(seed=rng.randint(2 ** 31))
        self._given_rng = RandomState(seed=rng.randint(2 ** 31))

    def initialize(self, x: Tuple[T0, T1], weight: float, rng: RandomState) -> None:

        if not self._init_rng:
            self._rng_initialize(rng)

        if x[0] in self.accumulator_map:
            self.accumulator_map[x[0]].initialize(x[1], weight, self._acc_rng[x[0]])
        else:
            if self.has_default:
                self.default_accumulator.initialize(x[1], weight, self._default_rng)

        if self.has_given:
            self.given_accumulator.initialize(x[0], weight, self._given_rng)

    def seq_initialize(self, x: 'ConditionalEncodedDataSequence', weights: np.ndarray, rng: RandomState) -> None:

        sz, cond_vals, eobs_vals, idx_vals, given_enc = x.data

        if not self._init_rng:
            self._rng_initialize(rng)

        for i in range(len(cond_vals)):
            if cond_vals[i] in self.accumulator_map:
                self.accumulator_map[cond_vals[i]].seq_initialize(eobs_vals[i], weights[idx_vals[i]],
                                                                  self._acc_rng[cond_vals[i]])
            else:
                if self.has_default:
                    self.default_accumulator.seq_initialize(eobs_vals[i], weights[idx_vals[i]], self._default_rng)

        if self.has_given:
            self.given_accumulator.seq_initialize(given_enc, weights, self._given_rng)

    def seq_update(self, x: 'ConditionalEncodedDataSequence', weights: np.ndarray, estimate: 'ConditionalDistribution') -> None:

        sz, cond_vals, eobs_vals, idx_vals, given_enc = x.data

        for i in range(len(cond_vals)):
            if cond_vals[i] in self.accumulator_map:
                self.accumulator_map[cond_vals[i]].seq_update(eobs_vals[i], weights[idx_vals[i]],
                                                              estimate.dmap[cond_vals[i]])
            else:
                if self.has_default:
                    if estimate is None:
                        self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], None)
                    else:
                        self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], estimate.default_dist)

        if self.has_given:
            if estimate is None:
                self.given_accumulator.seq_update(given_enc, weights, None)
            else:
                self.given_accumulator.seq_update(given_enc, weights, estimate.given_dist)

    def combine(self, suff_stat: Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS2]]) \
            -> 'ConditionalDistributionAccumulator':
        for k, v in suff_stat[0].items():
            if k in self.accumulator_map:
                self.accumulator_map[k].combine(v)
            else:
                self.accumulator_map[k].from_value(v)

        if self.has_default and suff_stat[1] is not None:
            self.default_accumulator.combine(suff_stat[1])

        if self.has_given and suff_stat[2] is not None:
            self.given_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[Dict[Any, Any], Optional[Any], Optional[Any]]:
        rv3 = self.given_accumulator.value()
        rv2 = self.default_accumulator.value()
        rv1 = {k: v.value() for k, v in self.accumulator_map.items()}

        return rv1, rv2, rv3

    def from_value(self, x: Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS1]]) -> 'ConditionalDistributionAccumulator':

        for k, v in x[0].items():
            self.accumulator_map[k].from_value(v)

        if self.has_default and x[1] is not None:
            self.default_accumulator.from_value(x[1])

        if self.has_given and x[2] is not None:
            self.given_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        for k, v in self.accumulator_map.items():
            v.key_merge(stats_dict)

        if self.has_default:
            self.default_accumulator.key_merge(stats_dict)

        if self.has_given:
            self.given_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        for k, v in self.accumulator_map.items():
            v.key_replace(stats_dict)

        if self.has_default:
            self.default_accumulator.key_replace(stats_dict)

        if self.has_given:
            self.given_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'ConditionalDistributionDataEncoder':

        encoder_map = {k: v.acc_to_encoder() for k, v in self.accumulator_map.items()}
        default_encoder = self.default_accumulator.acc_to_encoder()
        given_encoder = self.given_accumulator.acc_to_encoder()

        return ConditionalDistributionDataEncoder(encoder_map=encoder_map,
                                                  default_encoder=default_encoder,
                                                  given_encoder=given_encoder)


class ConditionalDistributionAccumulatorFactory(StatisticAccumulatorFactory):
    """ConditionalDistributionAccumulatorFactory object.

    Attributes:
        factory_map (Dict[T0, StatisticAccumulatorFactory]): Dictionary of StatisticAccumulatorFactory objects for
            creating SequenceEncodableStatisticAccumulator objects in ConditionalDistributionAccumulator
        default_factory (StatisticAccumulatorFactory): Used to create SequenceEncodableStatisticAccumulator for
            defualt_accumulator in ConditionalDistributionAccumulator.
        given_factory (StatisticAccumulatorFactory): Used to create SequenceEncodableStatisticAccumulator for
            given_accumulator in ConditionalDistributionAccumulator.
        name (Optional[str]): Name for object
        keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

    """

    def __init__(self,
                 factory_map: Dict[T0, StatisticAccumulatorFactory],
                 default_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
                 given_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """ConditionalDistributionAccumulatorFactory object.

        Args:
            factory_map (Dict[T0, StatisticAccumulatorFactory]): Dictionary of StatisticAccumulatorFactory objects for
                creating SequenceEncodableStatisticAccumulator objects in ConditionalDistributionAccumulator
            default_factory (StatisticAccumulatorFactory): Used to create SequenceEncodableStatisticAccumulator for
                defualt_accumulator in ConditionalDistributionAccumulator.
            given_factory (StatisticAccumulatorFactory): Used to create SequenceEncodableStatisticAccumulator for
                given_accumulator in ConditionalDistributionAccumulator.
            name (Optional[str]): Name for object
            keys (Optional[str]): All ConditionalAccumulator objects with same keys value will merge suff stats.

        """
        self.factory_map = factory_map
        self.default_factory = default_factory
        self.given_factory = given_factory
        self.name = name
        self.keys = keys

    def make(self) -> 'ConditionalDistributionAccumulator':

        acc = {k: v.make() for k, v in self.factory_map.items()}
        def_acc = self.default_factory.make()
        given_acc = self.given_factory.make()

        return ConditionalDistributionAccumulator(
            accumulator_map=acc, 
            default_accumulator=def_acc, 
            given_accumulator=given_acc, 
            keys=self.keys,
            name=self.name
            )


class ConditionalDistributionEstimator(ParameterEstimator):
    """ConditionalDistributionEstimator object used to estimate ConditionalDistribution from aggregated data.

    If None is passed for default_estimator, default_estimator is set to NullEstimator().
    If None is passed for given_estimator, given_estimator is set to NullEstimator().

    Attributes:
        estimator_map (Dict[T0, ParameterEstimator]):
        default_estimator (ParameterEstimator): ParameterEstimator for default_distribution set to NullEstimator,
            if None is passed as arg.
        given_estimator (ParameterEstimator): ParameterEstimator for given_distribution set to NullEstimator
            if None is passed as arg.
        name (Optional[str]): Name the ConditionalDistributionEstimator object.
        keys (Optional[str]): ConditionalDistributionEstimator with matching 'keys' will be aggregated.

    """

    def __init__(self,
                 estimator_map: Dict[T0, ParameterEstimator],
                 default_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 given_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """ConditionalDistributionEstimator object.

        Args:
            estimator_map (Dict[T0, ParameterEstimator]):
            default_estimator (Optional[ParameterEstimator]): ParameterEstimator for default_distribution, can be None.
            given_estimator (Optional[ParameterEstimator]): ParameterEstimator for given_distribution, can be None.
            name (Optional[str]): Name the ConditionalDistributionEstimator object.
            keys (Optional[str]): ConditionalDistributionEstimator with matching 'keys' will be aggregated.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("ConditionalDistributionEstimator requires keys to be of type 'str'.")

        self.estimator_map = estimator_map
        self.default_estimator = default_estimator if default_estimator is not None else NullEstimator()
        self.given_estimator = given_estimator if given_estimator is not None else NullEstimator()
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'ConditionalDistributionAccumulatorFactory':

        emap_items = {k: v.accumulator_factory() for k, v in self.estimator_map.items()}
        def_factory = self.default_estimator.accumulator_factory()
        given_factory = self.given_estimator.accumulator_factory()

        return ConditionalDistributionAccumulatorFactory(
            factory_map=emap_items, 
            default_factory=def_factory, 
            given_factory=given_factory, 
            keys=self.keys,
            name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[Dict[T0, SS0], Optional[SS1], Optional[SS2]]) \
            -> 'ConditionalDistribution':
        """Estimate a ConditionalDistribution from aggregated data.

        Calls the estimate() member function of each ParameterEstimator instance for estimator_map, default_estimator,
        and given_estimator.

        Input suff_stat if a Tuple of size three containing sufficient statistics compatible with each respective
        ParameterEstimator. Entry one of the Tuple must be a dict with keys of data type T0, matching the data type
        for the given distribution.

        Returns a ConditionalDistribution object estimated from the sufficient statistics in suff_stat.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency.
            suff_stat: See description above.

        Returns:
            ConditionalDistribution object.

        """
        default_dist = self.default_estimator.estimate(None, suff_stat[1])
        given_dist = self.given_estimator.estimate(None, suff_stat[2])
        dist_map = {k: self.estimator_map[k].estimate(None, v) for k, v in suff_stat[0].items()}

        return ConditionalDistribution(dist_map, default_dist=default_dist, given_dist=given_dist, name=self.name,
                                       keys=self.keys)


class ConditionalDistributionDataEncoder(DataSequenceEncoder):
    """ConditionalDistributionDataEncoder used to encode sequence of data.

    Data type should be Tuple[T0, T1] where T0 is the type of the conditional value in ConditionalDistribution.
    I.e.,
    p_mat(X1|X0), should have x_mat as type T0, and Y as type T1.

    Attributes:
        encoder_map (Dict[T0, DataSequenceEncoder]): Dictionary of DataSequenceEncoder objects for each conditional
            value of data type T0. Data types of the encoders must be of type T1.
        default_encoder (DataSequenceEncoder): DataSequenceEncoder compatible with data type T1.
        given_encoder (DataSequenceEncoder): DataSequenceEncoder compatible with data type T0.
        null_default_encoder (bool): True if default_encoder is instance of NullDataEncoder, else false.
        null_given_encoder (bool): True if default_encoder is instance of NullDataEncoder, else false.

    """

    def __init__(self,
                 encoder_map: Dict[T0, DataSequenceEncoder],
                 default_encoder: DataSequenceEncoder = NullDataEncoder(),
                 given_encoder: DataSequenceEncoder = NullDataEncoder()
                 ) -> None:
        """ConditionalDistributionDataEncoder object.

        Args:
            encoder_map (Dict[T0, DataSequenceEncoder]): Dictionary of DataSequenceEncoder objects for each conditional
                value of data type T0. Data types of the encoders must be of type T1.
            default_encoder (DataSequenceEncoder): DataSequenceEncoder compatible with data type T1.
            given_encoder ((DataSequenceEncoder): DataSequenceEncoder compatible with data type T0.

        """
        self.encoder_map = encoder_map
        self.default_encoder = default_encoder
        self.given_encoder = given_encoder

        self.null_default_encoder = isinstance(self.default_encoder, NullDataEncoder)
        self.null_given_encoder = isinstance(self.given_encoder, NullDataEncoder)

    def __str__(self) -> str:

        encoder_items = list(self.encoder_map.items())
        encoder_str = 'ConditionalDataEncoder('
        for k, v in encoder_items[:-1]:
            encoder_str += str(k) + ':' + str(v) + ','
        encoder_str += str(encoder_items[-1][0]) + ':' + str(encoder_items[-1][1])

        if not self.null_default_encoder:
            encoder_str += ',default=' + str(self.default_encoder)
        else:
            encoder_str += ',default=None'

        if not self.null_given_encoder:
            encoder_str += ',given=' + str(self.given_encoder)
        else:
            encoder_str += ',given=None)'

        return encoder_str

    def __eq__(self, other) -> bool:

        if not isinstance(other, ConditionalDistributionDataEncoder):
            return False
        else:
            if not self.encoder_map == other.encoder_map:
                return False

            if not self.default_encoder == other.default_encoder:
                return False

            if not self.given_encoder == other.given_encoder:
                return False

        return True

    def seq_encode(self, x: List[Tuple[T0, T1]]) -> 'ConditionalEncodedDataSequence':
        """Create ConditionalEncodedDataSequence object for vectorized "seq_" function calls.

        Data must be a List of Tuple of two types, T0 and T1. T0 is the data type compatible with the conditional
        values of the ConditionalDistribution. T1 must be consistent with the data type of the conditional
        distributions.

        E Tuple of length 5:
            E[0] (int): length of x (i.e. total observations).
            E[1] (Tuple[T0]): Unique conditional values in data.
            E[2] (Tuple[Encoded[T1]): Tuple of sequence encoded data of type T1 encoded by
                encoder_map[key] or default_encoder if key not in default_encoder and default_encoder is not
                the NullDataEncoder.
            E[3] (Tuple[np.ndarray,...]): Tuple of length equal to the number of unique conditional
                values encountered in the data. Each entry contains a numpy array for the indices of x that correspond
                to a unique conditional value.
            E[4] (Optional[Encoded[T0]]): If the given_encoder is not the NullDataEncoder, the
                observed conditional values of data type T0 are sequence encoded by given_encoder. Else return None.

        Args:
            x (List[Tuple[T0, T1]]): List of data observations.

        Returns:
            ConditionalEncodedDataSequence object.

        """
        cond_enc = dict()

        given_vals = []

        for i in range(len(x)):
            xx = x[i]
            given_vals.append(xx[0])
            if xx[0] not in cond_enc:
                cond_enc[xx[0]] = [[xx[1]], [i]]
            else:
                cond_enc_loc = cond_enc[xx[0]]
                cond_enc_loc[0].append(xx[1])
                cond_enc_loc[1].append(i)

        cond_enc_items = list(cond_enc.items())
        cond_vals = tuple([u[0] for u in cond_enc_items])

        eobs_vals = []
        idx_vals = []

        for u in cond_enc_items:
            if self.null_default_encoder:
                if u[0] in self.encoder_map:
                    eobs_vals.append(self.encoder_map[u[0]].seq_encode(u[1][0]))
            else:
                eobs_vals.append(self.encoder_map.get(u[0], self.default_encoder).seq_encode(u[1][0]))

            idx_vals.append(np.asarray(u[1][1]))

        given_enc = self.given_encoder.seq_encode(given_vals)

        return ConditionalEncodedDataSequence(data=(len(x), cond_vals, tuple(eobs_vals), tuple(idx_vals), given_enc))

class ConditionalEncodedDataSequence(EncodedDataSequence):
    """ConditionalEncodedDataSequence object.

    Data tuple of length 5:
            E[0] (int): length of x (i.e. total observations).
            E[1] (Tuple[T0]): Unique conditional values in data.
            E[2] (Tuple[Encoded[T1]): Tuple of sequence encoded data of type T1 encoded by
                encoder_map[key] or default_encoder if key not in default_encoder and default_encoder is not
                the NullDataEncoder.
            E[3] (Tuple[np.ndarray,...]): Tuple of length equal to the number of unique conditional
                values encountered in the data. Each entry contains a numpy array for the indices of x that correspond
                to a unique conditional value.
            E[4] (Optional[Encoded[T0]]): If the given_encoder is not the NullDataEncoder, the
                observed conditional values of data type T0 are sequence encoded by given_encoder. Else return None.

    Attributes:
        data (Tuple[int, Tuple[T0, ...], Tuple[EncodedDataSequence], Tuple[np.ndarray], Optional[EncodedDataSequence]]): see above.

    """

    def __init__(self, data: Tuple[int, Tuple[T0, ...], Tuple[EncodedDataSequence], Tuple[np.ndarray], Optional[EncodedDataSequence]]):
        """ConditionalEncodedDataSequence object.

        Args:
            data (Tuple[int, Tuple[T0, ...], Tuple[EncodedDataSequence], Tuple[np.ndarray], Optional[EncodedDataSequence]]): see above.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'ConditionalEncodedDataSequence(data={self.data})'



