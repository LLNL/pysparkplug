"""Create, estimate, and sample from an Optional distribution.

Defines the OptionalDistribution, OptionalSampler, OptionalAccumulatorFactory, OptionalAccumulator,
OptionalEstimator, and the OptionalDataEncoder classes for use with DMLearn.

This distribution assigns a probability (p) to data being missing. With probability (1-p) the data is assumed to come
from a base distribution set by the user.

The OptionalDistribution allows for potentially missing data. The value p (the probability of being missing)
must be specified to sample from the distribution.

"""
from dml.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory, EncodedDataSequence
import numpy as np
from numpy.random import RandomState

from typing import Optional, Any, Tuple, Dict, TypeVar, Sequence, List

T = TypeVar('T')
E = TypeVar('E')
SS = TypeVar('SS')


class OptionalDistribution(SequenceEncodableProbabilityDistribution):
    """OptionalDistribution for handling missing values in estimation.

    Attributes:
        dist (SequenceEncodableProbabilityDistribution): Base distribution.
        p (float): Probability that dist has missing_value.
        has_p (bool): True if distribution has arg p passed.
        log_p (float): log of p.
        log_pn (float): log(1-p).
        missing_value_is_nan (bool): True if the missing value is nan.
        missing_value (Any): Missing value from dist.
        name (Optional[str]): Set a name for the object instance.
        keys (Optional[str]): Keys for parameters. 

    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution, p: Optional[float] = None,
                 missing_value: Any = None, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """OptionalDistribution object.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Base distribution.
            p (Optional[float]): Probability that dist has missing_value.
            missing_value (Any): Missing value from dist.
            name (Optional[str]): Set a name for the object instance.
            keys (Optional[str]): Keys for parameters. 

        """
        self.dist = dist
        self.p = p if p is not None else 0.0
        self.has_p = p is not None
        self.log_p = -np.inf if self.p == 0 else np.log(self.p)
        self.log_pn = -np.inf if self.p == 1 else np.log1p(-self.p)

        self.missing_value_is_nan = isinstance(missing_value, (np.floating, float)) and np.isnan(missing_value)
        self.log1_p = np.log1p(self.p)
        self.missing_value = missing_value
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        s1 = str(self.dist)
        s2 = repr(None if not self.has_p else self.p)
        if self.missing_value_is_nan:
            s3 = 'float("nan")'
        else:
            s3 = repr(self.missing_value)
        s4 = repr(self.name)
        s5 = repr(self.keys)
        return 'OptionalDistribution(%s, p=%s, missing_value=%s, name=%s, keys=%s)' % (s1, s2, s3, s4, s5)

    def density(self, x: T) -> float:
        """Evaluate the density of the Optional distribution at x.

        Notes:
            See log_density().

        Args:
            x (T): Observation from base dist or missing value.

        Returns:
            float: Log-density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: T) -> float:
        """Evaluate the log density of the Optional distribution at x.

        Notes:
            If x is a missing value: return log(p) if p is not None, else return 0.0
            If x is not the missing_value: if p is not None, return the log_density(x) at base dist + log(1-p) else:
            return log_density(x).

        Args:
            x (T): Observation from base dist or missing value.

        Returns:
            float: Log-density at x.

        """
        if self.missing_value_is_nan:
            if isinstance(x, (np.floating, float)) and np.isnan(x):
                not_missing = False
            else:
                not_missing = True
        else:
            if x == self.missing_value:
                not_missing = False
            else:
                not_missing = True

        if self.has_p:
            if not_missing:
                return self.dist.log_density(x) + self.log_pn
            else:
                return self.log_p
        # This is a degenerate use case that should probably be deprecated
        else:
            if not_missing:
                return self.dist.log_density(x)
            else:
                return 0.0

    def seq_log_density(self, x: 'OptionalEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, OptionalEncodedDataSequence):
            raise Exception('OptionalEncodedDataSequence required for seq_log_density().')

        sz, z_idx, nz_idx, enc_data = x.data

        rv = np.zeros(sz)

        if self.has_p:
            rv[z_idx] = self.log_p
            rv[nz_idx] = self.dist.seq_log_density(enc_data) + self.log_pn
        else:
            rv[nz_idx] = self.dist.seq_log_density(enc_data)

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'OptionalSampler':
        return OptionalSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'OptionalEstimator':
        return OptionalEstimator(self.dist.estimator(pseudo_count=pseudo_count), missing_value=self.missing_value,
                                 pseudo_count=pseudo_count, est_prob=self.has_p, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'OptionalDataEncoder':
        return OptionalDataEncoder(encoder=self.dist.dist_to_encoder(), missing_value=self.missing_value)


class OptionalSampler(DistributionSampler):
    """OptionalSampler object for generating samples from OptionalDistribution.

    Attributes:
        dist (OptionalDistribution): OptionalDistribution to sample from.
        rng (RandomState): Seeded RandomState object.
        sampler (DistributionSampler): DistributionSampler for base distribution.

    """

    def __init__(self, dist: 'OptionalDistribution', seed: Optional[int] = None) -> None:
        """OptionalSampler object.

        Args:
            dist (OptionalDistribution): OptionalDistribution to sample from.
            seed (Optional[int]): Seed for RandomState object.

        """
        super().__init__(dist, seed)
        self.dist = dist
        self.sampler = self.dist.dist.sampler(self.new_seed())

    def sample(self, size: Optional[int] = None):
        """Generate samples from OptionalDistribution.

        Notes:
            Returns a missing_value or a sample from the base distribution (type T).

        Args:
            size (Optional[int]): Number of samples to generate.

        Returns:
            Union[Union[Any, T], Sequence[Union[Any, T]]
            
        """

        sampler = self.sampler

        if not self.dist.has_p:
            return self.sampler.sample(size=size)

        if size is None:
            if self.rng.choice([0, 1], replace=True, p=[self.dist.p, 1.0 - self.dist.p]) == 0:
                return self.dist.missing_value
            else:
                return sampler.sample(size=size)
        else:
            states = self.rng.choice([0, 1], size=size, replace=True, p=[self.dist.p, 1.0 - self.dist.p])

            nz_count = int(np.sum(states))

            if nz_count == size:
                return sampler.sample(size=size)
            elif nz_count == 0:
                return [self.dist.missing_value for i in range(size)]
            else:
                nz_vals = sampler.sample(size=nz_count)
                nz_idx = np.flatnonzero(states)
                rv = [self.dist.missing_value for i in range(size)]

                for cnt, i in enumerate(nz_idx):
                    rv[i] = nz_vals[cnt]

                return rv


class OptionalEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """OptionalEstimatorAccumulator object for accumulating sufficient statistics.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the base distribution.
        weights (List[float]): Suff stat weighted counts.
        missing_value (Any): Optional value
        missing_value_is_nan (bool): True if missing value is np.nan.
        keys (Optional[str]): Keys for the Optional distribution parameters and weights.
        name (Optional[str]): Name for the distribution.

    """

    def __init__(self, accumulator: SequenceEncodableStatisticAccumulator, missing_value: Any = None,
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """OptionalEstimatorAccumulator object.

        Args:
            accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the base distribution.
            missing_value (Any): Optional value
            keys (Optional[str]): Keys for the Optional distribution parameters and weights.
            name (Optional[str]): Name for the distribution.

        """
        self.accumulator = accumulator
        self.weights = [0.0, 0.0]
        self.missing_value = missing_value
        self.missing_value_is_nan = isinstance(missing_value, (np.floating, float)) and np.isnan(missing_value)
        self.keys = keys
        self.name = name

    def update(self, x: T, weight: float, estimate: OptionalDistribution) -> None:
        if self.missing_value_is_nan:
            if isinstance(x, (np.floating, float)) and np.isnan(x):
                self.weights[0] += weight
            else:
                self.accumulator.update(x, weight, estimate)
                self.weights[1] += weight
        else:
            if (x == self.missing_value) or (x is self.missing_value):
                self.weights[0] += weight
            else:
                self.accumulator.update(x, weight, estimate)
                self.weights[1] += weight

    def initialize(self, x: T, weight: float, rng: RandomState) -> None:
        if self.missing_value_is_nan:
            if isinstance(x, (np.floating, float)) and np.isnan(x):
                self.weights[0] += weight
            else:
                self.accumulator.initialize(x, weight, rng)
                self.weights[1] += weight
        else:
            if (x == self.missing_value) or (x is self.missing_value):
                self.weights[0] += weight
            else:
                self.accumulator.initialize(x, weight, rng)
                self.weights[1] += weight

    def seq_update(self, x: 'OptionalEncodedDataSequence', weights: np.ndarray,
                   estimate: OptionalDistribution) -> None:
        sz, z_idx, nz_idx, enc_data = x.data
        nz_weights = weights[nz_idx]
        z_weights = weights[z_idx]

        self.weights[0] += np.sum(z_weights)
        self.weights[1] += np.sum(nz_weights)
        self.accumulator.seq_update(enc_data, nz_weights, estimate.dist if estimate is not None else None)

    def seq_initialize(self, x: 'OptionalEncodedDataSequence', weights: np.ndarray, rng: RandomState) -> None:
        sz, z_idx, nz_idx, enc_data = x.data
        nz_weights = weights[nz_idx]
        z_weights = weights[z_idx]

        self.weights[0] += np.sum(z_weights)
        self.weights[1] += np.sum(nz_weights)
        self.accumulator.seq_initialize(enc_data, nz_weights, rng)

    def combine(self, suff_stat: Tuple[List[float], SS]) -> 'OptionalEstimatorAccumulator':
        self.weights[0] += suff_stat[0][0]
        self.weights[1] += suff_stat[0][1]
        self.accumulator.combine(suff_stat[1])

        return self

    def value(self) -> Tuple[List[float], Any]:
        return self.weights, self.accumulator.value()

    def from_value(self, x: Tuple[List[float], SS]) -> 'OptionalEstimatorAccumulator':
        self.weights = x[0]
        self.accumulator.from_value(x[1])

        return self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].from_value(self.value())
            else:
                stats_dict[self.keys] = self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def acc_to_encoder(self) -> 'OptionalDataEncoder':
        return OptionalDataEncoder(encoder=self.accumulator.acc_to_encoder(), missing_value=self.missing_value)


class OptionalEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """OptionalEstimatorAccumulatorFactory object for creating OptionalEstimatorAccumulator objects.

    Attributes:
        estimator (ParameterEstimator): Parameter Estimator for creating base distribution accumulator object.
        missing_value (Any): Optional value
        keys (Optional[str]): Keys for the Optional distribution parameters and weights.
        name (Optional[str]): Name for the distribution.

    """

    def __init__(self, estimator: ParameterEstimator, missing_value: Any = None, keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        """OptionalEstimatorAccumulatorFactory object for creating OptionalEstimatorAccumulator objects.

        Args:
            estimator (ParameterEstimator): Parameter Estimator for creating base distribution accumulator object.
            missing_value (Any): Optional value
            keys (Optional[str]): Keys for the Optional distribution parameters and weights.
            name (Optional[str]): Name for the distribution.

        """
        self.estimator = estimator
        self.missing_value = missing_value
        self.keys = keys
        self.name = name

    def make(self) -> 'OptionalEstimatorAccumulator':
        return OptionalEstimatorAccumulator(self.estimator.accumulator_factory().make(), self.missing_value,
                                            keys=self.keys, name=self.name)


class OptionalEstimator(ParameterEstimator):
    """OptionalEstimator for estimating OptionalDistribution from sufficient statistics.

    Attributes:
        estimator (ParameterEstimator): Estimator for base distribution.
        missing_value (Any): Missing_value specification.
        est_prob (bool): If true estimate the probability of a missing value.
        pseudo_count (Optional[float]): Regularize estimate of missing data.
        name (Optional[str]): Set name to object.
        keys (Optional[str]): Set keys for sufficient statistics.

    """

    def __init__(self, estimator: ParameterEstimator, missing_value: Any = None, est_prob: bool = False,
                 pseudo_count: Optional[float] = None, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """OptionalEstimator object.

        Args:
            estimator (ParameterEstimator): Estimator for base distribution.
            missing_value (Any): Missing_value specification.
            est_prob (bool): If true estimate the probability of a missing value.
            pseudo_count (Optional[float]): Regularize estimate of missing data.
            name (Optional[str]): Set name to object.
            keys (Optional[str]): Set keys for sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("OptionalEstimator requires keys to be of type 'str'.")

        self.estimator = estimator
        self.est_prob = est_prob
        self.pseudo_count = pseudo_count
        self.missing_value = missing_value
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'OptionalEstimatorAccumulatorFactory':
        return OptionalEstimatorAccumulatorFactory(self.estimator, self.missing_value, keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Optional[Tuple[List[float], SS]]) -> 'OptionalDistribution':
        dist = self.estimator.estimate(suff_stat[0][1], suff_stat[1])

        if self.pseudo_count is not None and self.est_prob:
            return OptionalDistribution(dist, (suff_stat[0][0] + self.pseudo_count) / (
                        (2 * self.pseudo_count) + suff_stat[0][0] + suff_stat[0][1]), missing_value=self.missing_value,
                                        name=self.name)

        elif self.est_prob:

            nobs_loc = suff_stat[0][0] + suff_stat[0][1]
            z_nobs = suff_stat[0][0]

            if nobs_loc == 0:
                return OptionalDistribution(dist, None, missing_value=self.missing_value, name=self.name)
            else:
                return OptionalDistribution(dist, p=z_nobs / nobs_loc, missing_value=self.missing_value, name=self.name)
        else:
            return OptionalDistribution(dist, p=None, missing_value=self.missing_value, name=self.name)


class OptionalDataEncoder(DataSequenceEncoder):
    """OptionalDataEncoder object for encoding sequences of Optional data.

    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for optional base distribution.
        missing_value (Any): Value considered 'optional' or missing.
        missing_value_is_nan (bool): True if the missing value is np.nan.

    """

    def __init__(self, encoder: DataSequenceEncoder, missing_value: Any = None) -> None:
        """OptionalDataEncoder object.

        Args:
            encoder (DataSequenceEncoder): DataSequenceEncoder for optional base distribution.
            missing_value (Any): Value considered 'optional' or missing.

        """
        self.encoder = encoder
        self.missing_value = missing_value
        self.missing_value_is_nan = isinstance(missing_value, (np.floating, float)) and np.isnan(missing_value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OptionalDataEncoder):
            # both are nan return True
            if self.missing_value_is_nan and other.missing_value_is_nan:
                return True

            # Both are not nan so check if they are the same
            if self.missing_value == other.missing_value:
                return True
            
            # default to False
            return False
        
        else:
            return False

    def seq_encode(self, x: Sequence[T]) -> 'OptionalEncodedDataSequence':
        nz_idx = []
        nz_val = []
        z_idx = []

        if self.missing_value_is_nan:
            for i, v in enumerate(x):
                if isinstance(v, (np.floating, float)) and np.isnan(v):
                    z_idx.append(i)
                else:
                    nz_idx.append(i)
                    nz_val.append(v)
        else:
            for i, v in enumerate(x):
                if v == self.missing_value:
                    z_idx.append(i)
                else:
                    nz_idx.append(i)
                    nz_val.append(v)

        enc_data = self.encoder.seq_encode(nz_val)

        nz_idx = np.asarray(nz_idx, dtype=int)
        z_idx = np.asarray(z_idx, dtype=int)

        return OptionalEncodedDataSequence(data=(len(x), z_idx, nz_idx, enc_data))

class OptionalEncodedDataSequence(EncodedDataSequence):
    """OptionalEncodedDataSequence object for vectorized function calls.

    Notes:
        E = Tuple[int, np.ndarray, np.ndarray, EncodedDataSequence]

    Attributes:
        data (E): Sequence encoded data for OptionalDistribution and OptionalAccumulator.


    """

    def __init__(self, data: Tuple[int, np.ndarray, np.ndarray, EncodedDataSequence]):
        """OptionalEncodedDataSequence object.

        Args:
            data (E): Sequence encoded data for OptionalDistribution and OptionalAccumulator.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'OptionalEncodedDataSequence(data={self.data})'

