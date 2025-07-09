"""Create, estimate, and sample from an exponential distribution with scale beta.

Defines the ExponentialDistribution, ExponentialSampler, ExponentialAccumulatorFactory,ExponentialAccumulator,
ExponentialEstimator, and the ExponentialDataEncoder classes for use with pysparkplug.

"""
from typing import Optional, Tuple
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from numpy.random import RandomState
import numpy as np
from typing import List, Union, Dict, Any


class ExponentialDistribution(SequenceEncodableProbabilityDistribution):
    """ExponentialDistribution object for scale beta.

    Attributes:
        beta (float): Positive valued real number defining scale of exponential distribution.
        log_beta (float): log of beta parameter.
        name (Optional[str]): Assign a name to ExponentialDistribution object.
        keys (Optional[str]): Key for parameters.

    """

    def __init__(self, beta: float, name: Optional[str] = None, keys: Optional[str] = None):
        """ExponentialDistribution object.

        Args:
            beta (float): Positive valued real number defining scale of exponential distribution.
            name (Optional[str]): Assign a name to ExponentialDistribution object.
            keys (Optional[str]): Key for parameters.

        """
        self.beta = beta
        self.log_beta = np.log(beta)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'ExponentialDistribution(%s, name=%s, keys=%s)' % (repr(self.beta), repr(self.name), repr(self.keys))

    def density(self, x: float) -> float:
        """Evaluate the density of exponential distribution with scale beta.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Density evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Evaluate the log-density of exponential distribution with scale beta.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Log-density evaluated at x.

        """
        if x < 0:
            return -inf
        else:
            return -x / self.beta - self.log_beta

    def seq_log_density(self, x: 'ExponentialEncodedDataSequence') -> np.ndarray:
        if not isinstance(x, ExponentialEncodedDataSequence):
            raise Exception('ExponentialEncodedDataSequence required for seq_log_density().')

        rv = x.data * (-1.0 / self.beta)
        rv -= self.log_beta
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'ExponentialSampler':
        return ExponentialSampler(dist=self, seed=seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'ExponentialEstimator':
        if pseudo_count is None:
            return ExponentialEstimator(name=self.name, keys=self.keys)
        else:
            return ExponentialEstimator(pseudo_count=pseudo_count, suff_stat=self.beta, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'ExponentialDataEncoder':
        return ExponentialDataEncoder()


class ExponentialSampler(DistributionSampler):
    """ExponentialSampler for drawing samples from ExponentialSampler instance.

    Attributes:
        dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
        rng (RandomState): RandomState with seed set to seed if passed in args.

    """

    def __init__(self, dist: 'ExponentialDistribution', seed: Optional[int] = None) -> None:
        """ExponentialSampler object.

        Args:
            dist (ExponentialDistribution): ExponentialDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from ExponentialSampler object.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            Union[float, np.ndarray]: Numpy array of length 'size' from exponential distribution with scale beta if
            size not None. Else a single sample is returned as float.

        """
        return self.rng.exponential(scale=self.dist.beta, size=size)


class ExponentialAccumulator(SequenceEncodableStatisticAccumulator):
    """ExponentialAccumulator object used to accumulate sufficient statistics.

    Attributes:
        sum (float): Tracks the sum of observation values.
        count (float): Tracks the sum of weighted observations used to form sum.
        keys (Optional[str]): Aggregate all sufficient statistics with same key.
        name (Optional[str]): Name for object. 

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """ExponentialAccumulator object.

        Args:
            keys (Optional[str]): Aggregate all sufficient statistics with same keys values.
            name (Optional[str]): Name for object.

        """
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: float, weight: float, estimate: Optional['ExponentialDistribution']) -> None:
        if x >= 0:
            self.sum += x * weight
            self.count += weight

    def initialize(self, x: float, weight: float, rng: Optional['np.random.RandomState']) -> None:
        self.update(x, weight, None)

    def seq_update(self,
                   x: 'ExponentialEncodedDataSequence',
                   weights: np.ndarray,
                   estimate: Optional['ExponentialDistribution']) -> None:
        self.sum += np.dot(x.data, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self,
                       x: 'ExponentialEncodedDataSequence',
                       weights: np.ndarray,
                       rng: np.random.RandomState) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> 'ExponentialAccumulator':
        """Aggregates sufficient statistics with ExponentialAccumulator member sufficient statistics.

        Args:
            suff_stat (Tuple[float, float]): Aggregated count and sum.

        Returns:
            ExponentialAccumulator

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> Tuple[float, float]:
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> 'ExponentialAccumulator':
        self.count = x[0]
        self.sum = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1 = stats_dict[self.keys]
                self.count += x0
                self.sum += x1
            else:
                stats_dict[self.keys] = (self.count, self.sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.count = stats_dict[self.keys][0]
                self.sum = stats_dict[self.keys][1]

    def acc_to_encoder(self) -> 'ExponentialDataEncoder':
        return ExponentialDataEncoder()


class ExponentialAccumulatorFactory(StatisticAccumulatorFactory):
    """ExponentialAccumulatorFactory object for creating ExponentialAccumulator.

    Attributes:
        keys (Optional[str]): Used for merging sufficient statistics of ExponentialAccumulator.
        name (Optional[str]): Name for object.

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """ExponentialAccumulatorFactory object.

        Args:
            keys (Optional[str]): Used for merging sufficient statistics of ExponentialAccumulator.
            name (Optional[str]): Name for object.

        """
        self.keys = keys
        self.name = name

    def make(self) -> 'ExponentialAccumulator':
        return ExponentialAccumulator(keys=self.keys, name=self.name)


class ExponentialEstimator(ParameterEstimator):
    """ExponentialEstimator object estimates ExponentialDistribution from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Optional[float]): Used to weight sufficient statistics.
        suff_stat (Optional[float]): Positive float value for scale of exponential distribution.
        name (Optional[str]): Assign a name to ExponentialEstimator.
        keys (Optional[str]): Assign keys to ExponentialEstimator for combining sufficient statistics.

    """

    def __init__(self,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[float] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """ExponentialEstimator object.

        Args:
            pseudo_count (Optional[float]): Used to weight sufficient statistics.
            suff_stat (Optional[float]): Positive float value for scale of exponential distribution.
            name (Optional[str]): Assign a name to ExponentialEstimator.
            keys (Optional[str]): Assign keys to ExponentialEstimator for combining sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("ExponentialEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'ExponentialAccumulatorFactory':
        return ExponentialAccumulatorFactory(keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> 'ExponentialDistribution':

        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[1] + self.suff_stat * self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[1] + self.pseudo_count) / (suff_stat[0] + self.pseudo_count)
        else:
            if suff_stat[0] > 0:
                p = suff_stat[1] / suff_stat[0]
            else:
                p = 1.0

        return ExponentialDistribution(beta=p, name=self.name)


class ExponentialDataEncoder(DataSequenceEncoder):
    """ExponentialDataEncoder object for encoding sequences of iid exponential observations with data type float."""

    def __str__(self) -> str:
        """Returns string representation of ExponentialDataEncoder."""
        return 'ExponentialDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Check if object is an instance of ExponentialDataEncoder.

        Args:
            other (object): Object to compare.

        Returns:
            True if object is an instance of ExponentialDataEncoder, else False.

        """
        return isinstance(other, ExponentialDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> 'ExponentialEncodedDataSequence':
        rv = np.asarray(x, dtype=float)

        if np.any(rv <= 0) or np.any(np.isnan(rv)):
            raise Exception('Exponential requires x > 0.')

        return ExponentialEncodedDataSequence(data=rv)

class ExponentialEncodedDataSequence(EncodedDataSequence):
    """ExponentialEncodedDataSequence object for vectorized function calls.

    Attributes:
        data (np.ndarray): Sequence of iid exponential observations.

    """

    def __init__(self, data: np.ndarray):
        """ExponentialEncodedDataSequence object.

        Args:
            data (np.ndarray): Sequence of iid exponential observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'ExponentialEncodedDataSequence(data={self.data})'

