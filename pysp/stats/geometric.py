"""Create, estimate, and sample from a geometric distribution with probability of success p.

Defines the GeometricDistribution, GeometricSampler, GeometricAccumulatorFactory, GeometricAccumulator,
GeometricEstimator, and the GeometricDataEncoder classes for use with pysparkplug.

Data type (int): The geometric distribution with probability of success p, has density

    P(x=k) = (k-1)*log(1-p) + log(p), for k = 1,2,...

"""
import numpy as np
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from numpy.random import RandomState
from typing import Optional, Tuple, Sequence, Dict, Union, Any


class GeometricDistribution(SequenceEncodableProbabilityDistribution):
    """GeometricDistribution object defining geometric distribution with probability of success p.

    Attributes:
        p (float): Probability of success, must between (0,1).
        log_p (float): Log of probability of success p.
        log_1p (float): Log of 1-p (prob of failure).
        name (Optional[str]): Assign name to GeometricDistribution object.
        keys (Optional[str]): Assign keys to parameter p. 

    """

    def __init__(self, p: float, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """GeometricDistribution object.

        Args:
            p (float): Must between (0,1).
            name (Optional[str]): Assign name to GeometricDistribution object.
            keys (Optional[str]): Assign keys to parameter p. 

        """
        self.p = max(0.0, min(p, 1.0))
        self.log_p = np.log(p)
        self.log_1p = np.log1p(-p)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'GeometricDistribution(%s, name=%s, keys=%s)' % (repr(self.p), repr(self.name), repr(self.keys))

    def density(self, x: int) -> float:
        """Density of geometric distribution evaluated at x.

            P(x=k) = (k-1)*log(1-p) + log(p), for x = 1,2,..., else 0.0.

        Args:
            x (int): Observed geometric value (1,2,3,....).


        Returns:
            float: Density of geometric distribution evaluated at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of geometric distribution evaluated at x.

        See density() for details.

        Args:
            x (int): Must be natural number (1,2,3,....).

        Returns:
            float: Log-density of geometric distribution evaluated at x.

        """
        return (x - 1) * self.log_1p + self.log_p

    def seq_log_density(self, x: 'GeometricEncodedDataSequence') -> np.ndarray:

        if not isinstance(x, GeometricEncodedDataSequence):
            raise Exception("GeometricEncodedDataSequence required for seq_log_density().")

        rv = x.data - 1
        rv *= self.log_1p
        rv += self.log_p

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'GeometricSampler':
        return GeometricSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GeometricEstimator':
        if pseudo_count is None:
            return GeometricEstimator(name=self.name, keys=self.keys)
        else:
            return GeometricEstimator(pseudo_count=pseudo_count, suff_stat=self.p, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'GeometricDataEncoder':
        return GeometricDataEncoder()


class GeometricSampler(DistributionSampler):
    """GeometricSampler object used to draw samples from GeometricDistribution.

    Attributes:
        rng (RandomState): RandomState with seed set for sampling.
        dist (GeometricDistribution): GeometricDistribution to sample from.

    """

    def __init__(self, dist: GeometricDistribution, seed: Optional[int] = None) -> None:
        """GeometricSampler object.

        Attributes:
            rng (RandomState): RandomState with seed set for sampling.
            dist (GeometricDistribution): GeometricDistribution to sample from.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate iid samples from geometric distribution.

        Generates a single geometric sample (int) if size is None, else a numpy array of integers of length size,
        iid samples, from the geometric distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, assumed to be 1.

        Returns:
            Union[int, np.ndarray]: If size is None, int, else size length numpy array of ints.

        """
        return self.rng.geometric(p=self.dist.p, size=size)


class GeometricAccumulator(SequenceEncodableStatisticAccumulator):
    """GeometricAccumulator object used to accumulate sufficient statistics from observations.

    Attributes:
        sum (float): Aggregate weighted sum of observations.
        count (float): Aggregate sum of weighted observation count.
        name (Optional[str]): Assigned from name arg.
        keys (Optional[str]): Assigned from keys arg.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None):
        """GeometricAccumulator object.

        Args:
            name (Optional[str]): Assign a name to the object instance.
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.

        """
        self.sum = 0.0
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: int, weight: float, estimate: Optional['GeometricDistribution']) -> None:

        if x >= 0:
            self.sum += x * weight
            self.count += weight

    def seq_update(self, x: 'GeometricEncodedDataSequence', weights: np.ndarray, estimate: Optional['GeometricDistribution']) -> None:
        self.sum += np.dot(x.data, weights)
        self.count += np.sum(weights)

    def initialize(self, x: int, weight: float, rng: Optional[RandomState]) -> None:

        self.update(x, weight, None)

    def seq_initialize(self, x: 'GeometricEncodedDataSequence', weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[float, float]) -> 'GeometricAccumulator':
        self.sum += suff_stat[1]
        self.count += suff_stat[0]

        return self

    def value(self) -> Tuple[float, float]:
        return self.count, self.sum

    def from_value(self, x: Tuple[float, float]) -> 'GeometricAccumulator':

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
                self.count, self.sum = stats_dict[self.keys]

    def acc_to_encoder(self) -> 'GeometricDataEncoder':
        return GeometricDataEncoder()


class GeometricAccumulatorFactory(StatisticAccumulatorFactory):
    """GeometricAccumulatorFactory object used to create GeometricAccumulator objects.

    Attributes:
        name (Optional[str]): Assigned from name arg.
        keys (Optional[str]): Assigned from keys arg.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """GeometricAccumulatorFactory object.

        Args:
            name (Optional[str]): Assign a name to the object instance.
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.

        """
        self.name = name
        self.keys = keys

    def make(self) -> 'GeometricAccumulator':
        return GeometricAccumulator(name=self.name, keys=self.keys)


class GeometricEstimator(ParameterEstimator):
    """GeometricEstimator object for estimating GeometricDistribution object from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Optional[float]): Assigned from pseudo_count arg.
        suff_stat (Optional[float]): Assigned from suff_stat arg (corrected for [0,1] constraint).
        name (Optional[str]): Assigned from name arg.
        keys (Optional[str]): Assigned from keys arg.

    """

    def __init__(self, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[float] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """GeometricEstimator object.

        Args:
            pseudo_count (Optional[float]): Float value for re-weighting suff_stat member variable.
            suff_stat (Optional[float]): Probability of success (value between (0,1)).
            name (Optional[str]): Assign a name to the object instance.
            keys (Optional[str]): GeometricAccumulator objects with same key merge sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("GeometricEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = min(min(suff_stat, 1.0),0.0) if suff_stat is not None else None
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'GeometricAccumulatorFactory':
        return GeometricAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float]) -> 'GeometricDistribution':
        if self.pseudo_count is not None and self.suff_stat is not None:
            p = (suff_stat[0] + self.pseudo_count * self.suff_stat) / (
                    suff_stat[1] + self.pseudo_count)
        elif self.pseudo_count is not None and self.suff_stat is None:
            p = (suff_stat[0] + self.pseudo_count) / (suff_stat[1] + self.pseudo_count)
        else:
            p = suff_stat[0] / suff_stat[1]

        return GeometricDistribution(p, name=self.name)


class GeometricDataEncoder(DataSequenceEncoder):
    """GeometricDataEncoder object for encoding sequences of iid geometric observations with data type int."""

    def __str__(self) -> str:
        return 'GeometricDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, GeometricDataEncoder)

    def seq_encode(self, x: Union[Sequence[int], np.ndarray]) -> 'GeometricEncodedDataSequence':
        rv = np.asarray(x)
        if np.any(rv < 1) or np.any(np.isnan(rv)):
            raise Exception('GeometricDistribution requires integers greater than 0 for x.')
        else:
            return GeometricEncodedDataSequence(data=np.asarray(rv, dtype=float))

class GeometricEncodedDataSequence(EncodedDataSequence):
    """GeometricEncodedDataSequence object for vectorized functions.

    Attributes:
        data (np.ndarray): Sequence of iid Geometric observations.

    """

    def __init__(self, data: np.ndarray):
        """GeometricEncodedDataSequence object.

        Args:
            data (np.ndarray): Sequence of iid Geometric observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'GeometricEncodedDataSequence(data={self.data})'




