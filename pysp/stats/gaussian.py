"""Evaluate, estimate, and sample from a gaussian distribution with mean mu and variance sigma2.

Defines the GaussianDistribution, GaussianSampler, GaussianAccumulatorFactory, GaussianAccumulator,
GaussianEstimator, and the GaussianDataEncoder classes for use with pysparkplug.

Data type: (float): The GaussianDistribution with mean mu and variance sigma2 > 0.0, has log-density
    log(f(x;mu, sigma2)) = -log(2*pi*sigma2) - (x-mu)^2/sigma2, for real-valued x.

"""
import numpy as np
from numpy.random import RandomState
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from typing import Optional, Tuple, List, Callable, Dict, Union, Any


class GaussianDistribution(SequenceEncodableProbabilityDistribution):
    """GaussianDistribution object defines Gaussian distribution with mean mu and variance sigma2.

    Attributes:
        mu (float): Mean of gaussian distribution.
        sigma2 (float): Variance of Gaussian distribution.
        name (Optional[str]): String for name of object.
        const (float): Normalizing constant of Gaussian (depends on sigma2).
        log_const (float): Log of const.
        keys (Optional[str]): Key for distribution

    """

    def __init__(self, mu: float, sigma2: float, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """GaussianDistribution object.

        Args:
            mu (float): Real-valued number.
            sigma2 (float): Positive real-valued number.
            name (Optional[str]): String for name of object.
            keys (Optional[str]): Key for distribution

        """
        self.mu = mu
        self.sigma2 = 1.0 if (sigma2 <= 0 or isnan(sigma2) or isinf(sigma2)) else sigma2
        self.log_const = -0.5 * log(2.0 * pi * self.sigma2)
        self.const = 1.0 / sqrt(2.0 * pi * self.sigma2)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return 'GaussianDistribution(%s, %s, name=%s, keys=%s)' % (repr(float(self.mu)), repr(float(self.sigma2)), repr(self.name),repr(self.keys))

    def density(self, x: float) -> float:
        """Density of Gaussian distribution at observation x.

        Args:
            x (float): Real-valued observation of Gaussian.

        Returns:
            float: Density of Gaussian at x.

        """
        return self.const * exp(-0.5 * (x - self.mu) * (x - self.mu) / self.sigma2)

    def log_density(self, x: float) -> float:
        """Log-density of Gaussian distribution at observation x.

        Args:
            x (float): Real-valued observation of Gaussian.

        Returns:
            float: Log-density at observation x.

        """
        return self.log_const - 0.5 * (x - self.mu) * (x - self.mu) / self.sigma2

    def seq_ld_lambda(self) -> List[Callable]:
        return [self.seq_log_density]

    def seq_log_density(self, x: 'GaussianEncodedDataSequence') -> np.ndarray:
        if not isinstance(x, GaussianEncodedDataSequence):
            raise Exception('GaussianDistribution.seq_log_density() requires GaussianEncodedDataSequence.')

        rv = x.data - self.mu
        rv *= rv
        rv *= -0.5 / self.sigma2
        rv += self.log_const

        return rv

    def sampler(self, seed:Optional[int] = None) -> 'GaussianSampler':
        return GaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GaussianEstimator':
        if pseudo_count is not None:
            suff_stat = (self.mu, self.sigma2)
            return GaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name, keys=self.keys)
        else:
            return GaussianEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'GaussianDataEncoder':
        return GaussianDataEncoder()


class GaussianSampler(DistributionSampler):
    """GaussianSampler for drawing samples from GaussianSampler instance.

    Attributes:
        dist (GaussianDistribution): GaussianDistribution instance to sample from.
        rng (RandomState): RandomState with seed set to seed if passed in args.

    """

    def __init__(self, dist: GaussianDistribution, seed: Optional[int] = None) -> None:
        """GaussianSampler object.

        Args:
            dist (GaussianDistribution): GaussianDistribution instance to sample from.
            seed (Optional[int]): Used to set seed in random sampler.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size' iid samples from GaussianSampler object.

        Numpy array of length 'size' from Gaussian distribution with mean mu and scale sigma2 if size not None.
        Else a single sample is returned as float.

        Args:
            size (Optional[int]): Treated as 1 if None is passed.

        Returns:
            Union[float, np.ndarray]: 'size' iid samples from Gaussian distribution.

        """
        return self.rng.normal(loc=self.dist.mu, scale=sqrt(self.dist.sigma2), size=size)


class GaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """GaussianAccumulator object used to accumulate sufficient statistics from observed data.

    Attributes:
        sum (float): Sum of weighted observations (sum_i w_i*X_i).
        sum2 (float): Sum of weighted squared observations (sum_i w_i*X_i^2)
        count (float): Sum of weights for observations (sum_i w_i).
        count2 (float): Sum of weights for squared observations (sum_i w_i).
        count (float): Tracks the sum of weighted observations used to form sum.
        keys (Optional[str]): Key for mean and variance.
        name (Optional[str]): Name for GaussianAccumulator object.

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """GaussianAccumulator object.

        Args:
            keys (Optional[str]): Key for mean and variance.
            name (Optional[str]): Set name for GaussianAccumulator object.

        """
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: float, weight: float, estimate: Optional['GaussianDistribution']) -> None:
        x_weight = x * weight
        self.sum += x_weight
        self.sum2 += x * x_weight
        self.count += weight
        self.count2 += weight

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, x: 'GaussianEncodedDataSequence', weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def seq_update(self, x: 'GaussianEncodedDataSequence', weights: np.ndarray, estimate: Optional[GaussianDistribution]) -> None:
        self.sum += np.dot(x.data, weights)
        self.sum2 += np.dot(x.data * x.data, weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat: Tuple[float, float, float, float]) -> 'GaussianAccumulator':
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        self.count2 += suff_stat[3]

        return self

    def value(self) -> Tuple[float, float, float, float]:
        return self.sum, self.sum2, self.count, self.count2

    def from_value(self, x: Tuple[float, float, float, float]) -> 'GaussianAccumulator':
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2, x3 = stats_dict[self.keys].value()
                self.sum += x0
                self.sum2 += x1
                self.count += x2
                self.count2 += x3

            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: Dict[str, 'GaussianAccumulator']) -> None:

        if self.keys is not None:
            if self.keys in stats_dict:
                self.sum, self.sum2, self.count, self.count2 = stats_dict[self.keys]

    def acc_to_encoder(self) -> 'GaussianDataEncoder':
        return GaussianDataEncoder()


class GaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """GaussianAccumulatorFactory object for creating GaussianAccumulator.

    Attributes:
        name (Optional[str]): Name of the GaussianAccumulatorFactory obejct.
        keys (Optional[str]): String id for merging sufficient statistics of GaussianAccumulator.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """GaussianAccumulatorFactory object.

        Args:
            name (Optional[str]): Assign a name to GaussianAccumulatorFactory object.
            keys (Optional[str]): Assign keys member for GaussianAccumulators.

        """
        self.keys = keys
        self.name = name

    def make(self) -> 'GaussianAccumulator':
        return GaussianAccumulator(name=self.name, keys=self.keys)


class GaussianEstimator(ParameterEstimator):
    """GaussianEstimator object used to estimate GaussianDistribution from aggregated sufficient statistics.

    Attributes:
        pseudo_count (Tuple[Optional[float], Optional[float]]): Weights for suff_stat.
        suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of mean (mu) and variance (sigma2).
        name (Optional[str]): String name of GaussianEstimator instance.
        keys (Optional[str]): key for mean and variance

    """

    def __init__(self,
                 pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
                 suff_stat: Tuple[Optional[float], Optional[float]] = (None, None),
                 name: Optional[str] = None,
                 keys: Optional[str] = None):
        """GaussianEstimator object.

        Args:
            pseudo_count (Tuple[Optional[float], Optional[float]]): Tuple of two positive floats.
            suff_stat (Tuple[Optional[float], Optional[float]]): Tuple of float and positive float.
            name (Optional[str]): Assign a name to GaussianEstimator.
            keys (Optional[str]): key for mean and variance

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("BinomialEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'GaussianAccumulatorFactory':
        return GaussianAccumulatorFactory(self.name, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float, float]) -> 'GaussianDistribution':
        nobs_loc1 = suff_stat[2]
        nobs_loc2 = suff_stat[3]

        if nobs_loc1 == 0.0:
            mu = 0.0
        elif self.pseudo_count[0] is not None and self.suff_stat[0] is not None:
            mu = (suff_stat[0] + self.pseudo_count[0] * self.suff_stat[0]) / (nobs_loc1 + self.pseudo_count[0])
        else:
            mu = suff_stat[0] / nobs_loc1

        if nobs_loc2 == 0.0:
            sigma2 = 0.0
        elif self.pseudo_count[1] is not None and self.suff_stat[1] is not None:
            sigma2 = (suff_stat[1] - mu * mu * nobs_loc2 + self.pseudo_count[1] * self.suff_stat[1]) / (
                        nobs_loc2 + self.pseudo_count[1])
        else:
            sigma2 = suff_stat[1] / nobs_loc2 - mu * mu

        return GaussianDistribution(mu, sigma2, name=self.name)


class GaussianDataEncoder(DataSequenceEncoder):
    """GaussianDataEncoder object for encoding sequences of iid Gaussian observations with data type float."""

    def __str__(self) -> str:
        return 'GaussianDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, GaussianDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> 'GaussianEncodedDataSequence':
        """Encode sequence of iid Gaussian observations.

        Data type must be List[float] or np.ndarray[float].

        Args:
            x (Union[List[float], np.ndarray]): Sequence of iid Gaussian observations.

        Returns:
            A numpy array of floats.

        """
        rv = np.asarray(x, dtype=float)

        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise Exception('GaussianDistribution requires support x in (-inf,inf).')

        return GaussianEncodedDataSequence(data=rv)


class GaussianEncodedDataSequence(EncodedDataSequence):
    """GaussianEncodedDataSequence object for use with vectorized function calls.

    Attributes:
        data (np.ndarray): Sequence of iid Guassian observations.

    """

    def __init__(self, data: np.ndarray):
        """GaussianEncodedDataSequence object.

        Args:
            data (np.ndarray): Sequence of iid Guassian observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'GaussianEncodedDataSequence(data={self.data})'


