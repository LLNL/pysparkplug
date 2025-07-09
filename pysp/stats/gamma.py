"""Create, estimate, and sample from a gamma distribution with shape k and scale theta.

Defines the GammaDistribution, GammaSampler, GammaAccumulatorFactory, GammaAccumulator, GammaEstimator,
and the GammaDataEncoder classes for use with pysparkplug.

"""
import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln
from typing import Tuple, List, Optional, Union, Dict, Any
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from pysp.utils.special import digamma, trigamma


class GammaDistribution(SequenceEncodableProbabilityDistribution):
    """GammaDistribution for shape k and scale theta.

     Attributes:
         k (float): Positive real-valued number.
         theta (float): Positive real-valued number.
         name (Optional[str]): Assign a name to GammaDistribution instance.
         log_const (float): Normalizing constant of gamma distribution.
        keys (Optional[str]): Assign key to parameters of distribution. 

     """

    def __init__(self, k: float, theta: float, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """GammaDistribution object.

        Args:
            k (float): Positive real-valued number.
            theta (float): Positive real-valued number.
            name (Optional[str]): Assign a name to GammaDistribution instance.
            keys (Optional[str]): Assign key to parameters of distribution. 

        """
        self.k = k
        self.theta = theta
        self.log_const = -(gammaln(k) + k * log(theta))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        s0 = repr(float(self.k))
        s1 = repr(float(self.theta))
        s2 = repr(self.name)
        s3 = repr(self.keys)

        return 'GammaDistribution(%s, %s, name=%s, keys=%s)' % (s0, s1, s2, s3) 

    def density(self, x: float) -> float:
        """Density of gamma distribution evaluated at x.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Density of gamma distribution evaluated at x.

        """
        return exp(self.log_const + (self.k - one) * log(x) - x / self.theta)

    def log_density(self, x: float) -> float:
        """Log-density of gamma distribution evaluated at x.

        Args:
            x (float): Positive real-valued number.

        Returns:
            float: Log-density of gamma distribution evaluated at x.


        """
        return self.log_const + (self.k - one) * log(x) - x / self.theta

    def seq_log_density(self, x: 'GammaEncodedDataSequence') -> np.ndarray:
        if not isinstance(x, GammaEncodedDataSequence):
            raise Exception("GammaEncodedDataSequence required for seq_log_density().")

        rv = x.data[0] * (-1.0 / self.theta)
        if self.k != 1.0:
            rv += x.data[1] * (self.k - 1.0)
        rv += self.log_const

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'GammaSampler':
        return GammaSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'GammaEstimator':
        if pseudo_count is None:
            return GammaEstimator(name=self.name, keys=self.keys)
        else:
            suff_stat = (self.k * self.theta, exp(digamma(self.k) + log(self.theta)))
            return GammaEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'GammaDataEncoder':
        return GammaDataEncoder()


class GammaSampler(DistributionSampler):
    """GammaSampler object used to draw samples from GammaDistribution.

    Attributes:
        rng (RandomState): RandomState with seed set for sampling.
        dist (GammaDistribution): GammaDistribution to sample from.
        seed (Optional[int]): Used to set seed on random number generator used in sampling.

    """

    def __init__(self, dist: 'GammaDistribution', seed: Optional[int] = None) -> None:
        """GammaSampler object.

        Args:
            dist (GammaDistribution): GammaDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.

        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw 'size'-iid observations from GammaSampler.

        Args:
            size (Optional[int]): Number of iid samples to draw from GammaSampler.

        Returns:
            Single sample (float) if size is None, else a numpy array of floats containing iid samples from
            GammaDistribution.

        """
        if size:
            return self.rng.gamma(shape=self.dist.k, scale=self.dist.theta,
                                  size=size).tolist()
        else:
            return float(self.rng.gamma(shape=self.dist.k,
                                        scale=self.dist.theta))


class GammaAccumulator(SequenceEncodableStatisticAccumulator):
    """GammaAccumulator object used to accumulate sufficient statistics from observations.

    Attributes:
        nobs (float): Number of observations accumulated.
        sum (float): Weighted-sum of observations accumulated.
        sum_of_logs (float): log weighted sum of weighted log(observations).
        key (Optional[str]): GammaAccumulator objects with same key merge sufficient statistics.
        name (Optional[str]): Name for object

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """GammaAccumulator object.

        Args:
            keys (Optional[str]): GammaAccumulator objects with same key merge sufficient statistics.
            name (Optional[str]): Name for object

        """
        self.nobs = zero
        self.sum = zero
        self.sum_of_logs = zero
        self.keys = keys
        self.name = name

    def initialize(self, x: float, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, x: 'GammaEncodedDataSequence', weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def update(self, x: float, weight: float, estimate: Optional['GammaDistribution']) -> None:
        self.nobs += weight
        self.sum += x * weight
        self.sum_of_logs += log(x) * weight

    def seq_update(self,
                   x: 'GammaEncodedDataSequence',
                   weights: np.ndarray,
                   estimate: Optional['GammaDistribution']) -> None:
        self.sum += np.dot(x.data[0], weights)
        self.sum_of_logs += np.dot(x.data[1], weights)
        self.nobs += np.sum(weights)

    def combine(self, suff_stat: Tuple[float, float, float]) -> 'GammaAccumulator':

        self.nobs += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_of_logs += suff_stat[2]

        return self

    def value(self) -> Tuple[float, float, float]:

        return self.nobs, self.sum, self.sum_of_logs

    def from_value(self, x: Tuple[float, float, float]) -> 'GammaAccumulator':

        self.nobs = x[0]
        self.sum = x[1]
        self.sum_of_logs = x[2]

        return self

    def key_merge(self, stats_dict: Dict[str,  Any]) -> None:

        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2 = stats_dict[self.keys]
                self.nobs += x0
                self.sum += x1
                self.sum_of_logs += x2

            else:
                stats_dict[self.keys] = (self.nobs, self.sum, self.sum_of_logs)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2 = stats_dict[self.keys]
                self.nobs = x0
                self.sum = x1
                self.sum_of_logs = x2

    def acc_to_encoder(self) -> 'GammaDataEncoder':
        return GammaDataEncoder()


class GammaAccumulatorFactory(StatisticAccumulatorFactory):
    """GammaAccumulatorFactory object for creating GammaAccumulator objects.

    Attributes:
        keys (Optional[str]): Used for merging sufficient statistics of GammaAccumulator.
        name (Optional[str]): Name for object

    """

    def __init__(self, keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """GammaAccumulatorFactory object.

        Args:
            keys (Optional[str]): Used for merging sufficient statistics of GammaAccumulator.
            name (Optional[str]): Name for object

        """
        self.keys = keys
        self.name = name

    def make(self) -> 'GammaAccumulator':
        return GammaAccumulator(keys=self.keys, name=self.name)


class GammaEstimator(ParameterEstimator):
    """GammaEstimator object used for estimating GammaDistribution from aggregated data.

    Attributes:
        pseudo_count (Tuple[float, float]): Values used to re-weight member instances of sufficient statistics.
        suff_stat (Tuple[float, float]):  shape 'k' and scale 'theta'.
        threshold (float): Threshold used for estimating the shape of gamma.
        name (Optional[str]): Assign a name to GammaEstimator.
        keys (Optional[str]): Assign keys to GammaEstimator for combining sufficient statistics.

    """

    def __init__(self, pseudo_count: Tuple[float, float] = (0.0, 0.0), suff_stat: Tuple[float, float] = (1.0, 0.0),
                 threshold: float = 1.0e-8, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """GammaEstimator object.

        Args:
            pseudo_count (Tuple[float, float]): Values used to re-weight member instances of sufficient statistics.
            suff_stat (Tuple[float, float]):  shape 'k' and scale 'theta'.
            threshold (float): Threshold used for estimating the shape of gamma.
            name (Optional[str]): Assign a name to GammaEstimator.
            keys (Optional[str]): Assign keys to GammaEstimator for combining sufficient statistics.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("GammaEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.threshold = threshold
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'GammaAccumulatorFactory':
        return GammaAccumulatorFactory(keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[float, float, float]) -> 'GammaDistribution':
        pc1, pc2 = self.pseudo_count
        ss1, ss2 = self.suff_stat

        if suff_stat[0] == 0:
            return GammaDistribution(1.0, 1.0)

        adj_sum = suff_stat[1] + ss1 * pc1
        adj_cnt = suff_stat[0] + pc1
        adj_mean = adj_sum / adj_cnt

        adj_lsum = suff_stat[2] + ss2 * pc2
        adj_lcnt = suff_stat[0] + pc2
        adj_lmean = adj_lsum / adj_lcnt

        k = self.estimate_shape(adj_mean, adj_lmean, self.threshold)

        return GammaDistribution(k, adj_sum / (k * adj_lcnt), name=self.name)

    @staticmethod
    def estimate_shape(avg_sum: float, avg_sum_of_logs: float, threshold: float) -> float:
        """Estimates the shape parameter of GammaDistribution.

        Args:
            avg_sum (float): Weighted sum of gamma observations.
            avg_sum_of_logs (float): Weighted log sum of gamma observations.
            threshold (float): Threshold used for assessing convergence of shape estimation.

        Returns:
            Estimate of shape parameter 'k'.

        """
        s = log(avg_sum) - avg_sum_of_logs
        old_k = inf
        k = (3 - s + sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s)
        while abs(old_k - k) > threshold:
            old_k = k
            k -= (log(k) - digamma(k) - s) / (one / k - trigamma(k))
        return k


class GammaDataEncoder(DataSequenceEncoder):
    """GammaDataEncoder object for encoding sequences of iid Gamma observations with data type float."""

    def __str__(self) -> str:
        return 'GammaDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GammaDataEncoder)

    def seq_encode(self, x: Union[List[float], np.ndarray]) -> 'GammaEncodedDataSequence':
        rv1 = np.asarray(x, dtype=float)

        if np.any(rv1 <= 0) or np.any(np.isnan(rv1)):
            raise Exception('GammaDistribution has support x > 0.')
        else:
            rv2 = np.log(rv1)

            return GammaEncodedDataSequence(data=(rv1, rv2))

class GammaEncodedDataSequence(EncodedDataSequence):
    """GammaEncodedDataSequence object for use with vectorized functions.

    Attributes:
        data (Tuple[np.ndarray, np.ndarray]): Encoded data for gamma distribution.

    """

    def __init__(self, data: Tuple[np.ndarray, np.ndarray]):
        """GammaEncodedDataSequence object.

        Args:
            data (Tuple[np.ndarray, np.ndarray]): Encoded data for gamma distribution.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'GammaEncodedDataSequence(data={self.data}'

