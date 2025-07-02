"""Create, estimate, and sample from a diagonal Gaussian distribution (independent-multivariate Gaussian).

Defines the DiagonalGaussianDistribution, DiagonalGaussianSampler, DiagonalGaussianAccumulatorFactory,
DiagonalGaussianAccumulator, DiagonalGaussianEstimator, and the DiagonalGaussianDataEncoder classes for use with
pysparkplug.

The log-density of an 'n' dimensional diagonal-gaussian observation x = (x_1,x_2,...,x_n) with mean mu=(m_1,m_2,..,m_n),
and diagonal covariance matrix given by covar = diag(s2_1, s2_2,...,s2_n).

    log(p_mat(x)) = -0.5*sum_{i=1}^{n} (x_i-m_i)^2 / s2_i - 0.5*log(s2_i) - (n/2)*log(pi).

Data type: x (List[float], np.ndarray).

"""
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory, EncodedDataSequence

from numpy.random import RandomState
import pysp.utils.vector as vec
import numpy as np
from numpy.random import RandomState

from typing import Sequence, Optional, Dict, Any, Tuple, List, Union


class DiagonalGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Create a DiagonalGaussianDistribution object with mean mu and covariance covar.

    Attributes:
         dim (int): Dimension of the multivariate Gaussian. Determined by mean length.
         mu (np.ndarray): Mean of the Gaussian.
         covar (np.ndarray): Variance for each component.
         name (Optional[str]): Name of object instance.
         log_c (float): Normalizing constant for diagonal Gaussian.
         ca (np.ndarray): Term for likelihood-calc.
         cb (np.ndarray): Term for likelihood-calc.
         cc (np.ndarray): Term for likelihood-calc.
         keys (Optional[str], Optional[str]): Key for mean and covariance.

    """

    def __init__(self, mu: Union[Sequence[float], np.ndarray], covar: Union[Sequence[float], np.ndarray],
                 name: Optional[str] = None, keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        """Create a DiagonalGaussianDistribution object with mean mu and covariance covar.

        Args:
            mu (Union[Sequence[float], np.ndarray]): Mean of Gaussian distribution.
            covar (Union[Sequence[float], np.ndarray]): Variance of each component.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str], Optional[str]): Key for mean and covariance.

        """
        self.dim = len(mu)
        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covar, dtype=float)
        self.name = name
        self.log_c = -0.5 * (np.log(2.0 * np.pi) * self.dim + np.log(self.covar).sum())

        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c
        self.keys = keys

    def __str__(self) -> str:
        s1 = repr(list(self.mu.flatten()))
        s2 = repr(list(self.covar.flatten()))
        s3 = repr(self.name)
        s4 = repr(self.keys)
        return 'DiagonalGaussianDistribution(%s, %s, name=%s, keys=%s)' % (s1, s2, s3, s4)

    def density(self, x: Union[Sequence[float], np.ndarray]):
        """Evaluate the density of DiagonalGaussianDistribution.

        Args:
            x (Union[Sequence[float], np.ndarray]): Observation from Diagonal Gaussian.

        Returns:
            float: Density at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: Union[Sequence[float], np.ndarray]):
        """Evaluate the log-density of DiagonalGaussianDistribution.

        Args:
            x (Union[Sequence[float], np.ndarray]): Observation from Diagonal Gaussian.

        Returns:
            float: Log-density at x.

        """
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc

        return rv

    def seq_log_density(self, x: 'DiagonalGaussianEncodedDataSequence') -> np.ndarray:
        if not isinstance(x, DiagonalGaussianEncodedDataSequence):
            raise Exception('DiagonalGaussianDistribution.seq_log_density() requires DiagonalGaussianEncodedDataSequence.')

        rv = np.dot(x.data * x.data, self.ca)
        rv += np.dot(x.data, self.cb)
        rv += self.cc

        # rv = np.sum(-0.5*(x - self.mu[None, :])**2 / self.covar[None, :], axis=1, keepdims=False)
        # rv -= 0.5*np.sum(np.log(self.covar))

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'DiagonalGaussianSampler':
        return DiagonalGaussianSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'DiagonalGaussianEstimator':
        if pseudo_count is None:
            return DiagonalGaussianEstimator(dim=self.dim, name=self.name, keys=self.keys)
        else:
            return DiagonalGaussianEstimator(dim=self.dim, pseudo_count=(pseudo_count, pseudo_count), name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'DiagonalGaussianDataEncoder':
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianSampler(DistributionSampler):
    """DiagonalGaussianSampler object for sampling from DiagonalGaussian instance.

    Attributes:
        dist (DiagonalGaussianDistribution): Object instance to sample from.
        seed (Optional[int]): Seed for random number generator.

    """

    def __init__(self, dist: DiagonalGaussianDistribution, seed: Optional[int] = None) -> None:
        """DiagonalGaussianSampler object.

        Args:
            dist (DiagonalGaussianDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[Sequence[np.ndarray], np.ndarray]:
        """Generate a samples from Diagonal Gaussian distribution.

        Args:
            size (Optional[int]): Number of samples to generate.

        Return:
            Union[Sequence[np.ndarray], np.ndarray]: Size number of Gaussian diagonal samples.

        """
        if size is None:
            rv = self.rng.randn(self.dist.dim)
            rv *= np.sqrt(self.dist.covar)
            rv += self.dist.mu
            return rv
        else:
            return [self.sample() for i in range(size)]


class DiagonalGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """DiagonalGaussianAccumulator object for aggregating sufficient statistics from iid observations.

    Attributes:
         dim (Optional[int]): Optional dimension of Gaussian.
         count (float): Used for tracking weighted observations counts.
         sum (np.ndarray): Sum of observation vectors.
         sum2 (np.ndarray): Sum of squared observation vectors.
         keys (Tuple[Optional[str], Optional[str]]): Key for mean and covariance.
         name (Optional[str]): Name for object. 

    """

    def __init__(self, dim: Optional[int] = None, keys: Tuple[Optional[str], Optional[str]] = (None, None), name: Optional[str] = None)-> None:
        """DiagonalGaussianAccumulator object.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            keys (Optional[str]): Set keys for merging sufficient statistics.
            name (Optional[str]): Name for object. 

        """
        self.dim = dim
        self.count = 0.0
        self.count2 = 0.0
        self.sum = vec.zeros(dim) if dim is not None else None
        self.sum2 = vec.zeros(dim) if dim is not None else None
        self.keys = keys
        self.name = name

    def update(self, x: Union[Sequence[float], np.ndarray], weight: float,
               estimate: Optional[DiagonalGaussianDistribution]) -> None:

        if self.dim is None:
            self.dim = len(x)
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        x_weight = x * weight
        self.count += weight
        self.count2 += weight
        self.sum += x_weight
        x_weight *= x
        self.sum2 += x_weight

    def initialize(self, x: Union[Sequence[float], np.ndarray], weight: float, rng: RandomState) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: 'DiagonalGaussianEncodedDataSequence', weights: np.ndarray, estimate: Optional[DiagonalGaussianDistribution]) -> None:
        if self.dim is None:
            self.dim = len(x.data[0])
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        x_weight = np.multiply(x.data.T, weights)
        self.count += weights.sum()
        self.count2 += weights.sum()
        self.sum += x_weight.sum(axis=1)
        x_weight *= x.data.T
        self.sum2 += x_weight.sum(axis=1)

    def seq_initialize(self, x: 'DiagonalGaussianEncodedDataSequence', weights: np.ndarray, rng: RandomState) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[np.ndarray, np.ndarray, float, float]) -> 'DiagonalGaussianAccumulator':
        if suff_stat[0] is not None and self.sum is not None:
            self.sum += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]
            self.count2 += suff_stat[3]

        elif suff_stat[0] is not None and self.sum is None:
            self.sum = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]
            self.count2 = suff_stat[3]

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        return self.sum, self.sum2, self.count, self.count2

    def from_value(self, x: Tuple[np.ndarray, np.ndarray, float, float]) -> 'DiagonalGaussianAccumulator':
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        self.count2 = x[3]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        mean_key, cov_key = self.keys
        if mean_key is not None:
            if mean_key in stats_dict:
                xw, wcnt = stats_dict[mean_key]
                self.sum += xw
                self.count += wcnt
            else:
                stats_dict[mean_key] = (self.sum, self.count)

        if cov_key is not None:
            if cov_key in stats_dict:
                x2w, wcnt2 = stats_dict[cov_key]
                self.sum2 += x2w
                self.count2 += wcnt2
            else:
                stats_dict[cov_key] = (self.sum2, self.count2)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        mean_key, cov_key = self.keys
        if mean_key is not None:
            if mean_key in stats_dict:
                xw, wcnt = stats_dict[mean_key]
                self.sum = xw
                self.count = wcnt

        if cov_key is not None:
            if cov_key in stats_dict:
                x2w, wcnt2 = stats_dict[cov_key]
                self.sum2 = x2w
                self.count2 = wcnt2

    def acc_to_encoder(self) -> 'DiagonalGaussianDataEncoder':
        return DiagonalGaussianDataEncoder(dim=self.dim)


class DiagonalGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """DiagonalGaussianAccumulatorFactory object for creating DiagonalGaussianAccumulator objects.

    Attributes:
        dim (Optional[int]): Optional dimension of Gaussian.
        keys (Optional[str], Optional[str]): Key for mean and covariance.
        name (Optional[str]): Name for object. 

    """

    def __init__(self, dim: Optional[int] = None, keys: Tuple[Optional[str], Optional[str]] = (None, None), name: Optional[str] = None) -> None:
        """DiagonalGaussianAccumulatorFactory object.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            keys (Optional[str]): Set keys for merging sufficient statistics.
            name (Optional[str]): Name for object. 

        """
        self.dim = dim
        self.keys = keys
        self.name = name

    def make(self) -> 'DiagonalGaussianAccumulator':
        return DiagonalGaussianAccumulator(dim=self.dim, keys=self.keys, name=self.name)


class DiagonalGaussianEstimator(ParameterEstimator):
    """DiagonalGaussianEstimator object for estimating diagonal Gaussian distributions from aggregated sufficient
        statistics.

    Attributes:
        name (Optinal[str]): Name for object instance.
        dim (int): Dimension of Gaussian, either set of determined from suff_stat arg.
        prior_mu (Optional[np.ndarray]): Set from suff_stat[0].
        prior_covar ((Optional[np.ndarray]): Set from suff_stat[1].
        pseudo_count (Tuple[Optional[float], Optional[float]]): Re-weight the sum of observations and sum of
            squared observations in estimation.
        keys (Optional[str], Optional[str]): Key for mean and covariance

    """

    def __init__(self, dim: Optional[int] = None, pseudo_count: Tuple[Optional[float], Optional[float]] = (None, None),
                 suff_stat: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
                 name: Optional[str] = None, keys: Tuple[Optional[str], Optional[str]]= (None, None)) -> None:
        """DiagonalGaussianEstimator object.

        Args:
            dim (Optional[int]): Optional dimension of Gaussian.
            pseudo_count (Tuple[Optional[float], Optional[float]]): Re-weight the sum of observations and sum of
                squared observations in estimation.
            suff_stat (Tuple[Optional[np.ndarray], Optional[np.ndarray]]): Sum of observations and sum of squared
                observations both having same dimension.
            name (Optinal[str]): Set name for object instance.
            keys (Optional[str], Optional[str]): Key for mean and covariance

        """
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and all(isinstance(k, (str, type(None))) for k in keys)
        ):
            self.keys = keys
        else:
            raise TypeError("DiagonalGaussianEstimator requires keys (Tuple[Optional[str], Optional[str]]).")

        dim_loc = dim if dim is not None else (
            (None if suff_stat[1] is None else int(np.sqrt(np.size(suff_stat[1])))) if suff_stat[0] is None else len(
                suff_stat[0]))

        self.name = name
        self.dim = dim_loc
        self.pseudo_count = pseudo_count
        self.prior_mu = None if suff_stat[0] is None else np.reshape(suff_stat[0], dim_loc)
        self.prior_covar = None if suff_stat[1] is None else np.reshape(suff_stat[1], dim_loc)
        self.keys = keys

    def accumulator_factory(self) -> 'DiagonalGaussianAccumulatorFactory':
        return DiagonalGaussianAccumulatorFactory(dim=self.dim, keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, np.ndarray, float]) \
            -> 'DiagonalGaussianDistribution':

        nobs = suff_stat[2]
        pc1, pc2 = self.pseudo_count

        if pc1 is not None and self.prior_mu is not None:
            mu = (suff_stat[0] + pc1 * self.prior_mu) / (nobs + pc1)
        else:
            mu = suff_stat[0] / nobs

        if pc2 is not None and self.prior_covar is not None:
            covar = (suff_stat[1] + (pc2 * self.prior_covar) - (mu * mu * nobs)) / (nobs + pc2)
        else:
            covar = (suff_stat[1] / nobs) - (mu * mu)

        #covar = np.maximum(covar, 1.0e-16)

        return DiagonalGaussianDistribution(mu, covar, name=self.name)


class DiagonalGaussianDataEncoder(DataSequenceEncoder):
    """DiagonalGaussianDataEncoder object for encoding sequences of iid diagonal-Gaussian observations.

    Attributes:
        dim (Optional[int]): Dimension of Gaussian distribution.

    """

    def __init__(self, dim: Optional[int] = None) -> None:
        """DiagonalGaussianDataEncoder object.

        Args:
            dim (Optional[int]): Dimension of Gaussian distribution.

        """
        self.dim = dim

    def __str__(self) -> str:
        return 'DiagonalGaussianDataEncoder(dim=' + str(self.dim) + ')'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DiagonalGaussianDataEncoder):
            return self.dim == other.dim
        else:
            return False

    def seq_encode(self, x: Sequence[Union[List[float], np.ndarray]]) -> 'DiagonalGaussianEncodedDataSequence':
        """Create DiagonalGaussianEncodedDataSequence object.

        Args:
            x (Sequence[Union[List[float], np.ndarray]]): Sequence of iid multivariate Gaussian observations.

        Return:
            DiagonalGaussianEncodedDataSequence

        """
        if self.dim is None:
            self.dim = len(x[0])
        xv = np.reshape(x, (-1, self.dim))

        return DiagonalGaussianEncodedDataSequence(data=xv)


class DiagonalGaussianEncodedDataSequence(EncodedDataSequence):
    """DiagonalGaussianEncodedDataSequence object for vectorized functions.

    Attributes:
        data (np.ndarray): Numpy array of obs.

    """

    def __init__(self, data: np.ndarray):
        """DiagonalGaussianEncodedDataSequence object for vectorized functions.

        Args:
            data (np.ndarray): Numpy array of obs.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'DiagonalGaussianEncodedDataSequence(data={self.data})'

