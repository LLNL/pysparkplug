"""Create, estimate, and sample from a Categorical distribution.

Defines the CategoricalDistribution, CategoricalSampler, CategoricalAccumulatorFactory, CategoricalAccumulator,
CategoricalEstimator, and the CategoricalDataEncoder classes for use with pysparkplug.

"""
import numpy as np
import math
from typing import Dict, Optional, Tuple, Any, TypeVar, Union, List, Sequence
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence
from numpy.random import RandomState

T = TypeVar('T')


class CategoricalDistribution(SequenceEncodableProbabilityDistribution):
    """Defines a CategoricalDistribution object for data type T.

    Attributes:
        name (Optional[str]): Assigns a name to the CategoricalDistribution object.
        pmap (Dict[Any, float]): Keys (x_i) are the support of the categorical, the value is the probability of
            the key (p_i).
        default_value (float): Value for prob of observation outside support of CategoricalDistribution, default to
            0.0.
        no_default (bool): True if a non-zero default value is given.
        log_default_value (float): log(default_value).
        log1p_default_value (float): log(1+default_value).
        keys (Optional[str]): Key for distribution

    """
    def __init__(self, pmap: Dict[Any, float], default_value: float = 0.0, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Create an instance of CategoricalDistribution object.

        Args:
            pmap (Dict[Any, float]): Keys (x_i) are the support of the categorical, the value is the probability of
                the key (p_i).
            default_value float: Value for prob of observation outside support of CategoricalDistribution.
            name (Optional[str]): Assigns a name to the CategoricalDistribution object.
            keys (Optional[str]): Key for distribution

        """
        self.name = name
        self.pmap = pmap
        self.no_default = default_value != 0.0
        self.default_value = max(0.0, min(default_value, 1.0))
        self.log_default_value = float(-np.inf if default_value == 0 else math.log(default_value))
        self.log1p_default_value = float(math.log1p(default_value))
        self.keys = keys

    def __str__(self) -> str:
        """Object string with member variables for CategoricalDistribution.

        Returns:
            String with pmap, defualt_value, and name printed.

        """
        s1 = ', '.join(['%s: %s' % (repr(k), repr(float(v))) for k, v in sorted(self.pmap.items(), key=lambda u: u[0])])
        s2 = repr(self.default_value)
        s3 = repr(self.name)
        s4 = repr(self.keys)

        return 'CategoricalDistribution({%s}, default_value=%s, name=%s, keys=%s)' % (s1, s2, s3, s4)

    def density(self, x: Any) -> float:
        """Density evaluation of CategoricalDistribution.

        Args:
            x (Any): Evaluate CategoricalDistribution density value at x.

        Returns:
            float: density value at x

        """
        return self.pmap.get(x, self.default_value) / (1.0 + self.default_value)

    def log_density(self, x: Any) -> float:
        """Log-Density evaluation of CategoricalDistribution.

        Args:
            x (Any): Evaluate CategoricalDistribution density value at x.

        Returns:
            float: Log-density of Categorical distribution evaluated at x.

        """
        return np.log(self.pmap.get(x, self.default_value)) - self.log1p_default_value

    def seq_log_density(self, x: 'CategoricalEncodedDataSequence') -> np.ndarray:
        if not isinstance(x, CategoricalEncodedDataSequence):
            raise Exception('CategoricalDistribution.seq_log_density() requires CategoricalEncodedDataSequence.')

        with np.errstate(divide='ignore'):
            xs, val_map_inv = x.data
            mapped_log_prob = np.asarray([self.pmap.get(u, self.default_value) for u in val_map_inv], dtype=np.float64)
            np.log(mapped_log_prob, out=mapped_log_prob)
            mapped_log_prob -= self.log1p_default_value
            rv = mapped_log_prob[xs]

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'CategoricalSampler':
        """Creates CategoricalSampler for sampling from CategoricalDistribution.

        Args:
            seed (Optional[int]): Seed for setting random number generator used to sample.

        Returns:
            CategoricalSampler object.

        """
        return CategoricalSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'CategoricalEstimator':
        """Creates a CategoricalEstimator for estimating parameters of CategoricalDistribution.

        Args:
            pseudo_count (Optional[float]): If set, inflates counts for currently set sufficient statistic (pmap).

        Returns:
            CategoricalEstimator object.
        """
        if pseudo_count is None:
            return CategoricalEstimator(name=self.name, keys=self.keys)

        else:
            return CategoricalEstimator(pseudo_count=pseudo_count, suff_stat=self.pmap, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'CategoricalDataEncoder':
        return CategoricalDataEncoder()

class CategoricalSampler(DistributionSampler):
    """CategoricalSampler object used to generate samples from CategoricalDistribution.

    Attributes:
         rng (RandomState): RandomState with seed set to seed if provided. Else just RandomState().
         levels (List[Any]): Category labels for the CategoricalDistribution.
         probs (List[float]): Probabilities for each category in CategoricalDistribution.
         num_levels (int): Total number of categories. I.e. len(levels).

    """

    def __init__(self, dist: CategoricalDistribution, seed: Optional[int] = None) -> None:
        """CategoricalSampler object used to generate samples from CategoricalDistribution.

        Args:
            dist (CategoricalDistribution): CategoricalDistribution used to draw samples from.
            seed (Optional[int]): Seed for setting random number generator used to sample.

        """
        self.rng = RandomState(seed)
        temp            = list(dist.pmap.items())
        self.levels     = [u[0] for u in temp]
        self.probs      = [u[1] for u in temp]
        self.num_levels = len(self.levels)

    def sample(self, size: Optional[int] = None) -> Union[Any, List[Any]]:
        """Draw size-number of samples from CategoricalSampler object.

        If size is not provided, size is assumed = 1. If size > 1, a list is returned.

        Args:
            size (Optional[int]): Number of samples to be draw. If size is None, size = 1.

        Returns:
            Union[Any, List[Any]]: List of levels if size > 1, else a single sample from levels with prob probs.

        """
        if size is None:
            idx = self.rng.choice(self.num_levels, p=self.probs, size=size)
            return self.levels[idx]

        else:
            levels = self.levels
            rv = self.rng.choice(self.num_levels, p=self.probs, size=size)

            return [levels[i] for i in rv]

class CategoricalAccumulator(SequenceEncodableStatisticAccumulator):
    """CategoricalAccumulator object used for aggregating sufficient statistics of CategoricalDistribution.

    Attributes:
        count_map (Dict[Any,float]): Keys (x_i) are the support of the categorical, the value is the weighted count
        of category observations.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """CategoricalAccumulator object used for aggregating sufficient statistics of CategoricalDistribution.

        Args:
            name (Optional[str]): Name for object.
            keys (Optional[str]): All CategoricalAccumulators with same keys will have suff-stats merged.

        """
        self.count_map = dict()
        self.name = name
        self.key = keys

    def update(self, x: Any, weight: float, estimate: Optional['CategoricalDistribution']) -> None:
        self.count_map[x] = self.count_map.get(x, 0.0) + weight

    def initialize(self, x: Any, weight: float, rng: RandomState) -> None:
        self.update(x, weight, None)

    def get_seq_lambda(self):
        return [self.seq_update]

    def seq_update(self, x: 'CategoricalEncodedDataSequence', weights: np.ndarray,
                   estimate: Optional['CategoricalDistribution']) -> None:
        inv_key_map = x.data[1]
        bcnt = np.bincount(x.data[0], weights=weights)

        if len(self.count_map) == 0:
            self.count_map = dict(zip(inv_key_map, bcnt))

        else:
            for i in range(0, len(bcnt)):
                self.count_map[inv_key_map[i]] += bcnt[i]

    def seq_initialize(self, x: 'CategoricalEncodedDataSequence', weights: np.ndarray, rng: Optional[RandomState]) -> None:
        return self.seq_update(x, weights, None)

    def combine(self, suff_stat: Dict[Any, float]) -> 'CategoricalAccumulator':
        for k, v in suff_stat.items():
            self.count_map[k] = self.count_map.get(k, 0.0) + v

        return self

    def value(self) -> Dict[Any, float]:
        return self.count_map.copy()

    def from_value(self, x: Dict[Any, float]) -> 'CategoricalAccumulator':
        self.count_map = x

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())

            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'CategoricalDataEncoder':
        return CategoricalDataEncoder()

class CategoricalAccumulatorFactory(StatisticAccumulatorFactory):
    """CategoricalAccumulatorFactory object.

    Attributes:
        keys (Optional[str]): Key for merging sufficient statistics.

    """

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """CategoricalAccumulatorFactory object used for lightweight construction of Accumulators.

        Args:
            name (Optional[str]): name for object.
            keys (Optional[str]): Declare keys for merging sufficient statistics of CategoricalAccumulators.

        """
        self.name = name 
        self.keys = keys

    def make(self) -> 'CategoricalAccumulator':
        return CategoricalAccumulator(name=self.name, keys=self.keys)

class CategoricalEstimator(ParameterEstimator):
    """CategoricalEstimator used to estimate CategoricalDistribution.

    Attributes:
        pseudo_count (Optional[float]): Inflate sufficient statistic counts by pseudo_count.
        suff_stat (Optional[Dict[Any, float]]): Dictionary with category labels and probabilities as values.
        default_value (bool): True is default value should be set.
        name (Optional[str]): Assign name to be passed to Distribution, Accumulator, ect.
        keys (Optional[str]): Assign key to Estimator designating all same key estimators to later be combined,
            in accumulation.

    """

    def __init__(self,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Dict[Any, float]] = None,
                 default_value: bool = False,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """CategoricalEstimator used to estimate CategoricalDistribution.

        Args:
            pseudo_count (Optional[float]): Inflate sufficient statistic counts by pseudo_count.
            suff_stat (Optional[Dict[Any, float]]): Dictionary with category labels and probabilities as values.
            default_value (bool): True is default value should be set.
            name (Optional[str]): Assign name to be passed to Distribution, Accumulator, ect.
            keys (Optional[str]): Assign key to Estimator designating all same key estimators to later be combined,
                in accumulation.

        """
        if isinstance(keys, str) or keys is None:
            self.keys = keys
        else:
            raise TypeError("CategoricalEstimator requires keys to be of type 'str'.")

        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.default_value = default_value
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> 'CategoricalAccumulatorFactory':
        return CategoricalAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Dict[Any, float]) -> 'CategoricalDistribution':
        """Estimate a CategoricalDistribution from suff_stat value.

        If default_value is True, we estimate a default value from the suff_stat counts. Else, it is set to 0.0.

        pseudo_count is used to averaged over the number of levels and added to the corresponding counts.

        If suff_stat member value is None, estimate for CategoricalDistribution is formed from the suff_stat passed.
        Otherwise, the suff_stat member value is combined with the suff_stat values passed to estimate.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.estimate.
            suff_stat (Dict[Any, float]): Dict with categories as keys and counts as values from accumulated data.

        Returns:
            CategoricalDistribution: Estimated from passed in suff_stat value and sufficient statistic member variable
                (if it is not None).

        """
        stats_sum = sum(suff_stat.values())

        if self.default_value:
            if stats_sum > 0:
                default_value = 1.0/stats_sum
                default_value *= default_value

            else:
                default_value = 0.5
        else:
            default_value = 0.0

        if self.pseudo_count is None and self.suff_stat is None:
            nobs_loc = stats_sum

            if nobs_loc == 0.0:
                p_map = {k : 1.0/float(len(suff_stat)) for k in suff_stat.keys()}
            else:
                p_map = {k: v / nobs_loc for k, v in suff_stat.items()}

        elif self.pseudo_count is not None and self.suff_stat is None:
            nobs_loc = stats_sum
            pseudo_count_per_level = self.pseudo_count/len(suff_stat)
            adjusted_nobs = nobs_loc + self.pseudo_count

            for k, v in suff_stat.items():
                suff_stat[k] = (v + pseudo_count_per_level) / adjusted_nobs

            p_map = suff_stat

        else:
            suff_stat_sum = sum(self.suff_stat.values())

            levels = set(suff_stat.keys()).union(self.suff_stat.keys())
            adjusted_nobs = suff_stat_sum * self.pseudo_count + stats_sum

            p_map = {k: (suff_stat.get(k, 0) + self.suff_stat.get(k, 0) * self.pseudo_count) / adjusted_nobs for k in levels}

        return CategoricalDistribution(pmap=p_map, default_value=default_value, name=self.name)

class CategoricalDataEncoder(DataSequenceEncoder):
    """CategoricalDataEncoder for encoding Categorical data for use with vectorized "seq_" functions."""

    def __str__(self) -> str:
        return 'CategoricalDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, CategoricalDataEncoder)

    def seq_encode(self, x: Sequence[Any]) -> 'CategoricalEncodedDataSequence':
        """Sequence encode list of categories for use with vectorized "seq_" functions.

        Args:
            x (Sequence[Any]): List of category labels.

        Returns:
            CategoricalEncodedDataSequence

        """
        val_map_inv, uidx, xs = np.unique(x, return_index=True, return_inverse=True)
        val_map_inv = np.asarray([x[i] for i in uidx], dtype=object)

        return CategoricalEncodedDataSequence(data=(xs, val_map_inv))

class CategoricalEncodedDataSequence(EncodedDataSequence):
    """CategoricalEncodedDataSequence object.

    Attributes:
        data: (Tuple[np.ndarray, np.ndarray]): Inverse mapping of unique values, unique values.

    """

    def __init__(self, data: Tuple[np.ndarray, np.ndarray]):
        """CategoricalEncodedDataSequence object.

        Args:
            data: (Tuple[np.ndarray, np.ndarray]): Inverse mapping of unique values, unique values.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'CategoricalEncodedDataSequence(data={self.data})'
