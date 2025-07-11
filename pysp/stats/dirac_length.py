"""Create, estimate, and sample from a mixture over a dirac delta at v and a length distribution.

Defines the DiracMixtureDistribution, DiracMixtureSampler, DiracMixtureAccumulatorFactory,
DiracMixtureAccumulator, DiracMixtureEstimator, and the DiracMixtureDataEncoder classes for use with
pysparkplug.

The DiracMixtureDistribution is defined by the density of the form,

P(Y) = p*P_1(Y) + (1-p)*Delta_{v}(Y),

where P_1() is a length distribution with support on non-negative integers, or a subset of them, and Delta_{v}(x) = 1
if x = v, else 0.

"""
from typing import List, Union, Tuple, Any, Optional, TypeVar, Sequence, Dict

import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import maxrandint
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder, EncodedDataSequence

SS0 = TypeVar('SS0')  # Type of component suff_stat
key_type = Union[Tuple[str, str], Tuple[None, None]]


class DiracMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """DiracMixtureDistribution object defined by a length distribution, choice of dirac value, and p.

    Notes:
        dist is the base distribution with support on integer values.

    Attributes:
        p (float): Probability of being drawn from length distribution. Must be between 0 and 1.
        dist (SequenceEncodableProbabilityDistribution): Distribution with support on non-negative integers.
        name (Optional[str]): Name for object instance.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components of mixture.

    """

    def __init__(self, dist: SequenceEncodableProbabilityDistribution, p: float, v: int = 0,
                 name: Optional[str] = None, keys: Tuple[Optional[str], Optional[str]] = (None, None)):
        """DiracMixtureDistribution object defined by a length distribution, choice of dirac value, and p.

        Args:
            p (float): Probability of being drawn from length distribution. Must be between 0 and 1.
            dist (SequenceEncodableProbabilityDistribution): Distribution with support on non-negative integers.
            name (Optional[str]): Set name for object instance.
            keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components of mixture.

        """
        if not 0 < p <= 1:
            raise Exception('p must be between (0,1].')
        with np.errstate(divide='ignore'):
            self.p = p
            self.v = v
            self.log_p = np.log(p)
            self.log_1p = np.log1p(-p)
            self.dist = dist
            self.name = name
            self.keys = keys

    def __str__(self) -> str:
        s1 = repr(self.dist)
        s2 = repr(self.p)
        s3 = repr(self.v)
        s4 = repr(self.name)
        s5 = repr(self.keys)

        return 'DiracMixtureDistribution(dist=%s, p=%s, v=%s, name=%s, keys=%s)' % (s1, s2, s3, s4, s5)

    def density(self, x: int) -> float:
        """Evaluate density of length Dirac mixture distribution at observation x.

        See log_density() for details.

        Args:
            x (int): Integer value.

        Returns:
            float: Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Evaluate the log-density of length Dirac mixture distribution at observation x.

        log(P(x)) = log( p*P_1(x) + (1-p)*Delta_{v}(x) ),

        Args:
            x (int): Integer value.

        Returns:
            float: log-density at x.

        """
        rv0 = self.log_p + self.dist.log_density(x)

        if x == self.v:
            c1 = self.log_1p
            if c1 > rv0:
                rv = np.log1p(np.exp(rv0-c1)) + c1
            else:
                rv = np.log1p(np.exp(c1-rv0)) + rv0
        else:
            rv = rv0

        return rv

    def component_log_density(self, x: int) -> np.ndarray:
        """Evaluate the log density for the components of the dirac mixture.

        Notes:
            The components are Dirac spike and `dist`.

        Args:
            x (int): Integer value with support on mixture components.

        Returns:
            float

        """
        rv = np.zeros(2, dtype=np.float64)
        rv[0] = self.dist.log_density(x)
        if x != self.v:
            rv[1] = -np.inf
        return rv

    def posterior(self, x: int) -> np.ndarray:
        """Evaluate the posterior for the components of the dirac mixture.

        Notes:
            The components are Dirac spike and `dist`.

        Args:
            x (int): Integer value with support on mixture components.

        Returns:
            float

        """
        comp_log_density = self.component_log_density(x)
        if comp_log_density[1] == -np.inf:
            return np.array([1, 0], dtype=np.float64)
        else:
            comp_log_density[0] += self.log_p
            comp_log_density[1] += self.log_1p

        max_val = np.max(comp_log_density)

        comp_log_density -= max_val
        np.exp(comp_log_density, out=comp_log_density)
        comp_log_density /= comp_log_density.sum()

        return comp_log_density

    def seq_component_log_density(self, x: 'DiracMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation the log density for the components of the dirac mixture.

        Notes:
            The components are Dirac spike and `dist`.

        Args:
            x (DiracMixtureEncodedDataSequence): EncodedDataSequence for DiracMixtureDistribution.

        Returns:
            np.ndarray

        """

        if not isinstance(x, DiracMixtureEncodedDataSequence):
            raise Exception('DiracMixtureEncodedDataSequence required for `seq_` function calls.')

        sz, idx_v, idx_nv, enc_x = x.data
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        ll_mat[:, 0] += self.dist.seq_log_density(enc_x)
        ll_mat[idx_nv, 1] = -np.inf

        return ll_mat

    def seq_log_density(self, x: 'DiracMixtureEncodedDataSequence') -> np.ndarray:
        if not isinstance(x, DiracMixtureEncodedDataSequence):
            raise Exception('DiracMixtureEncodedDataSequence required for `seq_` function calls.')

        sz, idx_v, idx_nv, enc_x = x.data
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        ll_mat[:, 0] += self.dist.seq_log_density(enc_x) + self.log_p
        ll_mat[idx_nv, 1] = -np.inf
        ll_mat[idx_v, 1] += self.log_1p

        ll_max = ll_mat.max(axis=1, keepdims=True)
        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            return ll_sum.flatten()

        else:

            ll_mat = ll_mat[good_rows, :]
            ll_max = ll_max[good_rows]
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)

            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            rv = np.zeros(good_rows.shape, dtype=float)
            rv[good_rows] = ll_sum.flatten()
            rv[~good_rows] = -np.inf

            return rv

    def seq_posterior(self, x: 'DiracMixtureEncodedDataSequence') -> np.ndarray:
        """Vectorized evaluation the posterior for the components of the dirac mixture.

        Notes:
            The components are Dirac spike and `dist`.

        Args:
            x (DiracMixtureEncodedDataSequence): EncodedDataSequence for DiracMixtureDistribution.

        Returns:
            np.ndarray

        """

        if not isinstance(x, DiracMixtureEncodedDataSequence):
            raise Exception('DiracMixtureEncodedDataSequence required for `seq_` function calls.')

        sz, idx_v, idx_nv, enc_x = x.data
        rv = np.zeros((sz, 2), dtype=np.float64)
        rv[:, 0] += self.dist.seq_log_density(enc_x) + self.log_p
        rv[:, 1] = self.log_1p

        if len(idx_v) > 0:
            ll_mat = rv[idx_v, :]
            ll_max = np.max(ll_mat, axis=1, keepdims=True)
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_mat /= np.sum(ll_mat, axis=1, keepdims=True)

            rv[idx_v, :] = ll_mat
            
        rv[idx_nv, 0] = 1.0
        rv[idx_nv, 1] = 0.0

        return rv


    def sampler(self, seed: Optional[int] = None) -> 'DiracMixtureSampler':
        return DiracMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'DiracMixtureEstimator':

        if pseudo_count is not None:
            est = self.dist.estimator(pseudo_count)
            return DiracMixtureEstimator(
                estimator=est, 
                v=self.v, 
                pseudo_count=pseudo_count,
                suff_stat = self.p,
                name=self.name,
                keys=self.keys)
        else:
            est = self.dist.estimator()
            return DiracMixtureEstimator(estimator=est, v=self.v, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'DiracMixtureDataEncoder':
        dist_encoder = self.dist.dist_to_encoder()
        return DiracMixtureDataEncoder(encoder=dist_encoder, v=self.v)


class DiracMixtureSampler(DistributionSampler):
    """DiracMixtureSampler used to generate samples.

    Attributes:
        rng (RandomState): Seeded RandomState for sampling.
        p (np.ndarray): Prob of drawing from length distribution.
        dist_sampler (DistributionSampler): Sampler for the length distribution.
        v (int): Dirac location.
.
    """

    def __init__(self, dist: DiracMixtureDistribution, seed: Optional[int] = None) -> None:
        """DiracMixtureSampler object.

        Args:
            dist (DiracMixtureDistribution): Assign DiracMixtureDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        """
        rng_loc = np.random.RandomState(seed)
        self.rng = np.random.RandomState(rng_loc.randint(0, maxrandint))
        self.p = np.exp(dist.log_p)
        self.dist_sampler = dist.dist.sampler(seed=self.rng.randint(maxrandint))
        self.v = dist.v

    def sample(self, size: Optional[int] = None) -> Union[List[int], int]:
        """Draw iid samples from a DiracMixture distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            Int or List[int] depending on size = None or size (int).

        """
        comp_state = self.rng.binomial(n=1, size=size, p=self.p)

        if size is None:
            if comp_state == 0:
                return self.v
            else:
                return self.dist_sampler.sample()
        else:
            rv = np.zeros(size, dtype=np.int32)
            rv.fill(self.v)

            idx = np.flatnonzero(comp_state == 1)
            if len(idx) > 0:
                rv[idx] = np.asarray(self.dist_sampler.sample(size=len(idx)), dtype=np.int32)
            return list(rv)


class DiracMixtureAccumulator(SequenceEncodableStatisticAccumulator):
    """DiracMixtureAccumulator object for aggregating sufficient statistics.

    Attributes:
        accumulator (SequenceEncodableStatisticAccumulator): Accumulator object for distribution with integer support.
        comp_counts (np.ndarray): Sufficient statistics
        weight_key (Optional[str]): Key or weights of mixture.
        comp_key (Optional[str]): Key for components of mixture.
        v (int): Dirac spike value, defaults to 0.
        name (Optional[str]): Name for object.

    """

    def __init__(self, accumulator: SequenceEncodableStatisticAccumulator, v: int = 0,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None), name: Optional[str] = None):
        """DiracMixtureAccumulator object for aggregating sufficient statistics.

        Args:
            accumulator (SequenceEncodableStatisticAccumulator): Accumulator object for distribution with integer support.
            v (int): Dirac spike value, defaults to 0.
            keys (Tuple[Optional[str], Optional[str]]): Keys for weight and components.
            name (Optional[str]): Name for object.

        """
        self.accumulator = accumulator
        self.comp_counts = np.zeros(2, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]
        self.v = v
        self.name = name

        # Initializer seeds
        self._init_rng: bool = False
        self._acc_rng: Optional[RandomState] = None
        self._w_rng: Optional[RandomState] = None

    def seq_update(self, x: 'DiracMixtureEncodedDataSequence', weights: np.ndarray, estimate: 'DiracMixtureDistribution'):
        sz, idx_v, idx_nv, enc_x = x.data
        ll_mat = np.zeros((sz, 2), dtype=np.float64)

        if len(idx_v) == 0:
            ll_mat[:, 0] += weights

        else:
            ll_mat[:, 0] += estimate.dist.seq_log_density(enc_x) + estimate.log_p
            ll_mat[idx_nv, 0] = weights[idx_nv].copy()

            rv = ll_mat[idx_v, :]
            rv[:, 1] += estimate.log_1p

            rv_max = rv.max(axis=1, keepdims=True)
            bad_rows = np.isinf(rv.flatten())

            if np.any(bad_rows):
                rv[bad_rows, :] = np.array([estimate.log_p, estimate.log_1p], dtype=np.float64)
                rv_max[bad_rows] = np.max(np.asarray([estimate.log_p, estimate.log_1p]))
            rv -= rv_max

            np.exp(rv, out=rv)
            np.sum(rv, axis=1, keepdims=True, out=rv_max)
            np.divide(weights[idx_v, None], rv_max, out=rv_max)
            rv *= rv_max

            ll_mat[idx_v, :] = rv

        self.comp_counts += ll_mat.sum(axis=0)
        self.accumulator.seq_update(enc_x, ll_mat[:, 0], estimate.dist)

    def update(self, x: int, weight: float, estimate: 'DiracMixtureDistribution') -> None:
        posterior = estimate.posterior(x)
        posterior *= weight
        self.comp_counts += posterior

        self.accumulator.update(x, posterior[0], estimate.dist)

    def _rng_initialize(self, rng: RandomState):
        seeds = rng.randint(2 ** 31, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._init_rng = True

    def initialize(self, x: int, weight: float, rng: np.random.RandomState):
        if not self._init_rng:
            self._rng_initialize(rng)

        if x == self.v:
            ww = self._w_rng.dirichlet(np.ones(2)/4)
            self.accumulator.initialize(x, weight*ww[0], rng=self._acc_rng)
            self.comp_counts += ww
        else:
            self.accumulator.initialize(x, weight, rng=self._acc_rng)
            self.comp_counts[0] += weight

    def seq_initialize(self, x: 'DiracMixtureEncodedDataSequence', weights: np.ndarray, rng: np.random.RandomState) -> None:

        sz, xi_v, xi_nv, enc_x = x.data

        if not self._init_rng:
            self._rng_initialize(rng)

        sz = len(weights)
        keep_len = len(xi_v)
        ww = np.ones((sz, 2))

        if keep_len > 0:
            ww[xi_v, :] = self._w_rng.dirichlet(alpha=np.ones(2) / 4, size=keep_len)

        ww *= np.reshape(weights, (sz, 1))

        self.accumulator.seq_initialize(enc_x, weights=ww[:, 0], rng=self._acc_rng)
        self.comp_counts[0] += np.sum(ww[:, 0])
        self.comp_counts[1] += np.sum(ww[xi_v, 1])

    def combine(self, suff_stat: Tuple[np.ndarray, SS0]) -> 'DiracMixtureAccumulator':
        self.comp_counts += suff_stat[0]
        self.accumulator.combine(suff_stat[1])

        return self

    def value(self) -> Tuple[np.ndarray, Any]:
        return self.comp_counts, self.accumulator.value()

    def from_value(self, x: Tuple[np.ndarray, SS0]) -> 'DiracMixtureAccumulator':
        self.comp_counts = x[0]
        self.accumulator.from_value(x[1])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                stats_dict[self.comp_key].combine(self.accumulator.value())
            else:
                stats_dict[self.comp_key] = self.accumulator

        self.accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulator = acc

        self.accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'DiracMixtureDataEncoder':
        acc_encoder = self.accumulator.acc_to_encoder()
        return DiracMixtureDataEncoder(encoder=acc_encoder, v=self.v)


class DiracMixtureAccumulatorFactory(StatisticAccumulatorFactory):
    """DiracMixtureAccumulatorFactory object for creating DiracMixtureAccumulator objects.

    Attributes:
        factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for mixture components.
        v (int): Dirac integer value.
        keys (Tuple[Optional[str], Optional[str]): Keys for weights and mixture components.
        name (Optional[str]): Name for object.

    """

    def __init__(self, factory: StatisticAccumulatorFactory, v: int = 0,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None), name: Optional[str] = None) -> None:
        """DiracMixtureAccumulatorFactory object for creating DiracMixtureAccumulator objects.

        Args:
            factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for mixture components.
            v (int): Dirac integer value.
            keys (Tuple[Optional[str], Optional[str]): Keys for weights and mixture components.
            name (Optional[str]): Name for object.

        """
        self.factory = factory
        self.v = v
        self.keys = keys
        self.name = name

    def make(self) -> 'DiracMixtureAccumulator':
        return DiracMixtureAccumulator(accumulator=self.factory.make(), v=self.v, keys=self.keys, name=self.name)


class DiracMixtureEstimator(ParameterEstimator):
    """DiracMixtureEstimator object for estimating DiracMixtureDistribution.

    Notes:
        estimator passed for mixture should have support on the integers.

    Attributes:
        estimator (ParameterEstimator): Estimator for components of mixture. Should have support on integers.
        v (int): Spiked value. Defaults to 0.
        pseudo_count (Optional[float]): Regularize suff stats.
        suff_stat (Optional[float]): Regularize estimation on the Dirac probability.
        keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components of mixture.
        name (Optional[str]): Assign a name to the object.
        fixed_p_vec (Optional[np.ndarray]): Fix the Dirac spike probability.

    """

    def __init__(self, estimator: ParameterEstimator, v: int = 0, fixed_p: Optional[int] = None,
                 suff_stat: Optional[float] = None, pseudo_count: Optional[float] = None,
                 name: Optional[str] = None, keys: Tuple[Optional[str], Optional[str]] = (None, None)):
        """DiracMixtureEstimator object for estimating DiracMixtureDistribution.

        Args:
            estimator (ParameterEstimator): Estimator for components of mixture. Should have support on integers.
            v (int): Spiked value. Defaults to 0.
            fixed_p (Optional[int]): Fix the Dirac spike probability.
            suff_stat (Optional[float]): Regularize estimation on the Dirac probability.
            name (Optional[str]): Assign a name to the object.
            keys (Tuple[Optional[str], Optional[str]]): Keys for weights and components of mixture.

        """
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and all(isinstance(k, (str, type(None))) for k in keys)
        ):
            self.keys = keys
        else:
            raise TypeError("DiracMixtureEstimator requires keys (Tuple[Optional[str], Optional[str]]).")

        self.estimator = estimator
        self.v = v
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name
        self.fixed_p_vec = np.asarray([fixed_p, 1-fixed_p]) if fixed_p is not None and 0 < fixed_p <= 1 else None

    def accumulator_factory(self) -> 'DiracMixtureAccumulatorFactory':
        factory = self.estimator.accumulator_factory()
        return DiracMixtureAccumulatorFactory(factory=factory, v=self.v, keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, SS0]) -> 'DiracMixtureDistribution':
        counts, comp_suff_stats = suff_stat

        dist = self.estimator.estimate(counts[0], comp_suff_stats)

        if self.fixed_p_vec is not None:
            p = self.fixed_p_vec[0]

        elif self.pseudo_count is not None and self.suff_stat is None:
            w = counts + self.pseudo_count / 2
            w /= w.sum()
            p = w[0]

        elif self.pseudo_count is not None and self.suff_stat is not None:
            ss = np.array([self.suff_stat, 1-self.suff_stat])
            w = (counts + ss*self.pseudo_count) / (counts.sum() + self.pseudo_count)
            p = w[0]

        else:
            nobs_loc = counts.sum()

            if nobs_loc == 0:
                p = 0.5
            else:
                w = counts / counts.sum()
                p = w[0]

        return DiracMixtureDistribution(dist=dist, p=p, v=self.v, name=self.name)


class DiracMixtureDataEncoder(DataSequenceEncoder):
    """DiracMixtureDataEncoder object for encoding sequences of dirac mixture observations.


    Attributes:
        encoder (DataSequenceEncoder): DataSequenceEncoder for distribution with support on the integers.
        v (int): Dirac spike value, defaults to 0.

    """

    def __init__(self, encoder: DataSequenceEncoder, v: int = 0) -> None:
        """DiracMixtureDataEncoder object.

        Args:
            encoder (DataSequenceEncoder): DataSequenceEncoder for distribution with support on the integers.
            v (int): Dirac spike value, defaults to 0.

        """
        self.encoder = encoder
        self.v = v

    def __str__(self) -> str:
        return 'DiracMixtureDataEncoder(encoder=%s, v=%s)' % (repr(self.encoder), repr(self.v))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DiracMixtureDataEncoder):
            if other.encoder == self.encoder:
                return other.v == self.v
            else:
                return False
        else:
            return False

    def seq_encode(self, x: Sequence[int]) -> 'DiracMixtureEncodedDataSequence':
        x = np.asarray(x, dtype=np.int32)
        xi_v = np.flatnonzero(x == self.v).astype(np.int32)
        xi_nv = np.flatnonzero(x != self.v).astype(np.int32)

        return DiracMixtureEncodedDataSequence(data=(len(x), xi_v, xi_nv, self.encoder.seq_encode(x)))


class DiracMixtureEncodedDataSequence(EncodedDataSequence):
    """DiracMixtureEncodedDataSequence object for use with vectorized function calls.

    Notes:
        E = Tuple[int, np.ndarray, np.ndarray, np.ndarray, EncodedDataSequence]
        EncodedDataSequence must be from a length distribution with support on integers.

    Attributes:
        data (E): Encoded sequence of iid Dirac mixture observations.

    """

    def __init__(self, data: Tuple[int, np.ndarray, np.ndarray, EncodedDataSequence]):
        """DiracMixtureEncodedDataSequence object.

        Args:
            data (E): Encoded sequence of iid Dirac mixture observations.

        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f'DiracMixtureEncodedDataSequence(data={self.data})'

