import numpy as np
import sys
from numpy.random import RandomState
from scipy.special import gammaln

import pysp.utils.vector as vec
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator
from pysp.utils.special import *


def dirichlet_param_solve(alpha, meanLogP, delta):
    dim = len(alpha)

    valid = np.bitwise_and(np.isfinite(alpha), alpha > 0)
    valid = np.bitwise_and(valid, np.isfinite(meanLogP))

    alpha = alpha[valid]
    mlp = meanLogP[valid]

    count = 0
    asum = alpha.sum()
    dalpha = (2 * delta) + 1

    while dalpha > delta:
        count += 1

        dasum = digamma(asum)
        old_alpha = alpha
        adj_alpha = mlp + dasum
        alpha = digammainv(adj_alpha)
        asum = np.sum(alpha)
        dalpha = np.abs(alpha - old_alpha).sum()
        dalpha /= asum

    if dim != alpha.size:
        rv = np.zeros(dim, dtype=float)
        rv[valid] = alpha
    else:
        rv = alpha

    return rv, count


def mpe(x0, f, eps):
    x1 = f(x0)
    x2 = f(x1)
    x3 = f(x2)
    X = np.asarray([x0, x1, x2, x3])
    s0 = x3
    s = s0
    res = np.abs(x3 - x2).sum()
    its_cnt = 2

    while res > eps:
        y = f(X[-1, :])
        dy = y - X[-1, :]
        U = (X[1:, :] - X[:-1, :]).T
        X2 = X[1:, :].T
        c = np.dot(np.linalg.pinv(U), dy)
        c *= -1
        s = (np.dot(X2, c) + y) / (c.sum() + 1)

        res = np.abs(s - s0).sum()
        s0 = s
        X = np.concatenate((X, np.reshape(y, (1, -1))), axis=0)
        its_cnt += 1

    return s, its_cnt


def alpha_seq_lambda(meanLogP):
    def next_alpha(currentAlpha):
        return digammainv(meanLogP + digamma(currentAlpha.sum()))

    return next_alpha


def find_alpha(current_alpha, mlp, thresh):
    f = alpha_seq_lambda(mlp)
    return mpe(current_alpha, f, thresh)


class DirichletDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, alpha, name=None):

        temp_alpha = np.asarray(alpha)
        temp_mask = temp_alpha <= 0

        self.dim = len(alpha)
        self.alpha = temp_alpha
        self.alpha_ma = ~temp_mask
        self.logConst = sum(gammaln(alpha)) - gammaln(sum(alpha))
        self.has_invalid = np.any(temp_mask)
        self.name = name

    def __str__(self):
        s1 = repr(list(self.alpha))
        s2 = repr(self.name)
        return 'DirichletDistribution(%s, name=%s)' % (s1, s2)

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):
        xx = np.asarray(x)
        zz = np.bitwise_or(xx > 0, self.alpha_ma)
        cnt = np.count_nonzero(zz)

        if cnt == self.dim:
            rv = np.dot(np.log(x), self.alpha - 1.0)
            rv -= self.logConst
        elif cnt == 0:
            rv = 0.0
        else:
            rv = np.dot(np.log(xx[zz]), self.alpha[zz] - 1.0)
            rv -= self.logConst

        return rv

    def seq_log_density(self, x):
        rv = np.dot(x[0], self.alpha - 1.0)
        rv -= self.logConst
        return rv

    def seq_encode(self, x):
        rv = np.asarray(x).copy()

        # TODO: Add warning for invalid values

        rv2 = np.maximum(rv, sys.float_info.min)
        np.log(rv2, out=rv2)
        return rv2, rv, rv * rv

    def sampler(self, seed=None):
        return DirichletSampler(self, seed)

    def estimator(self, pseudo_count=None):

        if pseudo_count is None:
            return DirichletEstimator(dim=self.dim, name=self.name)
        else:
            return DirichletEstimator(dim=self.dim, pseudo_count=pseudo_count,
                                      suff_stat=log(self.alpha / sum(self.alpha)), name=self.name)


class DirichletSampler(object):

    def __init__(self, dist, seed):
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):

        alpha = self.dist.alpha
        has_invalid = self.dist.has_invalid
        alpha_ma = self.dist.alpha_ma

        if has_invalid:
            if size is None:
                rv = np.zeros(alpha.size)
                rv[alpha_ma] = self.rng.dirichlet(alpha=alpha[alpha_ma])
            else:
                rv = np.zeros((size, alpha.size))
                rv[:, alpha_ma] = self.rng.dirichlet(alpha=alpha[alpha_ma], size=size)

            return rv
        else:
            return self.rng.dirichlet(alpha=self.dist.alpha, size=size)


class DirichletAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, dim, keys=None):
        self.dim = dim
        self.sumOfLogs = np.zeros(dim)
        self.sum = np.zeros(dim)
        self.sum2 = np.zeros(dim)
        self.counts = 0
        self.key = keys

    def update(self, x, weight, estimate):
        z = x > 0
        if np.all(z):
            self.sumOfLogs += log(x) * weight
            self.sum += weight * x
            self.sum2 += weight * x * x
            self.counts += weight
        else:
            self.sumOfLogs[z] += log(x[z]) * weight
            self.sum += weight * x
            self.sum2 += weight * x * x
            self.counts += weight

    def get_seq_lambda(self):
        return [self.seq_update]

    def seq_update(self, x, weights, estimate):
        self.sumOfLogs += np.dot(weights, x[0])
        self.counts += weights.sum()
        self.sum += np.dot(weights, x[1])
        self.sum2 += np.dot(weights, x[2])

    def combine(self, suff_stat):
        self.sumOfLogs += suff_stat[1]
        self.sum += suff_stat[2]
        self.sum2 += suff_stat[3]
        self.counts += suff_stat[0]
        return self

    def value(self):
        return self.counts, self.sumOfLogs, self.sum, self.sum2

    def from_value(self, x):
        self.counts = x[0]
        self.sumOfLogs = x[1]
        self.sum = x[2]
        self.sum2 = x[3]

    def key_merge(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class DirichletEstimator(ParameterEstimator):

    def __init__(self, dim, pseudo_count=None, suff_stat=None, delta=1.0e-8, keys=None, use_mpe=False, name=None):
        self.dim = dim
        self.pseudo_count = pseudo_count
        self.delta = delta
        self.suff_stat = suff_stat
        self.keys = keys
        self.use_mpe = use_mpe
        self.name = name

    def accumulatorFactory(self):
        dim = self.dim
        keys = self.keys
        obj = type('', (object,), {'make': lambda self: DirichletAccumulator(dim, keys)})()
        return (obj)

    def estimate(self, nobs, suff_stat):

        nobs, sum_of_logs, sum_v, sum_v2 = suff_stat
        dim = len(sum_of_logs)

        if self.pseudo_count is not None and self.suff_stat is None:
            c1 = digamma(one) - digamma(dim)
            c2 = sum_of_logs + c1 * self.pseudo_count
            initialEstimate = c2 * (dim / sum(c2))
            meanLogP = c2 / (nobs + self.pseudo_count)

        elif self.pseudo_count is not None and self.suff_stat is not None:
            c2 = sum_of_logs + self.suff_stat * self.pseudo_count
            initialEstimate = c2 * (dim / sum(c2))
            meanLogP = c2 / (nobs + self.pseudo_count)

        else:

            sum_v = sum_v / nobs
            sum_v2 = sum_v2 / nobs
            sum_v[-1] = 1.0 - sum_v[:-1].sum()

            '''
            #initialConst = (sum_v[0]-sum_v2[0])/(sum_v2[0]-sum_v[0]*sum_v[0])
            initialConst1 = (sum_v - sum_v2).mean()
            initialConst2 = (sum_v2 - sum_v*sum_v).mean()

            if initialConst2 > 0 and initialConst1 > 0:
                initialEstimate = (initialConst1/initialConst2)*sum_v
            else:
                initialEstimate = sum_of_logs * (dim / sum(sum_of_logs))

            #initialEstimate = sum_of_logs*(dim/sum(sum_of_logs))

            '''
            initialEstimate = sum_v

            meanLogP = sum_of_logs / nobs

        if nobs == 1.0:
            return DirichletDistribution(initialEstimate, name=self.name)

        else:

            if self.use_mpe:
                alpha, its_cnt = find_alpha(np.asarray(initialEstimate), meanLogP, self.delta)
            else:
                alpha, its_cnt = dirichlet_param_solve(np.asarray(initialEstimate), meanLogP, self.delta)

            return DirichletDistribution(alpha, name=self.name)

