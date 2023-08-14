from typing import Optional, List, Tuple
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
import numba
import math
from pysp.utils.optsutil import countByValue
from pysp.arithmetic import maxrandint


class IntegerHiddenAssociationDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, state_prob_mat, cond_weights, alpha=0.0, given_dist=None, len_dist=None, name=None, keys=None):

        self.cond_weights   = np.asarray(cond_weights, dtype=np.float64)
        self.state_prob_mat = np.asarray(state_prob_mat, dtype=np.float64)
        self.len_dist       = len_dist
        self.prev_dist      = given_dist
        self.has_prev_dist  = given_dist is not None
        self.num_vals2      = self.state_prob_mat.shape[1]
        self.num_vals1      = self.cond_weights.shape[0]
        self.num_states     = self.state_prob_mat.shape[0]
        self.alpha          = alpha
        self.name           = name
        self.keys           = keys
        self.init_prob_vec  = np.empty(0, dtype=np.float64)

    def __str__(self):
        #s1 = ','.join(map(str, self.init_prob_vec))
        s1 = ','.join(['[' + ','.join(map(str, self.state_prob_mat[i,:])) + ']' for i in range(len(self.state_prob_mat))])
        s2 = ','.join(['[' + ','.join(map(str, self.cond_weights[i, :])) + ']' for i in range(len(self.cond_weights))])
        s3 = str(self.alpha)
        s4 = repr(self.prev_dist) if self.prev_dist is None else str(self.prev_dist)
        s5 = str(self.len_dist)
        s6 = repr(self.name)
        s7 = repr(self.keys)

        return 'IntegerHiddenAssociationDistribution([%s], [%s], alpha=%s, given_dist=%s, len_dist=%s, name=%s, keys=%s)' % (s1, s2, s3, s4, s5, s6, s7)

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):

        nw = self.num_vals2
        a = self.alpha / nw
        b = 1 - self.alpha

        cx = np.asarray([u[1] for u in x[0]])
        vx = np.asarray([u[0] for u in x[0]])
        cy = np.asarray([u[1] for u in x[1]])
        vy = np.asarray([u[0] for u in x[1]])

        n1 = np.sum(cx)
        n2 = np.sum(cy)

        X = self.cond_weights[vx, :].T * (cx/np.sum(cx))
        X = np.dot(X.T, self.state_prob_mat[:, vy])*b + a
        rv = np.dot(np.log(np.sum(X, axis=0)), cy)
        #rv += np.dot(np.log(self.init_prob_vec[vx]), cx)

        if self.has_prev_dist:
            rv += self.prev_dist.log_density(x[0])

        if self.len_dist is not None:
            rv += self.len_dist.log_density(n2)

        return rv

    def seq_log_density(self, x):

        nw = self.num_vals2
        a = self.alpha / nw
        b = 1 - self.alpha

        if x[1] is None:

            xx = x[0]

            rv = np.zeros(len(xx[0]), dtype=np.float)

            for i, entry in enumerate(xx[0]):
                vx, cx, vy, cy = entry

                X = self.cond_weights[vx, :].T * (cx / np.sum(cx))
                X = np.dot(X.T, self.state_prob_mat[:, vy]) * b + a
                rv[i] = np.dot(np.log(np.sum(X, axis=0)), cy)
                #rv[i] += np.dot(np.log(self.init_prob_vec[vx]), cx)

            if self.prev_dist is not None:
                rv += self.prev_dist.seq_log_density(xx[1])

            if self.len_dist is not None:
                rv += self.len_dist.seq_log_density(xx[2])

        else:

            (s0, s1, x0, x1, c0, c1, w0), xv, nn = x[1]

            rv = np.zeros(len(s0), dtype=np.float64)
            t0 = np.concatenate([[0], s0]).cumsum().astype(np.int32)
            t1 = np.concatenate([[0], s1]).cumsum().astype(np.int32)
            max_len = s0.max()
            numba_seq_log_density(self.num_states, max_len, t0, t1, x0, x1, c0, c1, w0, self.cond_weights, self.state_prob_mat, self.init_prob_vec, a, b, rv)

            if self.prev_dist is not None:
                rv += self.prev_dist.seq_log_density(xv)
            if self.len_dist is not None:
                rv += self.len_dist.seq_log_density(nn)

        return rv

    def _seq_encode(self, x):

        rv = []
        nn = []

        for xx in x:
            rv0 = []
            nn0 = []
            for cvec in xx:
                rv0.append(np.asarray([v for v, c in cvec], dtype=int))
                rv0.append(np.asarray([c for v, c in cvec], dtype=float))
                nn0.append(np.sum(rv0[-1]))

            rv.append(tuple(rv0))
            nn.append(tuple(nn0))

        if self.len_dist is not None:
            nn = self.len_dist.seq_encode(nn)
        else:
            nn = None

        if self.prev_dist is not None:
            xv = self.prev_dist.seq_encode([x[0] for x in x])
        else:
            xv = None

        return (rv, xv, nn), None


    def seq_encode(self, x):

        x1  = []
        x0  = []
        s1  = []
        s0  = []
        c0  = []
        c1  = []
        w0  = []
        nn  = []

        for i, xx in enumerate(x):

            xx0 = [v for v, c in xx[0]]
            cc0 = [c for v, c in xx[0]]
            xx1 = [v for v, c in xx[1]]
            cc1 = [c for v, c in xx[1]]

            x0.extend(xx0)
            x1.extend(xx1)
            c0.extend(cc0)
            c1.extend(cc1)
            w0.append(sum(cc0))
            s1.append(len(xx1))
            s0.append(len(xx0))
            nn.append(sum(cc1))

        if self.len_dist is not None:
            nn = self.len_dist.seq_encode(nn)
        else:
            nn = None

        if self.prev_dist is not None:
            xv = self.prev_dist.seq_encode([x[0] for x in x])
        else:
            xv = None

        x0 = np.asarray(x0, dtype=np.int32)
        x1 = np.asarray(x1, dtype=np.int32)
        c0 = np.asarray(c0, dtype=np.float64)
        c1 = np.asarray(c1, dtype=np.float64)
        s0 = np.asarray(s0, dtype=np.int32)
        s1 = np.asarray(s1, dtype=np.int32)
        w0 = np.asarray(w0, dtype=np.float64)

        return None, ((s0, s1, x0, x1, c0, c1, w0), xv, nn)

    def sampler(self, seed=None):
        return IntegerHiddenAssociationSampler(self, seed)


class IntegerHiddenAssociationSampler(object):

    def __init__(self, dist: IntegerHiddenAssociationDistribution, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.size_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

        if self.dist.prev_dist is not None:
            self.prev_sampler = self.dist.prev_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample_given(self, x):

        slen = self.size_sampler.sample()
        rng = np.random.RandomState(self.rng.randint(0, maxrandint))

        x0 = np.asarray([xx[0] for xx in x])
        x1 = np.asarray([xx[1] for xx in x], dtype=float)
        s1 = np.sum(x1)

        if s1 > 0:
            x1 /= s1
        else:
            return []

        v2 = []
        z1 = rng.choice(len(x0), p=x1, replace=True, size=slen)
        ns = self.dist.num_states
        nw = self.dist.num_vals2

        for zz1 in z1:

            if rng.rand() >= self.dist.alpha:
                u = rng.choice(ns, p=self.dist.cond_weights[x0[zz1], :])
                v2.append(rng.choice(nw, p=self.dist.state_prob_mat[u, :]))
            else:
                v2.append(rng.choice(nw))

        return list(countByValue(v2).items())

    def sample(self, size: Optional[int] = None):

        if size is None:
            x = self.prev_sampler.sample()
            return x, self.sample_given(x)
        else:
            return [self.sample() for i in range(size)]


class IntegerHiddenAssociationAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals1, num_vals2, num_states, prev_acc=None, size_acc=None, keys=(None, None)):
        self.init_count   = np.zeros(num_vals1, dtype=np.float64)
        self.weight_count = np.zeros((num_vals1, num_states), dtype=np.float64)
        self.state_count  = np.zeros((num_states, num_vals2), dtype=np.float64)
        self.size_accumulator = size_acc
        self.prev_accumulator = prev_acc
        self.num_vals1 = num_vals1
        self.num_vals2 = num_vals2
        self.num_states = num_states
        self.weight_key = keys[0]
        self.state_key = keys[1]

    def update(self, x, weight, estimate):

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)
        nx = np.sum(cx)

        X = (estimate.cond_weights[vx, :].T * (cx/nx)).T
        Y = estimate.state_prob_mat[:, vy]
        Z = X[:, :, None] * Y[None, :, :]

        # [old word] x [state] x [new word]

        ss = np.sum(np.sum(Z, axis=0, keepdims=True), axis=1, keepdims=True)
        Z /= ss

        self.weight_count[vx, :] += np.dot(Z, cy) * weight
        self.state_count[:,vy] += np.sum(Z, axis=0) * cy * weight
        self.init_count[vx] += cx

        if self.prev_accumulator is not None:
            self.prev_accumulator.update(x[0], weight, None if estimate is None else estimate.prev_dist)

        if self.size_accumulator is not None:
            self.size_accumulator.update(cy.sum(), weight, None if estimate is None else estimate.len_dist)

    def initialize(self, x, weight, rng):

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        self.weight_count[vx, :] += rng.dirichlet(np.ones(self.num_states),size=len(vx)) * weight
        self.state_count[:, vy] += rng.dirichlet(np.ones(self.num_states),size=len(vy)).T * cy * weight
        self.init_count[vx] += cx

        if self.prev_accumulator is not None:
            self.prev_accumulator.initialize(x[0], weight, rng)

        if self.size_accumulator is not None:
            self.size_accumulator.initialize(cy.sum(), weight, rng)

    def seq_update(self, x, weights, estimate):

        if x[1] is None:
            xx = x[0]

            for i, (entry, weight) in enumerate(zip(xx[0], weights)):

                vx, cx, vy, cy = entry
                nx = np.sum(cx)
                X = (estimate.cond_weights[vx, :].T * (cx / nx)).T
                Y = estimate.state_prob_mat[:, vy]
                Z = X[:, :, None] * Y[None, :, :]

                # [old word] x [state] x [new word]

                ss = np.sum(np.sum(Z, axis=0, keepdims=True), axis=1, keepdims=True)
                Z /= ss

                self.weight_count[vx, :] += np.dot(Z, cy) * weight
                self.state_count[:, vy] += np.sum(Z, axis=0) * cy * weight
                self.init_count[vx] += cx

            if self.prev_accumulator is not None:
                self.prev_accumulator.seq_update(xx[1], weights, None if estimate is None else estimate.prev_dist)
            if self.size_accumulator is not None:
                self.size_accumulator.seq_update(xx[2], weights, None if estimate is None else estimate.len_dist)
        else:

            (s0, s1, x0, x1, c0, c1, w0), xv, nn = x[1]

            t0 = np.concatenate([[0], s0]).cumsum().astype(np.int32)
            t1 = np.concatenate([[0], s1]).cumsum().astype(np.int32)
            max_len = s0.max()

            numba_seq_update(self.num_states, max_len, t0, t1, x0, x1, c0, c1, w0, estimate.cond_weights, estimate.state_prob_mat, self.weight_count, self.state_count, self.init_count, weights)

            if self.prev_accumulator is not None:
                self.prev_accumulator.seq_update(xv, weights, None if estimate is None else estimate.prev_dist)
            if self.size_accumulator is not None:
                self.size_accumulator.seq_update(nn, weights, None if estimate is None else estimate.len_dist)

    def combine(self, suff_stat):

        init_count, weight_count, state_count, prev_acc, size_acc = suff_stat

        if self.prev_accumulator is not None:
            self.prev_accumulator.combine(prev_acc)

        if self.size_accumulator is not None:
            self.size_accumulator.combine(size_acc)

        self.init_count += init_count
        self.weight_count += weight_count
        self.state_count += state_count

        return self

    def value(self):

        pval = None if self.prev_accumulator is None else self.prev_accumulator.value()
        sval = None if self.size_accumulator is None else self.size_accumulator.value()

        return self.init_count, self.weight_count, self.state_count, pval, sval

    def from_value(self, x):

        init_count, weight_count, state_count, prev_acc, size_acc = x

        self.init_count = init_count
        self.weight_count = weight_count
        self.state_count = state_count

        if self.prev_accumulator is not None:
            self.prev_accumulator.from_value(prev_acc)

        if self.size_accumulator is not None:
            self.size_accumulator.from_value(size_acc)

        return self

    def key_merge(self, stats_dict):

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.weight_count
            else:
                stats_dict[self.weight_key] = self.weight_count.copy()

        if self.state_key is not None:
            if self.state_key in stats_dict:
                stats_dict[self.state_key] += self.state_count
            else:
                stats_dict[self.state_key] = self.state_count.copy()

        if self.prev_accumulator is not None:
            self.prev_accumulator.key_merge(stats_dict)

        if self.size_accumulator is not None:
            self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict):

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.weight_count = stats_dict[self.weight_key].copy()

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.state_count = stats_dict[self.state_key].copy()

        if self.prev_accumulator is not None:
            self.prev_accumulator.key_replace(stats_dict)

        if self.size_accumulator is not None:
            self.size_accumulator.key_replace(stats_dict)


class IntegerHiddenAssociationAccumulatorFactory(object):

    def __init__(self, num_vals1, num_vals2, num_states, prev_factory, len_factory, keys):
        self.len_factory = len_factory
        self.prev_factory = prev_factory
        self.keys = keys
        self.num_vals1 = num_vals1
        self.num_vals2 = num_vals2
        self.num_states = num_states

    def make(self):
        len_acc = None if self.len_factory is None else self.len_factory.make()
        prev_acc = None if self.prev_factory is None else self.prev_factory.make()

        return IntegerHiddenAssociationAccumulator(self.num_vals1, self.num_vals2,  self.num_states, prev_acc, len_acc, keys=self.keys)

class IntegerHiddenAssociationEstimator(ParameterEstimator):

    def __init__(self, num_vals, num_states, alpha=0.0, given_estimator=None, len_estimator=None, suff_stat=None, pseudo_count=None, name=None, keys=(None, None)):

        self.prev_estimator = given_estimator
        self.len_estimator = len_estimator
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.num_vals = num_vals
        self.num_states = num_states
        self.alpha = alpha
        self.name = name
        self.keys = keys

        if isinstance(num_vals, (tuple,list)):
            if len(num_vals) >= 2:
                self.num_vals1 = num_vals[0]
                self.num_vals2 = num_vals[1]
            elif len(num_vals) == 1:
                self.num_vals1 = num_vals[0]
                self.num_vals2 = num_vals[0]
        else:
            self.num_vals1 = num_vals
            self.num_vals2 = num_vals

    def accumulatorFactory(self):

        len_factory = None if self.len_estimator is None else self.len_estimator.accumulatorFactory()
        prev_factory = None if self.prev_estimator is None else self.prev_estimator.accumulatorFactory()

        return IntegerHiddenAssociationAccumulatorFactory(self.num_vals1, self.num_vals2, self.num_states, prev_factory, len_factory, self.keys)

    def estimate(self, nobs, suff_stat):

        init_count, weight_count, state_count, prev_stats, size_stats = suff_stat

        if self.len_estimator is not None:
            len_dist = self.len_estimator.estimate(nobs, size_stats)
        else:
            len_dist = None

        if self.prev_estimator is not None:
            prev_dist = self.prev_estimator.estimate(nobs, prev_stats)
        else:
            prev_dist = None

        if self.pseudo_count is not None:
            init_count  += self.pseudo_count/len(init_count)
            state_count += self.pseudo_count/(self.num_states * self.num_vals2)
            weight_count += self.pseudo_count/(self.num_states * self.num_vals1)

        #init_prob = init_count / np.sum(init_count)

        wsum = np.sum(weight_count, axis=1, keepdims=True)
        ssum = np.sum(state_count, axis=1, keepdims=True)
        ssum[ssum == 0] = 1.0
        wsum[wsum == 0] = 1.0

        weight_prob = weight_count / wsum
        state_prob = state_count / ssum

        #return IntegerHiddenAssociationDistribution(init_prob, state_prob, weight_prob, self.alpha, len_dist)
        return IntegerHiddenAssociationDistribution(state_prob, weight_prob, alpha=self.alpha, given_dist=prev_dist, len_dist=len_dist, name=self.name, keys=self.keys)


@numba.njit('void(int64, int64, int32[:], int32[:], int32[:], int32[:], float64[:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:], float64, float64, float64[:])')
def numba_seq_log_density(num_states, max_len1, t0, t1, x0, x1, c0, c1, w0, cond_weights, state_prob_mat, init_prob_vec, a, b, out):

    X = np.zeros((max_len1, num_states), dtype=np.float64)

    for i in range(len(t0)-1):

        vx = x0[t0[i]:t0[i + 1]]
        cx = c0[t0[i]:t0[i + 1]]
        vy = x1[t1[i]:t1[i + 1]]
        cy = c1[t1[i]:t1[i + 1]]
        sx = w0[i]

        l1 = t0[i+1]-t0[i]
        l2 = t1[i+1]-t1[i]

        for j in range(l1):
            temp = cx[j] / sx
            #out[i] += math.log(init_prob_vec[vx[j]])*cx[j]
            for k in range(num_states):
                X[j, k] = cond_weights[vx[j], k] * temp

        for w in range(l2):
            wid = vy[w]
            temp_sum = 0
            for j in range(l1):
                for k in range(num_states):
                    temp_sum += X[j,k] * state_prob_mat[k, wid]
            out[i] += math.log(temp_sum) * cy[w]



@numba.njit('void(int64, int64, int32[:], int32[:], int32[:], int32[:], float64[:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:])')
def numba_seq_update(num_states, max_len1, t0, t1, x0, x1, c0, c1, w0, cond_weights, state_prob_mat, weight_count, state_count, init_count, weights):

    X = np.zeros((max_len1, num_states), dtype=np.float64)
    Z = np.zeros((max_len1, num_states), dtype=np.float64)

    for i in range(len(t0)-1):

        weight = weights[i]
        vx = x0[t0[i]:t0[i + 1]]
        cx = c0[t0[i]:t0[i + 1]]
        vy = x1[t1[i]:t1[i + 1]]
        cy = c1[t1[i]:t1[i + 1]]

        l1 = t0[i+1]-t0[i]
        l2 = t1[i+1]-t1[i]

        nx = w0[i]

        for j in range(l1):
            temp = cx[j]/nx
            #init_count[vx[j]] += cx[j] * weight
            for k in range(num_states):
                X[j,k] = cond_weights[vx[j],k] * temp

        for w in range(l2):
            wid = vy[w]
            temp_sum = 0
            for j in range(l1):
                for k in range(num_states):
                    temp = X[j, k] * state_prob_mat[k,wid]
                    Z[j, k] = temp
                    temp_sum += temp

            temp_weight = cy[w] * weight / temp_sum
            for j in range(l1):
                for k in range(num_states):
                    temp = temp_weight * Z[j, k]
                    weight_count[vx[j], k] += temp
                    state_count[k, wid] += temp
