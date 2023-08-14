from typing import List, Optional, Sequence, Tuple
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
from pysp.utils.optsutil import countByValue
import numba
from pysp.arithmetic import maxrandint


class IntegerPLSIDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, state_word_mat, doc_state_mat, doc_vec, len_dist=None, name=None):

        self.prob_mat    = np.asarray(state_word_mat, dtype=np.float64)
        self.state_mat   = np.asarray(doc_state_mat, dtype=np.float64)
        self.doc_vec     = np.asarray(doc_vec, dtype=np.float64)
        self.log_doc_vec = np.log(self.doc_vec)
        self.num_vals    = self.prob_mat.shape[0]
        self.num_states  = self.prob_mat.shape[1]
        self.num_docs    = self.state_mat.shape[0]
        self.name        = name
        self.len_dist    = len_dist

    def __str__(self):
        s1 = ','.join(['[' + ','.join(map(str, self.prob_mat[i, :])) + ']' for i in range(len(self.prob_mat))])
        s2 = ','.join(['[' + ','.join(map(str, self.state_mat[i, :])) + ']' for i in range(len(self.state_mat))])
        s3 = ','.join(map(str, self.doc_vec))
        s4 = str(self.name)
        s5 = str(self.len_dist)
        return 'IntegerPLSIDistribution([%s], [%s], [%s], name=%s, len_dist=%s)'%(s1, s2, s3, s4, s5)

    def density(self, x: Tuple[int, Sequence[Tuple[int,float]]]) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: Tuple[int, Sequence[Tuple[int,float]]]) -> float:

        id = x[0]
        xv = np.asarray([u[0] for u in x[1]], dtype=int)
        xc = np.asarray([u[1] for u in x[1]], dtype=float)

        rv = 0.0
        rv += np.dot(np.log(np.dot(self.prob_mat[xv,:], self.state_mat[id,:])), xc)
        rv += np.log(self.doc_vec[id])

        if self.len_dist is not None:
            rv += self.len_dist.log_density(np.sum(xc))

        return rv

    def component_log_density(self, x: Tuple[int, Sequence[Tuple[int,float]]]) -> np.ndarray:

        xv = np.asarray([u[0] for u in x[1]], dtype=int)
        xc = np.asarray([u[1] for u in x[1]], dtype=float)

        return np.dot(np.log(self.prob_mat[xv, :]).T, xc)

    def seq_log_density(self, x) -> np.ndarray:

        nn, (xv, xc, xd, xi, xn, xm) = x
        cnt = len(xn)

        w = np.zeros(len(xv), dtype=np.float64)
        index_dot(self.prob_mat, xv, self.state_mat, xd, w)
        #w = np.sum(self.prob_mat[xv,:] * self.state_mat[xd,:], axis=1)
        w = np.log(w, out=w)
        w *= xc

        rv = np.zeros(cnt, dtype=np.float64)
        bincount(xi, w, rv)
        #rv = np.bincount(xi, weights=w, minlength=cnt)
        rv += self.log_doc_vec[xm]

        if self.len_dist is not None:
            rv += self.len_dist.seq_log_density(nn)

        return rv


    def seq_component_log_density(self, x) -> np.ndarray:

        nn, (xv, xc, xd, xi, xn, xm) = x
        rv = np.zeros((xi[-1]+1, self.num_states), dtype=np.float64)
        wmat = self.prob_mat
        fast_seq_component_log_density(xv, xc, xd, xi, xm, wmat, rv)
        return rv

    def seq_encode(self, x: Sequence[Tuple[int, Sequence[Tuple[int,float]]]]) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):

        xv = []
        xc = []
        xd = []
        xi = []
        xn = []
        xm = []

        for i, (id,xx) in enumerate(x):

            v = [u[0] for u in xx]
            c = [u[1] for u in xx]

            xv.extend(v)
            xc.extend(c)
            xd.extend([id]*len(v))
            xi.extend([i]*len(v))
            xn.append(np.sum(c))
            xm.append(id)

        xv = np.asarray(xv, dtype=np.int32)
        xc = np.asarray(xc, dtype=np.float64)
        xd = np.asarray(xd, dtype=np.int32)
        xi = np.asarray(xi, dtype=np.int32)
        xn = np.asarray(xn, dtype=np.float64)
        xm = np.asarray(xm, dtype=np.int32)

        if self.len_dist is not None:
            nn = self.len_dist.seq_encode(xn)
        else:
            nn = None

        return nn, (xv, xc, xd, xi, xn, xm)

    def sampler(self, seed: Optional[int] = None):
        return IntegerPLSISampler(self, seed)




class IntegerPLSISampler(object):

    def __init__(self, dist: IntegerPLSIDistribution, seed: Optional[int] = None):
        self.rng  = np.random.RandomState(seed)
        self.dist = dist
        self.size_rng = self.dist.len_dist.sampler(self.rng.randint(0, maxrandint))

    def sample(self, size=None):

        if size is None:

            id    = self.rng.choice(self.dist.num_docs, p=self.dist.doc_vec)
            cnt   = self.size_rng.sample()
            z = self.rng.multinomial(cnt, pvals=self.dist.state_mat[id,:])
            rv = []
            for i,n in enumerate(z):
                if n > 0:
                    rv.extend(self.rng.choice(self.dist.num_vals, p=self.dist.prob_mat[:,i], replace=True, size=n))

            return id, list(countByValue(rv).items())

        else:
            return [self.sample() for i in range(size)]


class IntegerPLSIAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals, num_states, num_docs, len_acc=None, name=None, keys=(None,None,None)):

        self.num_vals   = num_vals
        self.num_states = num_states
        self.num_docs   = num_docs
        self.word_count = np.zeros((num_states, num_vals), dtype=np.float64)
        self.comp_count = np.zeros((num_docs, num_states), dtype=np.float64)
        self.doc_count  = np.zeros(num_docs, dtype=np.float64)
        self.name       = name
        self.wc_key     = keys[0]
        self.sc_key     = keys[1]
        self.dc_key     = keys[2]
        self.len_acc    = len_acc

    def update(self, x, weight, estimate):

        id = x[0]
        xv = np.asarray([u[0] for u in x[1]])
        xc = np.asarray([u[1] for u in x[1]])

        update = (estimate.prob_mat[xv,:] * estimate.state_mat[id,:]).T
        update *= xc*weight/np.sum(update, axis=0)
        self.comp_count[id,:] += np.sum(update, axis=1)
        self.word_count[:,xv] += update
        self.doc_count[id] += weight

        if self.len_acc is not None:
            self.len_acc.update(np.sum(xc), weight, estimate.len_dist)

    def initialize(self, x, weight, rng):

        id = x[0]
        xv = np.asarray([u[0] for u in x[1]])
        xc = np.asarray([u[1] for u in x[1]])

        update = rng.dirichlet(np.ones(self.num_states)/self.num_states, size=len(xc)).T
        update *= xc*weight
        self.word_count[:,xv] += update
        self.comp_count[id,:] += np.sum(update, axis=1)
        self.doc_count[id] += weight

        if self.len_acc is not None:
            self.len_acc.update(np.sum(xc), weight, rng)

    def seq_update(self, x, weights, estimate):

        nn, (xv, xc, xd, xi, xn, xm) = x

        fast_seq_update(xv, xc, xd, xi, xm, weights, estimate.prob_mat, estimate.state_mat, self.word_count, self.comp_count, self.doc_count)

        '''

        temp = xc*weights[xi]
        update  = estimate.prob_mat[xv, :] * estimate.state_mat[xd, :]

        temp /= np.sum(update, axis=1)
        update *= temp[:,None]

        #vec_bincount1(xv, update, self.word_count.T)
        #vec_bincount1(xd, update, self.comp_count)
        #bincount(xm, weights, self.num_docs)

        for i in range(self.num_states):
            self.word_count[i,:] += np.bincount(xv, weights=update[:,i], minlength=self.num_vals)
            self.comp_count[:,i] += np.bincount(xd, weights=update[:,i], minlength=self.num_docs)
        self.doc_count += np.bincount(xm, weights=weights, minlength=self.num_docs)
        '''

        if self.len_acc is not None:
            self.len_acc.seq_update(nn, weights, estimate.len_dist)

    def combine(self, suff_stat):
        self.word_count += suff_stat[0]
        self.comp_count += suff_stat[1]
        self.doc_count  += suff_stat[2]
        if self.len_acc is not None:
            self.len_acc.combine(suff_stat[3])
        return self

    def value(self):
        if self.len_acc is not None:
            return self.word_count, self.comp_count, self.doc_count, self.len_acc.value()
        else:
            return self.word_count, self.comp_count, self.doc_count, None

    def from_value(self, x):
        self.word_count = x[0]
        self.comp_count = x[1]
        self.doc_count  = x[2]
        if self.len_acc is not None:
            self.len_acc.from_value(x[3])

    def key_merge(self, stats_dict):
        if self.wc_key is not None:
            if self.wc_key in stats_dict:
                stats_dict[self.wc_key] += self.word_count
            else:
                stats_dict[self.wc_key] = self.word_count

        if self.sc_key is not None:
            if self.sc_key in stats_dict:
                stats_dict[self.sc_key] += self.comp_count
            else:
                stats_dict[self.sc_key] = self.comp_count

        if self.dc_key is not None:
            if self.dc_key in stats_dict:
                stats_dict[self.dc_key] += self.doc_count
            else:
                stats_dict[self.dc_key] = self.doc_count

        if self.len_acc is not None:
            self.len_acc.key_merge(stats_dict)

    def key_replace(self, stats_dict):

        if self.wc_key is not None:
            if self.wc_key in stats_dict:
                self.word_count = stats_dict[self.wc_key]
        if self.sc_key is not None:
            if self.sc_key in stats_dict:
                self.comp_count = stats_dict[self.sc_key]
        if self.dc_key is not None:
            if self.dc_key in stats_dict:
                self.doc_count = stats_dict[self.dc_key]

        if self.len_acc is not None:
            self.len_acc.key_replace(stats_dict)

class IntegerPLSIAccumulatorFactory(object):

    def __init__(self, num_vals, num_states, num_docs, len_factory, keys):
        self.len_factory = len_factory
        self.keys = keys
        self.num_vals = num_vals
        self.num_states = num_states
        self.num_docs = num_docs

    def make(self):
        if self.len_factory is None:
            return IntegerPLSIAccumulator(self.num_vals, self.num_states, self.num_docs, len_acc=None, keys=self.keys)
        else:
            return IntegerPLSIAccumulator(self.num_vals, self.num_states, self.num_docs, len_acc=self.len_factory.make(), keys=self.keys)



class IntegerPLSIEstimator(ParameterEstimator):

    def __init__(self, num_vals, num_states, num_docs, len_estimator=None, pseudo_count=(None,None,None), suff_stat=None, name=None, keys=(None,None,None)):

        self.suff_stat     = suff_stat
        self.pseudo_count  = pseudo_count
        self.num_vals      = num_vals
        self.num_states    = num_states
        self.num_docs      = num_docs
        self.len_estimator = len_estimator
        self.keys          = keys
        self.name          = name

    def accumulatorFactory(self):
        len_est = self.len_estimator.accumulatorFactory() if self.len_estimator is not None else None
        return IntegerPLSIAccumulatorFactory(self.num_vals, self.num_states, self.num_docs, len_est, self.keys)

    def estimate(self, nobs, suff_stat):

        word_count, comp_count, doc_count, len_suff_stats = suff_stat

        if self.pseudo_count[0] is not None:
            adj_cnt = self.pseudo_count[0] / np.prod(word_count.shape)
            word_prob_mat = word_count.T + adj_cnt
            word_prob_mat /= np.sum(word_prob_mat, axis=0, keepdims=True)
        else:
            word_prob_mat = word_count.T / np.sum(word_count, axis=1)

        if self.pseudo_count[1] is not None:
            adj_cnt = self.pseudo_count[1] / comp_count.shape[1]
            state_prob_mat = comp_count + adj_cnt
            state_prob_mat /= np.sum(state_prob_mat, axis=1, keepdims=True)
        else:
            state_prob_mat = comp_count / np.sum(comp_count, axis=1, keepdims=True)

        if self.pseudo_count[2] is not None:
            adj_cnt = self.pseudo_count[1] / len(doc_count)
            doc_prob_vec = doc_count + adj_cnt
            doc_prob_vec /= np.sum(doc_prob_vec)
        else:
            doc_prob_vec = doc_count / np.sum(doc_count)

        if self.len_estimator is not None:
            len_dist = self.len_estimator.estimate(None, len_suff_stats)
        else:
            len_dist = None

        return IntegerPLSIDistribution(word_prob_mat, state_prob_mat, doc_prob_vec, name=self.name, len_dist=len_dist)




@numba.njit('void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:,:], float64[:,:], float64[:], float64[:])', fastmath=True)
def fast_seq_log_density(xv, xc, xd, xi, xm, wmat, smat, dvec, out):
    n = len(xv)
    m = len(xm)
    k = smat.shape[1]
    for i in range(n):
        ll = 0.0
        cc = xc[i]
        i1 = xv[i]
        i2 = xd[i]
        i3 = xi[i]
        for j in range(k):
            ll += wmat[i1,j]*smat[i2,j]
        out[i3] += cc*np.log(ll)
    for i in range(m):
        out[i] += dvec[xm[i]]

@numba.njit('void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:,:], float64[:,:])', fastmath=True)
def fast_seq_component_log_density(xv, xc, xd, xi, xm, wmat, out):
    n = len(xv)
    k = wmat.shape[1]
    for i in range(n):
        cc = xc[i]
        i1 = xv[i]
        i3 = xi[i]
        for j in range(k):
            out[i3, j] += np.log(wmat[i1,j])*cc

@numba.njit('void(int32[:], float64[:], int32[:], int32[:], int32[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:])', fastmath=True)
def fast_seq_update(xv, xc, xd, xi, xm, weights, wmat, smat, wcnt, scnt, dcnt):
    n = len(xv)
    m = len(xm)
    k = smat.shape[1]
    posterior = np.zeros(k, dtype=np.float64)
    for i in range(n):
        norm_const = 0.0
        cc = xc[i]
        i1 = xv[i]
        i2 = xd[i]
        ww = weights[xi[i]]
        for j in range(k):
            temp = wmat[i1,j]*smat[i2,j]
            posterior[j] = temp
            norm_const  += temp
        norm_const = ww*cc/norm_const
        for j in range(k):
            temp = posterior[j]*norm_const
            wcnt[j,i1] += temp
            scnt[i2,j] += temp
    for i in range(m):
        dcnt[xm[i]] += weights[i]


@numba.njit('float64[:](float64[:,:], int32[:], float64[:,:], int32[:], float64[:])')
def index_dot(x, xi, y, yi, out):
    n = x.shape[1]
    for i in range(len(xi)):
        i1 = xi[i]
        i2 = yi[i]
        for j in range(n):
            out[i] += x[i1,j]*y[i2,j]
    return out

@numba.njit('float64[:](int32[:], float64[:], float64[:])')
def bincount(x, w, out):
    for i in range(len(x)):
        out[x[i]] += w[i]
    return out

@numba.njit('float64[:,:](int32[:], float64[:,:], float64[:,:])')
def vec_bincount1(x, w, out):
    n = w.shape[1]
    for i in range(len(x)):
        for j in range(n):
            out[x[i],j] += w[i,j]
    return out

@numba.njit('float64[:,:](int32[:], float64[:,:], int32[:], float64[:,:])')
def vec_bincount2(x, w, y, out):
    for i in range(len(x)):
        out[x[i],:] += w[y[i],:]
    return out

