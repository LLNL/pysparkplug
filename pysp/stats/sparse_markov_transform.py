from typing import Optional, List, Tuple, Union, Sequence
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from pysp.utils.optsutil import countByValue
import itertools
from pysp.arithmetic import maxrandint


class SparseMarkovAssociationDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, init_prob_vec: Union[Sequence[float], np.ndarray], cond_prob_mat: csr_matrix, alpha: float = 0.0, len_dist=None, low_memory=False):

        self.init_prob_vec = np.asarray(init_prob_vec, dtype=np.float)
        self.cond_prob_mat = csr_matrix(cond_prob_mat, dtype=np.float)
        self.len_dist = len_dist
        self.num_vals = len(init_prob_vec)
        self.alpha = alpha
        self.low_memory = low_memory

    def __str__(self):
        s1 = ','.join(map(str, self.init_prob_vec))
        temp = self.cond_prob_mat.nonzero()
        tt = np.asarray(self.cond_prob_mat[temp[0], temp[1]]).flatten()
        s20 = ','.join(map(str, tt))
        s21 = ','.join(map(str, temp[0]))
        s22 = ','.join(map(str, temp[1]))
        s2 = '([%s], ([%s],[%s]))' % (s20, s21, s22)
        s3 = str(self.alpha)
        s4 = str(self.len_dist)
        return 'SparseMarkovAssociationDistribution([%s], %s, alpha=%s, len_dist=%s)' % (s1, s2, s3, s4)

    def density(self, x: Tuple[List[Tuple[int,float]], List[Tuple[int,float]]]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Tuple[List[Tuple[int,float]], List[Tuple[int,float]]]) -> float:

        nw = self.num_vals
        a  = self.alpha / nw
        b  = 1 - self.alpha

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        nx = np.sum(cx)
        ny = np.sum(cy)

        temp = self.cond_prob_mat[vx[:, None], vy].toarray()
        ll2 = np.dot(np.log(np.dot((temp * b + a).T, cx / nx)), cy)
        ll1 = np.dot(np.log(self.init_prob_vec[vx] * b + a), cx)
        rv  = ll2# + ll2

        if self.len_dist is not None:
            rv += self.len_dist.log_density([nx, ny])

        return rv

    def seq_log_density(self, x):

        nw = self.num_vals
        a = self.alpha / nw
        b = 1 - self.alpha

        xlen = len(x[0])

        if x[3] is not None:

            obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec = x[3]

            vv = x[2]

            p = np.asarray(self.cond_prob_mat[vv[:,0], vv[:,1]]).flatten()
            p = (p*b + a)
            sval = np.bincount(seqidx, weights=p[pairidx]*cxvec)
            np.log(sval, out=sval)
            sval *= fcyvec
            rv = np.bincount(fsqyvec, weights=sval, minlength=xlen)
            #rv += np.bincount(fsqxvec, weights=np.log(self.init_prob_vec[fvxvec]*b + a), minlength=xlen)

        else:

            rv = np.zeros(len(x[0]), dtype=np.float)

            for i, entry in enumerate(x[0]):
                xx, cx, yy, cy = entry
                nx = np.sum(cx)

                temp = self.cond_prob_mat[xx[:,None], yy].toarray()
                ll2 = np.dot(np.log(np.dot((temp*b + a).T, cx/nx)),cy)
                ll1 = np.dot(np.log(self.init_prob_vec[xx]*b + a), cx)

                rv[i] = ll2# + ll2

        if self.len_dist is not None:
            lln = self.len_dist.seq_log_density(x[1])
            rv += lln

        return rv

    def seq_encode(self, x: Sequence[Tuple[List[Tuple[int,float]], List[Tuple[int,float]]]]):

        if self.low_memory:

            rv = []
            nn = []
            vset = set()

            for k, xx in enumerate(x):

                vx = np.asarray([u[0] for u in xx[0]], dtype=int)
                cx = np.asarray([u[1] for u in xx[0]], dtype=float)
                vy = np.asarray([u[0] for u in xx[1]], dtype=int)
                cy = np.asarray([u[1] for u in xx[1]], dtype=float)
                nx = np.sum(cx)

                vset.update(itertools.product(vx, vy))
                rv.append((vx, cx, vy, cy))
                nn.append((cx.sum(), cy.sum()))

            if self.len_dist is not None:
                nn = self.len_dist.seq_encode(nn)
            else:
                nn = None

            vv = np.zeros((len(vset), 2), dtype=int)
            for i, vvv in enumerate(vset):
                vv[i, :] = vvv[:]

            qq = None

        else:


            rv = []
            nn = []
            vmap = dict()

            obsidx  = []
            pairidx = []
            seqidx  = []
            cxvec   = []
            cyvec   = []

            fcyvec  = []
            fcxvec  = []
            fvxvec  = []
            fsqxvec  = []
            fsqyvec  = []

            ridx = -1
            for k, xx in enumerate(x):

                vx = np.asarray([u[0] for u in xx[0]], dtype=int)
                cx = np.asarray([u[1] for u in xx[0]], dtype=float)
                vy = np.asarray([u[0] for u in xx[1]], dtype=int)
                cy = np.asarray([u[1] for u in xx[1]], dtype=float)
                nx = np.sum(cx)

                fcyvec.extend(cy)
                fcxvec.extend(cx)
                fvxvec.extend(vx)
                fsqxvec.extend([k]*len(vx))
                fsqyvec.extend([k]*len(vy))

                for i, vvy in enumerate(vy):
                    ridx += 1
                    for j, vvx in enumerate(vx):

                        if (vvx, vvy) not in vmap:
                            vmap[(vvx, vvy)] = len(vmap)
                        widx = vmap[(vvx, vvy)]
                        obsidx.append(k)
                        seqidx.append(ridx)
                        pairidx.append(widx)
                        cxvec.append(cx[j]/nx)
                        cyvec.append(cy[i])

                rv.append((vx, cx, vy, cy))
                nn.append((cx.sum(), cy.sum()))

            if self.len_dist is not None:
                nn = self.len_dist.seq_encode(nn)
            else:
                nn = None

            vv = np.zeros((len(vmap), 2), dtype=int)
            for vvv,i in vmap.items():
                vv[i, :] = vvv[:]

            obsidx = np.asarray(obsidx, dtype=int)
            seqidx = np.asarray(seqidx, dtype=int)
            cxvec = np.asarray(cxvec, dtype=float)
            cyvec = np.asarray(cyvec, dtype=float)
            pairidx = np.asarray(pairidx, dtype=int)

            fcxvec = np.asarray(fcxvec, dtype=float)
            fcyvec = np.asarray(fcyvec, dtype=float)
            fvxvec = np.asarray(fvxvec, dtype=int)
            fsqxvec = np.asarray(fsqxvec, dtype=int)
            fsqyvec = np.asarray(fsqyvec, dtype=int)

            qq = (obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec)

        return rv, nn, vv, qq

    def sampler(self, seed=None):
        return SparseMarkovAssociationSampler(self, seed)


class SparseMarkovAssociationSampler(object):

    def __init__(self, dist: SparseMarkovAssociationDistribution, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.size_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None):

        if size is None:

            slens = self.size_sampler.sample()
            rng = np.random.RandomState(self.rng.randint(0, maxrandint))

            v1 = list(rng.choice(len(self.dist.init_prob_vec), p=self.dist.init_prob_vec, replace=True, size=slens[0]))
            v2 = []

            z1 = list(rng.choice(len(v1), replace=True, size=slens[1]))
            nw = self.dist.num_vals

            for zz1 in z1:

                if rng.rand() > self.dist.alpha:
                    p = self.dist.cond_prob_mat[v1[zz1], :].toarray().flatten()
                    v2.append(rng.choice(nw, p=p))
                else:
                    v2.append(rng.choice(nw))

            return list(countByValue(v1).items()), list(countByValue(v2).items())

        else:
            return [self.sample() for i in range(size)]


class SparseMarkovAssociationAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals, size_acc=None, keys=(None, None)):
        self.init_count = np.zeros(num_vals)
        self.trans_count = None
        self.size_accumulator = size_acc
        self.num_vals = num_vals
        self.init_key = keys[0]
        self.trans_key = keys[1]

    def update(self, x, weight, estimate):

        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = lil_matrix((num_vals, num_vals))

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        temp = estimate.cond_prob_mat[vx[:,None], vy].toarray()

        loc_cprob = temp * cx[:,None]
        w = loc_cprob.sum(axis=0)
        loc_cprob *= (cy / w) * weight

        self.trans_count[vx[:,None], vy] += loc_cprob
        self.init_count[vx] += cx

        if self.size_accumulator is not None:
            self.size_accumulator.update((cx.sum(), cy.sum()), weight, estimate.len_dist)

    def initialize(self, x, weight, rng):

        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = lil_matrix((num_vals, num_vals))

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        self.trans_count[vx[:,None], vy] += np.outer(cx/np.sum(cx), cy)*weight
        self.init_count[vx] += cx*weight

        if self.size_accumulator is not None:
            self.size_accumulator.initialize((cx.sum(), cy.sum()), weight, rng)

    def seq_update(self, x, weights, estimate):

        if self.trans_count is None:
            num_vals = self.num_vals
            self.trans_count = csr_matrix((num_vals, num_vals))

        nw = self.num_vals

        if x[3] is not None:

            obsidx, seqidx, pairidx, cxvec, cyvec, fsqxvec, fvxvec, fcxvec, fsqyvec, fcyvec = x[3]

            vv = x[2]

            p = np.asarray(estimate.cond_prob_mat[vv[:,0], vv[:,1]]).flatten()
            pp = p[pairidx]*cxvec
            sval = np.bincount(seqidx, weights=pp)
            np.divide(weights[fsqyvec], sval, out=sval)
            sval *= fcyvec
            pp   *= sval[seqidx]
            pp = np.bincount(pairidx, weights=pp)

            umat = csr_matrix((pp, (vv[:,0], vv[:,1])), shape=(nw, nw))
            self.trans_count += umat
            self.init_count += np.bincount(fvxvec, weights=fcxvec, minlength=nw)

        else:

            nzv = x[2]
            umat = csr_matrix((np.zeros(nzv.shape[0]), (nzv[:, 0], nzv[:, 1])), shape=(nw, nw))

            for i, (entry, weight) in enumerate(zip(x[0], weights)):

                vx,cx,vy,cy = entry

                temp = estimate.cond_prob_mat[vx[:, None], vy].toarray()

                loc_cprob = temp * cx[:, None]
                w = loc_cprob.sum(axis=0)
                loc_cprob *= (cy / w) * weight

                umat[vx[:, None], vy] += loc_cprob
                self.init_count[vx] += cx

            self.trans_count += umat

            if self.size_accumulator is not None:
                self.size_accumulator.seq_update(x[1], weights, estimate.len_dist)



    def combine(self, suff_stat):

        init_count, trans_count, size_acc = suff_stat

        if self.size_accumulator is not None:
            self.size_accumulator.combine(size_acc)

        self.init_count += init_count
        self.trans_count += trans_count

        return self

    def value(self):
        if self.size_accumulator is not None:
            return self.init_count, self.trans_count, self.size_accumulator.value()
        else:
            return self.init_count, self.trans_count, None

    def from_value(self, x):

        init_count, trans_count, size_acc = x

        self.init_count = init_count
        self.trans_count = trans_count
        if self.size_accumulator is not None:
            self.size_accumulator.from_value(size_acc)

        return self

    def key_merge(self, stats_dict):

        if self.init_key is not None:
            if self.init_key in stats_dict:
                stats_dict[self.init_key] += self.init_count
            else:
                stats_dict[self.init_key] = self.init_count

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                stats_dict[self.trans_key] += self.trans_count
            else:
                stats_dict[self.trans_key] = self.trans_count

        if self.size_accumulator is not None:
            self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict):

        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_count = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_count = stats_dict[self.trans_key]

        if self.size_accumulator is not None:
            self.size_accumulator.key_replace(stats_dict)


class SparseMarkovAssociationAccumulatorFactory(object):

    def __init__(self, num_vals, len_factory, keys):
        self.len_factory = len_factory
        self.keys = keys
        self.num_vals = num_vals

    def make(self):
        if self.len_factory is None:
            return SparseMarkovAssociationAccumulator(self.num_vals, size_acc=None, keys=self.keys)
        else:
            return SparseMarkovAssociationAccumulator(self.num_vals, size_acc=self.len_factory.make(), keys=self.keys)


class SparseMarkovAssociationEstimator(ParameterEstimator):

    def __init__(self, num_vals: int, alpha: float = 0.0, len_estimator=None, suff_stat=None, pseudo_count=None, keys=(None, None)):
        self.keys = keys
        self.len_estimator = len_estimator
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.num_vals = num_vals
        self.alpha = alpha

    def accumulatorFactory(self):
        return SparseMarkovAssociationAccumulatorFactory(self.num_vals, self.len_estimator, self.keys)

    def estimate(self, nobs, suff_stat):

        init_count, trans_count, size_stats = suff_stat

        if self.len_estimator is not None:
            len_dist = self.len_estimator.estimate(nobs, size_stats)
        else:
            len_dist = None
        trans_count = trans_count.tocsr()
        row_sum = trans_count.sum(axis=1)
        row_sum = csr_matrix(row_sum)
        row_sum.eliminate_zeros()
        row_sum.data = 1.0/row_sum.data

        init_prob = init_count / np.sum(init_count)
        trans_prob = trans_count.multiply(row_sum)

        return SparseMarkovAssociationDistribution(init_prob, trans_prob, self.alpha, len_dist)

