from __future__ import annotations
from typing import Optional, Sequence, Union, NewType, Tuple, Any
import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, ParameterEstimator
import numpy as np
from pysp.arithmetic import maxrandint


DatumType = 'Sequence[Tuple[int, float]]'
EncodedDatumType = 'Tuple[int, int[:], float[:], int[:], Optional[Any]]'

class IntegerMultinomialDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, minVal: int = 0, pVec: float[:] = None,
                 len_dist: SequenceEncodableProbabilityDistribution = None, name: Optional[str] = None,
                 keys: Optional[str] = None):

        super().__init__()
        pVec = np.empty(0,dtype=np.float64) if pVec is None else pVec

        self.pVec = np.asarray(pVec, dtype=np.float64)
        self.minVal = minVal
        self.maxVal = minVal + self.pVec.shape[0] - 1
        self.logPVec = np.log(self.pVec)
        self.num_vals = self.pVec.shape[0]
        self.len_dist = len_dist
        self.keys = keys
        self.name = name

    def __str__(self):
        s1 = repr(self.minVal)
        s2 = repr(list(self.pVec))
        s3 = str(self.len_dist)
        s4 = repr(self.name)
        return 'IntegerMultinomialDistribution(%s, %s, len_dist=%s, name=%s)' % (s1, s2, s3, s4)

    def density(self, x: DatumType) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: DatumType) -> float:
        rv = 0.0
        for xx,cnt in x:
            rv += (-inf if (xx < self.minVal or xx > self.maxVal) else self.logPVec[xx-self.minVal])*cnt
        return rv

    def seq_log_density(self, x: EncodedDatumType) -> float[:]:

        sz, idx, cnt, val, tcnt = x

        v  = val - self.minVal
        u  = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(v))
        rv.fill(-np.inf)
        rv[u] = self.logPVec[v[u]]
        rv[u] *= cnt[u]
        ll = np.bincount(idx, weights=rv, minlength=sz)

        if self.len_dist is not None:
            ll += self.len_dist.seq_log_density(tcnt)

        return ll


    def seq_encode(self, x: Sequence[DatumType]) -> EncodedDatumType:

        idx = []
        cnt = []
        val = []
        tcnt = []

        for i,y in enumerate(x):
            cc = 0
            for z in y:
                idx.append(i)
                cnt.append(z[1])
                val.append(z[0])
                cc += z[1]
            tcnt.append(cc)

        sz   = len(x)
        idx  = np.asarray(idx, dtype=np.int32)
        cnt  = np.asarray(cnt, dtype=np.float64)
        val  = np.asarray(val, dtype=np.int32)
        tcnt = np.asarray(tcnt, dtype=np.int32)

        if self.len_dist is not None:
            tcnt = self.len_dist.seq_encode(tcnt)
        else:
            tcnt = None

        return sz, idx, cnt, val, tcnt

    def sampler(self, seed: Optional[int] = None) -> IntegerMultinomialSampler:
        return IntegerMultinomialSampler(self, seed)

    def estimator(self, pseudo_count: Optional[int] = None):

        len_est = None if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)

        if pseudo_count is None:
            return IntegerMultinomialEstimator(len_estimator=len_est, name=self.name)
        else:
            return IntegerMultinomialEstimator(minVal=self.minVal, maxVal=self.maxVal, len_estimator=len_est, pseudo_count=pseudo_count, suff_stat=(self.minVal, self.pVec), name=self.name)


class IntegerMultinomialSampler(object):

    def __init__(self, dist: IntegerMultinomialDistribution, seed: Optional[int] = None):
        self.dist = dist
        self.rng = np.random.RandomState(seed)
        self.lenSampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None):

        if size is None:
            cnt = self.lenSampler.sample()
            entry = self.rng.multinomial(cnt, self.dist.pVec)
            rrv = []
            for j in np.flatnonzero(entry):
                rrv.append((j + self.dist.minVal, entry[j]))
            return rrv

        else:
            cnt = self.lenSampler.sample(size=size)
            rv = []

            for i in range(size):
                rrv = []
                entry = self.rng.multinomial(cnt[i], self.dist.pVec)
                for j in np.flatnonzero(entry):
                    rrv.append((j+self.dist.minVal,entry[j]))
                rv.append(rrv)
            return rv


class IntegerMultinomialAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, minVal=None, maxVal=None, name=None, keys=None, len_accumulator=None):

        self.minVal = minVal
        self.maxVal = maxVal
        self.name   = name
        self.len_accumulator = len_accumulator

        if minVal is not None and maxVal is not None:
            self.countVec = vec.zeros(maxVal-minVal+1)
        else:
            self.countVec = None

        self.key = keys

    def update(self, x, weight, estimate):

        cc = 0
        for xx,cnt in x:
            cc += cnt
            if self.countVec is None:
                self.minVal = xx
                self.maxVal = xx
                self.countVec = vec.make([weight*cnt])
            elif self.maxVal < xx:
                tempVec = self.countVec
                self.maxVal = xx
                self.countVec = vec.zeros(self.maxVal - self.minVal + 1)
                self.countVec[:len(tempVec)] = tempVec
                self.countVec[xx - self.minVal] += weight*cnt
            elif self.minVal > xx:
                tempVec = self.countVec
                tempDiff = self.minVal - xx
                self.minVal = xx
                self.countVec = vec.zeros(self.maxVal - self.minVal + 1)
                self.countVec[tempDiff:] = tempVec
                self.countVec[xx - self.minVal] += weight*cnt
            else:
                self.countVec[xx - self.minVal] += weight*cnt

        if self.len_accumulator is not None:
            if estimate is None:
                self.len_accumulator.update(cc, weight, None)
            else:
                self.len_accumulator.update(cc, weight, estimate.len_dist)

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_update(self, x, weights, estimate):

        sz, idx, cnt, val, tenc = x

        min_x = val.min()
        max_x = val.max()

        loc_cnt = np.bincount(val-min_x, weights=cnt*weights[idx])

        if self.countVec is None:
            self.countVec = np.zeros(max_x-min_x+1)
            self.minVal = min_x
            self.maxVal = max_x

        if self.minVal > min_x or self.maxVal < max_x:
            prev_min    = self.minVal
            self.minVal = min(min_x, self.minVal)
            self.maxVal = max(max_x, self.maxVal)
            temp        = self.countVec
            prev_diff   = prev_min - self.minVal
            self.countVec = np.zeros(self.maxVal - self.minVal + 1)
            self.countVec[prev_diff:(prev_diff + len(temp))] = temp

        min_diff = min_x - self.minVal
        self.countVec[min_diff:(min_diff+len(loc_cnt))] += loc_cnt

        if self.len_accumulator is not None:
            if estimate is None:
                self.len_accumulator.seq_update(tenc, weights, None)
            else:
                self.len_accumulator.seq_update(tenc, weights, estimate.len_dist)

    def combine(self, suff_stat):

        if self.countVec is None and suff_stat[1] is not None:
            self.minVal   = suff_stat[0]
            self.maxVal   = suff_stat[0] + len(suff_stat[1]) - 1
            self.countVec = suff_stat[1]

        elif self.countVec is not None and suff_stat[1] is not None:

            if self.minVal == suff_stat[0] and len(self.countVec) == len(suff_stat[1]):
                self.countVec += suff_stat[1]

            else:
                minVal = min(self.minVal, suff_stat[0])
                maxVal = max(self.maxVal, suff_stat[0] + len(suff_stat[1]) - 1)

                countVec = vec.zeros(maxVal-minVal+1)

                i0 = self.minVal - minVal
                i1 = self.maxVal - minVal + 1
                countVec[i0:i1] = self.countVec

                i0 = suff_stat[0] - minVal
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - minVal + 1
                countVec[i0:i1] += suff_stat[1]

                self.minVal   = minVal
                self.maxVal   = maxVal
                self.countVec = countVec

        if self.len_accumulator is not None:
            self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self):
        if self.len_accumulator is None:
            return self.minVal, self.countVec, None
        else:
            return self.minVal, self.countVec, self.len_accumulator.value()

    def from_value(self, x):
        self.minVal   = x[0]
        self.maxVal   = x[0] + len(x[1]) - 1
        self.countVec = x[1]
        if self.len_accumulator is not None:
            self.len_accumulator.from_value(x[2])

    def key_merge(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)


class IntegerMultinomialAccumulatorFactory(object):

    def __init__(self, minVal, maxVal, name, keys, len_factory):
        self.minVal = minVal
        self.maxVal = maxVal
        self.name = name
        self.len_factory = len_factory
        self.keys = keys

    def make(self):
        len_acc = None if self.len_factory is None else self.len_factory.make()
        return IntegerMultinomialAccumulator(minVal=self.minVal, maxVal=self.maxVal, name=self.name, keys=self.keys, len_accumulator=len_acc)


class IntegerMultinomialEstimator(ParameterEstimator):

    def __init__(self, minVal=None, maxVal=None, len_estimator=None, len_dist=None, name=None, pseudo_count=None, suff_stat=None, keys=None):

        self.suff_stat = suff_stat
        self.pseudo_count = pseudo_count
        self.minVal = minVal
        self.maxVal = maxVal
        self.len_estimator = len_estimator
        self.len_dist = len_dist
        self.keys = keys
        self.name = name

    def accumulatorFactory(self):

        minVal = None
        maxVal = None

        if self.suff_stat is not None:
            minVal = self.suff_stat[0]
            maxVal = minVal + len(self.suff_stat[1]) - 1
        elif self.minVal is not None and self.maxVal is not None:
            minVal = self.minVal
            maxVal = self.maxVal

        len_factory = None if self.len_estimator is None else self.len_estimator.accumulatorFactory()
        return IntegerMultinomialAccumulatorFactory(minVal=minVal, maxVal=maxVal, name=self.name, keys=self.keys, len_factory=len_factory)


    def estimate(self, nobs, suff_stat):


        if self.len_estimator is not None:
            len_dist = self.len_estimator.estimate(nobs, suff_stat[2])
        elif self.len_dist is not None:
            len_dist = self.len_dist
        else:
            len_dist = None

        if self.pseudo_count is not None and self.suff_stat is None:

            pseudo_countPerLevel = self.pseudo_count / float(len(suff_stat[1]))
            adjustedNobs        = suff_stat[1].sum() + self.pseudo_count

            return IntegerMultinomialDistribution(suff_stat[0], (suff_stat[1]+pseudo_countPerLevel)/adjustedNobs, len_dist=len_dist, name=self.name, keys=self.keys)

        elif self.pseudo_count is not None and self.minVal is not None and self.maxVal is not None:

            minVal = min(self.minVal, suff_stat[0])
            maxVal = max(self.maxVal, suff_stat[0] + len(suff_stat[1]) - 1)

            countVec = vec.zeros(maxVal - minVal + 1)

            i0 = suff_stat[0] - minVal
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - minVal + 1
            countVec[i0:i1] += suff_stat[1]

            pseudo_countPerLevel = self.pseudo_count / float(len(countVec))
            adjustedNobs        = suff_stat[1].sum() + self.pseudo_count

            return IntegerMultinomialDistribution(minVal, (countVec+pseudo_countPerLevel)/adjustedNobs, len_dist=len_dist, name=self.name, keys=self.keys)

        elif self.pseudo_count is not None and self.suff_stat is not None:

            sMaxVal = self.suff_stat[0] + len(self.suff_stat[1]) - 1
            sMinVal = self.suff_stat[0]

            minVal = min(sMinVal, suff_stat[0])
            maxVal = max(sMaxVal, suff_stat[0] + len(suff_stat[1]) - 1)

            countVec = vec.zeros(maxVal - minVal + 1)

            i0 = sMinVal - minVal
            i1 = sMaxVal - minVal + 1
            countVec[i0:i1] = self.suff_stat[1]*self.pseudo_count

            i0 = suff_stat[0] - minVal
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - minVal + 1
            countVec[i0:i1] += suff_stat[1]

            return IntegerMultinomialDistribution(minVal, countVec/(countVec.sum()), len_dist=len_dist, name=self.name, keys=self.keys)


        else:
            return IntegerMultinomialDistribution(suff_stat[0], suff_stat[1]/(suff_stat[1].sum()), len_dist=len_dist, name=self.name, keys=self.keys)

