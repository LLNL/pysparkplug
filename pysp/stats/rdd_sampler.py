from pyspark import SparkContext, SparkConf
from numpy.random import RandomState
from pysp.arithmetic import *
import numpy as np
import pickle
from pysp.arithmetic import maxrandint


def take_sample(rdd, withReplacement, n, seed=None):

	rng = RandomState(seed)
	sample = rdd.zipWithUniqueId().takeSample(withReplacement, n, rng.randint(0, maxrandint))
	sidx   = np.argsort([u[1] for u in sample])
	sample = [sample[i][0] for i in sidx]
	sidx   = np.argsort(rng.uniform(size=n))
	return [sample[i] for i in sidx]

def sample_seq_as_rdd(sc, dist, seq_len, count_per_split, num_splits, seed=None):

	distB = sc.broadcast(dist)
	seeds = RandomState(seed).randint(0, maxrandint, size=num_splits)

	def fmap(u):
		ddist   = distB.value
		sampler = [ddist.sampler(seed=h) for h in u]
		return iter([v for h in sampler for v in h.sample_seq(seq_len, size=count_per_split)])

	return sc.parallelize(seeds, num_splits).mapPartitions(fmap, True)


def sample_rdd(sc, dist, count_per_split, num_splits, seed=None):

	dd = pickle.dumps(dist, protocol=0)
	distB = sc.broadcast(dd)
	seeds = RandomState(seed).randint(0, maxrandint, size=num_splits)


	def fmap(u):
		ddist   = pickle.loads(distB.value)
		sampler = [ddist.sampler(seed=h) for h in u]
		return iter([v for h in sampler for v in h.sample(size=count_per_split)])

	return sc.parallelize(seeds, num_splits).mapPartitions(fmap, True)
