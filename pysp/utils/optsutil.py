import numpy as np
from collections import defaultdict

def map_to_integers(x, val_map):

	rv = [None]*len(x)
	for i,u in enumerate(x):
		if u not in val_map:
			val_map[u] = len(val_map)
		rv[i] = val_map[u]
	return rv

def get_inv_map(val_map):
	max_val = max(val_map.values())

	rv = [None]*(max_val+1)

	for k,v in val_map.items():
		rv[v] = k

	return rv

def textFile(f):
	fin = open(f, 'r')
	rv  = fin.read()

	if rv is not None and len(rv) > 0 and rv[-1] == '\n':
		return rv[:-1].split('\n')
	else:
		return rv.split('\n')

def reduceByKey(f, x):
	rv = dict()

	for key,val in x:
		if key in rv:
			rv[key] = f(rv[key], val)
		else:
			rv[key] = val

	return rv


def sumByKey(x):
	rv = dict()

	for key, val in x:
		if key in rv:
			rv[key] += val
		else:
			rv[key] = val

	return rv

def groupByKey(x):
	rv = defaultdict(list)

	for key, val in x:
		#if key in rv:
		rv[key].append(val)
		#else:
		#	rv[key] = [val]

	return rv

def groupBy(f, x):
	rv = defaultdict(list)

	for val in x:
		key = f(val)
		#if key in rv:
		rv[key].append(val)
		#else:
		#	rv[key] = [val]

	return rv


def countByValue(x):
	rv = dict()

	for u in x:
		rv[u] = rv.get(u,0) + 1

	return rv


def flatMap(f, x):
	return [u for v in x for u in f(v)]


def leastOccurring(x, count=None, percent=None, keep_freq=True):

	cntMap = countByValue(x).items()
	sidx   = np.argsort([u[1] for u in cntMap])

	if count is not None:
		n = min(len(sidx), count)
	elif percent is not None:
		n = max(int(len(sidx)*percent), 1)
	else:
		return x

	vals   = [cntMap[i][0] for i in sidx[:n]]

	if keep_freq:
		vals = set(vals)
		return filter(lambda u: u in vals, x)
	else:
		return vals