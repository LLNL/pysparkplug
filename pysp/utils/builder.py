from pysp.stats import *
import numpy as np # This is for the eval-lambdas

def read_index_csv(filename):
	fin   = open(filename, 'r')
	lines = map(lambda v: v.split('#',1)[0].split(',', 3), fin.read().split('\n'))
	fin.close()
	lines = filter(lambda v: len(v) == 4, lines)

	return lines


def get_indexed_rdd_pne(field_info=None, filename=None):

	if filename is not None and field_info is None:
		field_info = read_index_csv(filename)

	def entry_lambda(idx, mapstr):
		if mapstr != '':
			tempLambda0 = eval('lambda x: ' + mapstr)
			tempLambda  = lambda u: tempLambda0(u[idx])
		else:
			tempLambda = lambda u: u[idx]

		def ff(entry):
			rv = tempLambda(entry)
			return rv

		return ff

	parser_list    = []
	estimator_list = []
	maxidx         = -1

	for entry in field_info:
		idx, name, lam, dist = entry
		estimator = eval(dist)

		if estimator is not None:
			idxi = int(idx)
			parser_list.append(entry_lambda(idxi, lam.strip()))
			estimator_list.append(estimator)
			maxidx = idxi if idxi > maxidx else maxidx

	def line_parser(line):
		parts = line.split(',')
		if len(parts) < (maxidx+1):
			return None
		else:
			return tuple([parser(parts) for parser in parser_list])

	estimator = CompositeEstimator(tuple(estimator_list))

	return estimator,line_parser
