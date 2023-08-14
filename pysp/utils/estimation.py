import numpy as np
import sys
import time
from pysp.stats import initialize, seq_estimate, seq_log_density_sum, seq_encode, seq_log_density

def empirical_kl_divergence(dist1, dist2, enc_data):

	ll = seq_log_density(enc_data, estimate=(dist1, dist2), is_list=True)

	r1 = 0.0
	r2 = 0
	r3 = 0

	ll = np.hstack(ll)

	l1 = ll[0, :]
	l2 = ll[1, :]
	g1 = np.bitwise_and(l1 != -np.inf, ~np.isnan(l1))
	g2 = np.bitwise_and(l2 != -np.inf, ~np.isnan(l2))
	gg = np.bitwise_and(g1, g2)

	max_l1 = np.max(l1[gg])
	max_l2 = np.max(l2[gg])


	p1 = np.exp(l1[gg] - max_l1)
	p1 /= p1.sum()

	p2 = np.exp(l2[gg] - max_l2)
	p2 /= p2.sum()

	r1 = (p1[gg]*(np.log(p1[gg]) - np.log(p2[gg]))).sum()
	r2 = (~g1).sum()
	r3 = (~g2).sum()

	return r1, r2, r3

def k_fold_split_index(sz, k, rng):

	idx  = rng.rand(sz)
	sidx = np.argsort(idx)

	rv = np.zeros(sz, dtype=int)
	for i in k:
		rv[sidx[np.arange(start=i, stop=sz, step=k, dtype=int)]] = i

	return rv


def partition_data_index(sz, pvec, rng):

	idx  = rng.rand(sz)
	sidx = np.argsort(idx)

	rv = []
	p_tot = 0
	prev_idx = 0

	for p in pvec:
		next_idx = int(round(sz*(p_tot + p), 0))
		rv.append(sidx[prev_idx:next_idx])
		p_tot += p
		prev_idx = next_idx

	return rv

def partition_data(data, pvec, rng):

	idx_list = partition_data_index(len(data), pvec, rng)

	return [[data[i] for i in u] for u in idx_list]



def best_of(data, vdata, est, trials, max_its, init_p, delta, rng, init_estimator=None, enc_data=None, enc_vdata=None, out=sys.stdout, print_iter=1):

	rv_ll = -np.inf
	rv_mm = None


	if init_estimator is None:
		iest = est
	else:
		iest = init_estimator

	for kk in range(trials):

		mm = initialize(data, iest, rng, init_p)

		if enc_data is None:
			enc_data = seq_encode(data, mm)
		if enc_vdata is None:
			enc_vdata = seq_encode(vdata, mm)

		_, old_ll = seq_log_density_sum(enc_data, mm)
		#_, old_vll = seq_log_density_sum(enc_vdata, mm)

		for i in range(max_its):

			mm_next = seq_estimate(enc_data, est, mm)
			_, ll = seq_log_density_sum(enc_data, mm_next)
			#_, vll = seq_log_density_sum(enc_vdata, mm_next)

			#dvll = vll - old_vll
			dll = ll - old_ll

			if (i+1) % print_iter == 0:
				out.write('Iteration %d. LL=%f, delta LL=%e\n'% (i+1, ll, dll))

			if (dll >= 0) or (delta is None):
				mm = mm_next

			if (delta is not None) and (dll < delta):
				break

			old_ll = ll
			#old_vll = vll


		_, vll = seq_log_density_sum(enc_vdata, mm)
		out.write('Trial %d. VLL=%f\n' % (kk + 1, vll))

		if vll > rv_ll:
			rv_mm = mm
			rv_ll = vll

	return rv_ll, rv_mm


def optimize(data, estimator, max_its=10, delta=1.0e-9, init_estimator=None, init_p=0.1, rng=np.random.RandomState(), prev_estimate=None, vdata=None, enc_data=None, enc_vdata=None, out=sys.stdout, print_iter=1, num_chunks=1):

	if data is None and enc_data is None:
		raise Exception('Optimization called with empty data or enc_data.')

	if init_estimator is None:
		iest = estimator
	else:
		iest = init_estimator

	if prev_estimate is None:
		mm = initialize(data, iest, rng, init_p)
	else:
		mm = prev_estimate

	if enc_data is None:
		enc_data = seq_encode(data, mm, num_chunks=num_chunks)

	if enc_vdata is None and vdata is not None:
		enc_vdata = seq_encode(vdata, mm, num_chunks=num_chunks)

	_, old_ll = seq_log_density_sum(enc_data, mm)

	if enc_vdata is not None:
		_, old_vll = seq_log_density_sum(enc_vdata, mm)
	else:
		old_vll = old_ll

	best_model = mm
	best_vll = old_vll
	best_itr = 0

	for i in range(max_its):

		mm_next  = seq_estimate(enc_data, estimator, mm)
		cnt, ll  = seq_log_density_sum(enc_data, mm_next)

		if enc_vdata is not None:
			_, vll = seq_log_density_sum(enc_vdata, mm_next)
		else:
			vll = ll

		dll = ll - old_ll

		if (dll >= 0) or (delta is None):
			mm = mm_next

		if (delta is not None) and (dll < delta):
			if enc_vdata is not None:
				out.write('Iteration %d: ln[P(Data|Model)]=%e, ln[P(Data|Model)]-ln[P(Data|PrevModel)]=%e, ln[P(Valid Data|Model)]=%e\n' % (i + 1, ll, dll, vll))
			else:
				out.write('Iteration %d: ln[P(Data|Model)]=%e, ln[P(Data|Model)]-ln[P(Data|PrevModel)]=%e\n' % (i + 1, ll, dll))
			break

		if (i+1) % print_iter == 0:
			if enc_vdata is not None:
				out.write('Iteration %d: ln[P(Data|Model)]=%e, ln[P(Data|Model)]-ln[P(Data|PrevModel)]=%e, ln[P(Valid Data|Model)]=%e\n'% (i+1, ll, dll, vll))
			else:
				out.write('Iteration %d: ln[P(Data|Model)]=%e, ln[P(Data|Model)]-ln[P(Data|PrevModel)]=%e\n' % (i + 1, ll, dll))

		old_ll = ll

		if best_vll < vll:
			best_vll = vll
			best_model = mm
			best_itr = i+1

	return best_model

def iterate(data, estimator, max_its, prev_estimate=None, init_p=0.1, rng=np.random.RandomState(), out=sys.stdout, is_encoded=False, init_estimator=None, print_iter=1):

	if init_estimator is None:
		iest = estimator
	else:
		iest = init_estimator

	if prev_estimate is None:
		mm = initialize(data, iest, rng, init_p)
	else:
		mm = prev_estimate

	if is_encoded:
		enc_data = data
	else:
		enc_data = seq_encode(data, mm)

	if hasattr(enc_data, 'cache'):
		enc_data.cache()

	t0 = time.time()
	for i in range(max_its):

		mm = seq_estimate(enc_data, estimator, mm)

		if (i+1) % print_iter == 0:
			out.write('Iteration %d\t E[dT]=%f.\n'% (i+1, (time.time()-t0)/float(i+1)))

	return mm


def hill_climb(data, vdata, estimator, prev_estimate, max_its, metric_lambda, best_estimate=None, enc_data=None, enc_vdata=None, out=sys.stdout, print_iter=1):

	mm = prev_estimate

	if enc_data is None:
		enc_data = mm.seq_encode(data)
		enc_data = [(len(data), enc_data)]
	if enc_vdata is None:
		enc_vdata = mm.seq_encode(vdata)
		enc_vdata = [(len(vdata), enc_vdata)]

	best_model = prev_estimate if best_estimate is None else best_estimate
	_, best_ll = seq_log_density_sum(enc_vdata, best_model)
	best_score = metric_lambda(vdata, best_model)

	for i in range(max_its):

		mm_next = seq_estimate(enc_data, estimator, mm)

		_, next_ll = seq_log_density_sum(enc_vdata, mm_next)
		next_score = metric_lambda(vdata, mm_next)

		if (next_score > best_score) or ((next_score == best_score) and (best_ll < next_ll)):
			best_model = mm_next
			best_ll    = next_ll
			best_score = next_score


		if i % print_iter == 0:
			out.write('Iteration %d. LL=%f, Best LL=%f, Best Score=%f\n'% (i+1, next_ll, best_ll, best_score))

		mm = mm_next

	return best_model


