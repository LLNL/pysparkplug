from pysp.arithmetic import *
import numpy as np
import scipy.linalg
import numpy.ma as ma
import numba


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
        i0 = x[i]
        for j in range(n):
            out[i0,j] += w[i,j]
    return out

@numba.njit('float64[:,:](int32[:], float64[:,:], int32[:], float64[:,:])')
def vec_bincount2(x, w, y, out):
    n = w.shape[1]
    for i in range(len(x)):
        i0 = x[i]
        i1 = y[i]
        for j in range(n):
            out[i0,j] += w[i1,j]
    return out




def sorted_merge(a, b):
    if len(a) < len(b):
        b, a = a, b
    c = np.empty(len(a) + len(b), dtype=a.dtype)
    b_indices = np.arange(len(b)) + np.searchsorted(a, b)
    a_indices = np.ones(len(c), dtype=bool)
    a_indices[b_indices] = False
    c[b_indices] = b
    c[a_indices] = a
    return c

def sorted_dict_merge_add(kvec1, cvec1, kvec2, cvec2):

    if len(kvec2) == 0:
        return kvec1, cvec1
    elif len(kvec1) == 0:
        return kvec2, cvec2

    if len(kvec1) < len(kvec2):
        return sorted_dict_merge_add(kvec2, cvec2, kvec1, cvec1)

    _, idx1, idx2 = np.intersect1d(kvec1, kvec2, assume_unique=True, return_indices=True)

    adj_cnt  = cvec1[idx1] + cvec2[idx2]
    new_vals = np.delete(kvec2, idx2)
    new_cnts = np.delete(cvec2, idx2)
    new_idx = np.searchsorted(kvec1, new_vals)
    rv_vals = np.insert(kvec1, new_idx, new_vals)
    rv_cnts = np.insert(adj_cnt, new_idx, new_cnts)

    return rv_vals, rv_cnts

def make(x):
    return np.asarray(x)

def makepdf(x):
    rv = np.asarray(x)
    if rv.max() == -inf:
        rv = zeros(len(x))-log(len(x))
    return rv


def zeros(n):
    return np.zeros(n)

def diag(x):
    return np.diag(x)

def matinv(x):
    return np.linalg.inv(x)

def dot(x,y):
    return np.dot(x,y)

def outer(x,y):
    return np.outer(x,y)

def diag(x):
    return np.diag(x)

def reshape(x, sz):
    return np.reshape(x, sz)

def cholesky(x):

    try:
        rv = scipy.linalg.cho_factor(x)
    except np.linalg.LinAlgError:
        rv = None

    return rv

def cho_solve(A, b):
    return scipy.linalg.cho_solve(A, b)

#def vecmap(f, x):
#    return np.vectorize(f)(x)

def maximum(x,y,output=None):
    return np.maximum(x,y,output=output)

def log_sum(x):

    max_val = np.max(x)

    if max_val == -np.inf:
        return -np.inf
    else:
        rv = x - max_val
        np.exp(rv, out=rv)
        return np.log(rv.sum()) + max_val


def weighted_log_sum(x, w):

    #bad1 = np.bitwise_and(np.isinf(x), ~np.isinf(w))
    #bad2 = np.bitwise_and(~np.isinf(x), np.isinf(w))
    #bad  = np.any(np.bitwise_or(bad1, bad2))
    y = x + w
    y[np.bitwise_or(np.isinf(x), np.isinf(w))] = -np.inf

    return log_sum(y)


def log_posterior(x):

    maxVal = x.max()

    if isinf(maxVal) or isnan(maxVal):
        return zeros(len(x))-log(len(x))

    mass = log(exp(x - maxVal).sum()) + maxVal
    return x - mass


def posterior(log_x, out=None, log_sum=False):

    if out is None:
        rv = np.zeros(len(log_x))
    else:
        rv = out

    max_val = log_x.max()
    rv_sum  = 0.0

    if isinf(max_val) or isnan(max_val):
        rv.fill(1.0/float(len(log_x)))

    else:
        np.subtract(log_x, max_val, out=rv)
        np.exp(rv, out=rv)
        rv_sum = rv.sum()
        rv /= rv_sum
        rv_sum = np.log(rv_sum) + max_val

    if log_sum:
        return rv, rv_sum
    else:
        return rv


def log_posterior_sum(x):
    maxVal = x.max()
    #maxVal = one if maxVal == -inf else maxVal
    if isinf(maxVal) or isnan(maxVal):
        return zeros(len(x))-log(len(x))

    mass   = log(exp(x - maxVal).sum()) + maxVal
    return x - mass, mass



def weighted_log_posterior(x, w):
    maxVal = -inf

    rv = [None]*len(x)

    for i in range(len(x)):
        r = w[i] + x[i]
        if r > maxVal:
            maxVal = r
        rv[i] = r

    esum = 0.0
    for i in range(len(x)):
        esum  += exp(rv[i] - maxVal)

    mass = log(esum) + maxVal

    for i in range(len(x)):
        rv[i] -= mass

    return(rv)

def weighted_log_posterior_sum(x, w):

    maxVal = -inf

    rv = [None]*len(x)

    for i in range(len(x)):
        r = w[i] + x[i]
        if r > maxVal:
            maxVal = r
        rv[i] = r

    esum = 0.0
    for i in range(len(x)):
        esum  += exp(rv[i] - maxVal)

    mass = log(esum) + maxVal

    for i in range(len(x)):
        rv[i] -= mass

    return rv, mass


def matrix_log_posteriors(x, U, u):

    h               = U.shape[0]
    w               = U.shape[1]
    z               = x.shape[1]

    rowPosteriors   = zeros((h,w,z))
    outerPosterior  = zeros(h)
    outerMax        = -inf

    for i in range(h):

        rowSum = zero

        for j in range(z):
            temp     = U[i,:] + x[:,j]
            innerMax = temp.max()
            temp     = exp(temp - innerMax)
            innerSum = temp.sum()

            rowPosteriors[i,:,j] = temp/innerSum
            rowSum += log(innerSum) + innerMax

        rowSum = rowSum + u[i]
        if rowSum > outerMax:
            outerMax = rowSum
        outerPosterior[i] = rowSum

    outerPosterior   = exp(outerPosterior - outerMax)
    outerSum         = outerPosterior.sum()
    outerPosterior  /= outerSum

    ll = log(outerSum) + outerMax

    return rowPosteriors, outerPosterior, ll
