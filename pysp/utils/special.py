from scipy.special import gammaln
from scipy.special import gammaln, betaln, beta
from scipy.special import digamma, psi  # as digammaS0
from scipy.special import zeta, gamma, polygamma
import numpy as np
import math
from pysp.arithmetic import *


def logpdet(X: np.ndarray) -> float:
    """
	Computes the log-pseudo-determinant for a symmetric dense matrix.
    :param X:
    :return: float, log-pseudo-determinant
    """
    eigs = np.abs(np.linalg.eig(X))
    eigs = eigs[eigs != 0]

    if len(eigs) > 0:
        return float(np.sum(np.log(eigs)))
    else:
        return -math.inf


def stirling2(n: int, k: int) -> int:
    assert n > 0 and k > 0

    if n == 0 and k == 0:
        return 1
    elif n == 0:
        return 0
    elif k == 0:
        return 0
    else:
        return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)


def polygamma_loc(n, y, out=None):
    if out is not None:
        fac2 = zeta(n + 1, y, out=out)
        fac2 *= (-1.0) ** (n + 1) * gamma(n + 1.0)
    else:
        fac2 = (-1.0) ** (n + 1) * gamma(n + 1.0) * zeta(n + 1, y)

    return fac2


def trigamma(y, out=None):
    return zeta(2, y, out=out)


'''
def digammaS1(y):
    if y <= 0.0:
        return -inf
    else:
        return digammaS0(y)

digammaS2 = vectorize(digammaS1)

def digamma(y):
    return digammaS2(y)
'''

'''
def digammainv(y, res=1.0e-6):

    if y == -inf:
        return 0.0

    x = (exp(y) + half) if y >= -2.22 else (-one/(y-digamma(one)))

    temp = digamma(x)-y
    ee = abs(temp)
    maxIts = 20
    while maxIts != 0 and ee > res:
        maxIts -= 1
        #x = x - ((digamma(x) - y) / trigamma(x))
        x -= ((temp) / trigamma(x))
        temp = digamma(x) - y
        #ee = abs(digamma(x) - y)
        ee = abs(temp)

        #if ee > res:
        #    break

    return x
'''

digamma1 = digamma(1.0)


def digammainv(y, out=None):
    # if y == -inf:
    #    return 0.0

    if isinstance(y, np.ndarray):

        rv = np.zeros(y.shape, dtype=float)
        rv[np.isposinf(y)] = np.inf

        Q = np.isfinite(y)
        z = y[Q]
        M = (z >= -2.22)
        x = M * (exp(z) + 0.5) + (1.0 - M) * (-1.0 / (z - digamma1))

        t1 = np.zeros(x.shape, dtype=float)
        t2 = np.zeros(x.shape, dtype=float)

        for i in range(5):
            digamma(x, out=t1)
            zeta(2, x, out=t2)

            # if np.any(t2 == 0) or np.any(np.isnan(t2)) or np.any(np.isinf(t2)):
            #    print('bad')
            # if np.any(np.isnan(t1)) or np.any(np.isinf(t1)):
            #    print('bad')

            t1 -= z
            t1 /= t2
            x -= t1

        rv[Q] = x
        x = rv

    else:

        M = (y >= -2.22)
        x = M * (exp(y) + 0.5) + (1.0 - M) * (-1.0 / (y - digamma1))

        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))
        x -= ((digamma(x) - y) / trigamma(x))

    return x
