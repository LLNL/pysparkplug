from typing import Tuple
import numpy as np
from scipy.optimize import root_scalar
from pysp.utils.automatic import get_estimator, get_dpm_mixture
from pysp.bstats.bestimation import optimize
from pysp.bstats import *


def adj_perplexity(x, ss):
    s = 1 / ss
    P = -x * s
    M = P.max()
    P = np.exp(P - M)
    sumP = P.sum()
    H = np.log(sumP) + M + s * np.sum(x * P) / sumP
    P = P / sumP
    return H, P


def vec_perplexity(x, s):
    H, P = adj_perplexity(x, s)
    return H


def row_perplexity_solve(x, a, s0, s1, d=10):
    if d == 0:
        return (s0 + s1) / 2.0
    s2 = (s0 + s1) / 2.0
    f0 = vec_perplexity(x, s0)
    f1 = vec_perplexity(x, s1)
    f2 = vec_perplexity(x, s2)

    if f0 >= a:
        return s0
    elif f1 <= a:
        return s1
    elif f2 > a:
        return row_perplexity_solve(x, a, s0, s2, d - 1)
    elif f2 < a:
        return row_perplexity_solve(x, a, s2, s1, d - 1)
    else:
        return s2


def fix_row_perplexity(P, a):
    rv = np.zeros([P.shape[0]] * 2)
    ent_p = np.log2(a)*np.log(2)
    for i in range(P.shape[0]):
        x = P[i, :].copy()
        x /= x.sum()
        x = np.concatenate((x[:i], x[(i + 1):]))
        x = -np.log(x)
        c = row_perplexity_solve(x, ent_p, 1.0e-12, 1000, 20)
        _, x = adj_perplexity(x, c)
        rv[i, :i] = x[:i]
        rv[i, (i + 1):] = x[i:]

    return rv


def get_pmat_vlen(posterior_mat, ll_mat, targ_perplexity=None):

    with np.errstate(divide='ignore'):

        n = len(posterior_mat)
        z_ij = posterior_mat
        l_ij = ll_mat
        v_ij = l_ij.max(axis=1, keepdims=True)
        g_ij = np.exp(l_ij - v_ij)
        p_ij = np.dot(g_ij, z_ij.T)
        p_ij[range(n), range(n)] = 0
        np.log(p_ij, out=p_ij)
        p_ij += v_ij.T
        p_ij -= np.max(p_ij, axis=1, keepdims=True)
        np.exp(p_ij, out=p_ij)
        p_ij /= np.sum(p_ij, axis=1, keepdims=True)
        p_ij /= np.sum(p_ij, axis=0, keepdims=True)
        p_ij = p_ij.T

        np.maximum(p_ij, 1.0e-128, out=p_ij)
        if targ_perplexity is not None:
            p_ij = fix_row_perplexity(p_ij, targ_perplexity)

        p_ij /= n
        return p_ij


def get_pmat(posterior_mat, ll_mat, targ_perplexity=None, vlen=False):

    if vlen:
        return get_pmat_vlen(posterior_mat, ll_mat, targ_perplexity)

    with np.errstate(divide='ignore'):

        n = len(posterior_mat)
        z_ij = posterior_mat
        l_ij = ll_mat
        v_ij = l_ij.max(axis=1, keepdims=True)
        g_ij = np.exp(l_ij - v_ij)
        p_ij = np.dot(g_ij, z_ij.T)
        p_ij[range(n), range(n)] = 0
        np.log(p_ij, out=p_ij)
        p_ij += v_ij
        p_ij -= np.max(p_ij, axis=0, keepdims=True)
        np.exp(p_ij, out=p_ij)
        p_ij /= np.sum(p_ij, axis=0, keepdims=True)
        p_ij = p_ij.T

        np.maximum(p_ij, 1.0e-128, out=p_ij)
        if targ_perplexity is not None:
            p_ij = fix_row_perplexity(p_ij, targ_perplexity)

        p_ij /= n
        return p_ij


def t_cond_prob_mat(tx: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    n, m = tx.shape[0:2]

    rsum = np.sum(np.square(tx), axis=1, keepdims=True)
    d_ij = np.dot(-2 * tx, tx.T)
    d_ij += rsum
    d_ij += rsum.T + 1
    np.power(d_ij, -(alpha + 1.0)/2.0, out=d_ij)

    d_ij[np.arange(n), np.arange(n)] = 0
    q_ij = d_ij / np.sum(d_ij)

    return q_ij, d_ij


def t_cond_prob_mat_alpha(tx: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    n, m = tx.shape[0:2]

    rsum = np.sum(np.square(tx), axis=1, keepdims=True)
    d_ij = np.dot(-2.0 * tx, tx.T)
    d_ij += rsum
    d_ij += rsum.T

    c_ij = np.power((d_ij/alpha) + 1.0, -(alpha + 1.0)/2.0)
    c_ij[np.arange(n), np.arange(n)] = 0
    c_ij /= np.sum(c_ij)

    return c_ij, d_ij


def update_embed(P: np.ndarray, Y: np.ndarray, iY: np.ndarray, gains: np.ndarray, momentum: float, eta: float, alpha: float, min_gain: float, min_value: float = 1.0e-128) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nn, mm = Y.shape

    Q, d_ij = t_cond_prob_mat(Y, alpha)
    np.maximum(Q, min_value, out=Q)
    PQ = P - Q
    dC = np.zeros((nn, mm))

    for i in range(nn):
        dCi = ((PQ[i, :] * d_ij[i, :]))
        dC[i, :] = np.dot((Y[i, None, :] - Y).T, dCi)
    dC *= (2.0*alpha + 2.0)/alpha

    gains = (gains + 0.2) * ((dC > 0) != (iY > 0)) + (gains * 0.8) * ((dC > 0) == (iY > 0))
    gains[gains < min_gain] = min_gain

    iY = momentum * iY - eta * (gains * dC)
    Y += iY
    Y -= np.mean(Y, axis=0, keepdims=True)

    return Y, iY, gains, Q


def update_alpha(P, Y, alpha: float, min_alpha: float, min_value: float, max_its: int = 30, step: float = 0.1, eps: float = 1.0e-4) -> float:

    Q, e_ij = t_cond_prob_mat_alpha(Y, alpha)
    np.maximum(Q, min_value, out=Q)
    PQ = P - Q
    GG = np.bitwise_and(P > 0, Q > 0)
    CC = np.log(P[GG] / Q[GG]) * P[GG]
    CC = CC.sum()
    its = 0

    while True:

        e_ij_a = (e_ij/alpha) + 1
        dCa = (np.log(e_ij_a)*0.5 + (((-alpha - 1.0)/(2.0*alpha*alpha))*e_ij/e_ij_a)) * PQ
        dCa = dCa[GG].sum()

        if np.isfinite(dCa) and (dCa != 0):
            if (alpha - (CC / dCa)) < (alpha*(1.0-step)):
                alpha *= (1.0-step)
            elif (alpha - (CC / dCa)) > (alpha*(1.0+step)):
                alpha *= (1.0+step)
            elif np.abs(dCa) > 0.0:
                alpha = alpha - (CC / dCa)

        alpha = max(alpha, min_alpha)

        its += 1
        if its >= max_its:
            break

        CC_old = CC
        Q, e_ij = t_cond_prob_mat_alpha(Y, alpha)
        np.maximum(Q, min_value, out=Q)
        PQ = P - Q
        GG = np.bitwise_and(P > 0, Q > 0)
        CC = np.log(P[GG] / Q[GG]) * P[GG]
        CC = CC.sum()

        if CC_old - CC < eps:
            break

    return alpha


def htsne(data, emb_dim=2, alpha=1.0, max_components=30, mix_threshold_count=0.5, Y=None, perplexity=None, max_its=1000, print_iter=100, eta=500, momentum=0.8, min_gain=0.01, min_value=1.0e-128, optimize_alpha=False, min_alpha=1.0e-6, max_alpha_its=3, seed=None, comp_estimator=None, mix_model=None, variable_length=False):

    if max_components <= 1 or not isinstance(max_components, (int, np.int_)):
        raise Exception('max_components must be and integer greater than 1.')

    if mix_model is None:
        mix_model = get_dpm_mixture(data, estimator=comp_estimator, max_comp=max_components, rng=np.random.RandomState(seed), max_its=max_its, print_iter=print_iter, mix_threshold_count=mix_threshold_count)

    if mix_model.num_components == 0:
        raise Exception('Something is broken. Mixture model has zero components.')

    enc_data = mix_model.seq_encode(data)
    z_ij = mix_model.seq_posterior(enc_data)
    l_ij = mix_model.seq_component_log_density(enc_data)

    P = get_pmat(z_ij, l_ij, targ_perplexity=perplexity, vlen=variable_length)
    P = np.asarray(P)
    P += P.T
    P /= np.sum(P)
    np.maximum(P, min_value, out=P)

    if Y is None:
        rng = np.random.RandomState(seed)
        nn = P.shape[0]
        Y = rng.randn(nn, emb_dim) * 1.0e-4
        iY = np.zeros((nn, emb_dim))
        gains = np.zeros((nn, emb_dim))
        P *= 4
        for i in range(20):
            Y, iY, gains, Q = update_embed(P, Y, iY, gains, 0.5, eta, alpha, min_gain, min_value)
        for i in range(80):
            Y, iY, gains, Q = update_embed(P, Y, iY, gains, momentum, eta, alpha, min_gain, min_value)
        P /= 4
    else:
        Y = np.asarray(Y)
        nn = Y.shape[0]
        emb_dim = Y.shape[1]
        iY = np.zeros((nn, emb_dim))
        gains = np.zeros((nn, emb_dim))

    for i in range(1, max_its + 1):
        Y, iY, gains, Q = update_embed(P, Y, iY, gains, momentum, eta, alpha, min_gain, min_value)
        if optimize_alpha:
            alpha = update_alpha(P, Y, alpha, min_alpha, min_value, max_alpha_its)
        if (i % print_iter) == 0:
            KL = np.bitwise_and(P > 0, Q > 0)
            KL = np.dot(P[KL], (np.log(P[KL]) - np.log(Q[KL])))
            print('Iteration %d: alpha = %f, KL(P||Q)=%f' % (i, alpha, KL))

    return Y


def dpmsne(P=None, data=None, emb_dim=2, alpha=1.0, max_components=30, mix_threshold_count=0.5, Y=None, perplexity=None, max_its=1000, print_iter=100, eta=500, momentum=0.8, min_gain=0.01, min_value=1.0e-128, optimize_alpha=False, min_alpha=1.0e-6, max_alpha_its=3, seed=None, comp_estimator=None, mix_model=None, variable_length=False):


    P = np.asarray(P)
    P /= np.sum(P)

    if Y is None:
        rng = np.random.RandomState(seed)
        nn = P.shape[0]
        Y = rng.randn(nn, emb_dim) * 1.0e-4
        iY = np.zeros((nn, emb_dim))
        gains = np.zeros((nn, emb_dim))
        P *= 4
        for i in range(20):
            Y, iY, gains, Q = update_embed(P, Y, iY, gains, 0.5, eta, alpha, min_gain, min_value)
        for i in range(80):
            Y, iY, gains, Q = update_embed(P, Y, iY, gains, momentum, eta, alpha, min_gain, min_value)
        P /= 4
    else:
        Y = np.asarray(Y)
        nn = Y.shape[0]
        emb_dim = Y.shape[1]
        iY = np.zeros((nn, emb_dim))
        gains = np.zeros((nn, emb_dim))

    for i in range(1, max_its + 1):
        Y, iY, gains, Q = update_embed(P, Y, iY, gains, momentum, eta, alpha, min_gain, min_value)
        if optimize_alpha:
            alpha = update_alpha(P, Y, alpha, min_alpha, min_value, max_alpha_its)
        if (i % print_iter) == 0:
            KL = np.bitwise_and(P > 0, Q > 0)
            KL = np.dot(P[KL], (np.log(P[KL]) - np.log(Q[KL])))
            print('Iteration %d: alpha = %f, KL(P||Q)=%f' % (i, alpha, KL))

    return Y


