"""featgap.screen -- interaction screens for the gap.

Which feature PAIRS jointly inform the residual beyond their marginals? This is
the detector for interactions with little/no marginal signal -- the XOR-like
gaps a greedy axis miner can never seed.

Two measures:
  * interaction_information = MI(y; i,j) - MI(y; i) - MI(y; j), via histogram MI
    (fast, scalable). Positive == synergy.
  * hsic = Hilbert-Schmidt Independence Criterion (kernel dependence) on a
    subsample -- the rigorous nonparametric dependence measure. This is the same
    HSIC contrasted with attention at the very start of this project: attention
    builds the rules; HSIC finds the dependence the rules can't reach.
"""

from __future__ import annotations

import numpy as np


def mutual_information(a, b, na, nb):
    """MI of two integer-coded discrete variables (nats)."""
    n = len(a)
    joint = np.zeros((na, nb))
    np.add.at(joint, (a, b), 1)
    joint /= n
    pa = joint.sum(1)
    pb = joint.sum(0)
    nz = joint > 0
    denom = (pa[:, None] * pb[None, :])[nz]
    return float((joint[nz] * np.log(joint[nz] / denom)).sum())


def _bin(x, bins):
    e = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(e, x).astype(np.int64)


def interaction_screen(X, label, *, mask=None, bins=8, top_k=15):
    """Rank feature pairs by interaction information w.r.t. a binary `label`.

    Returns (pairs, marginals) where pairs = [(i, j, synergy, mi_joint,
    mi_i, mi_j), ...] sorted by synergy desc. A high-synergy pair with ~zero
    marginals is exactly a hidden interaction (e.g. a ring, an XOR).
    """
    if mask is not None:
        X, label = X[mask], label[mask]
    y = label.astype(np.int64)
    F = X.shape[1]
    codes = [_bin(X[:, f], bins) for f in range(F)]
    marg = [mutual_information(y, codes[f], 2, bins) for f in range(F)]
    out = []
    for i in range(F):
        for j in range(i + 1, F):
            joint = codes[i] * bins + codes[j]
            mij = mutual_information(y, joint, 2, bins * bins)
            out.append((i, j, mij - marg[i] - marg[j], mij, marg[i], marg[j]))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:top_k], marg


def hsic(x, y, *, sample=1500, seed=0):
    """Biased HSIC with RBF kernels (median heuristic) on a subsample.

    x, y may be (n,) or (n, d). Returns a non-negative dependence score; larger
    means stronger (possibly nonlinear) dependence. O(sample^2).
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(sample, n), replace=False)
    X = np.asarray(x)[idx].astype(float).reshape(len(idx), -1)
    Y = np.asarray(y)[idx].astype(float).reshape(len(idx), -1)

    def K(A):
        d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        return np.exp(-d2 / (med if med > 0 else 1.0))

    m = len(idx)
    H = np.eye(m) - np.ones((m, m)) / m
    Kx, Ky = K(X), K(Y)
    return float(np.trace(Kx @ H @ Ky @ H) / (m - 1) ** 2)
