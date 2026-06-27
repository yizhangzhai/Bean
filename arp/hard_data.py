"""Hard synthetic fraud: deep, disjunctive, banded, heavy-tailed, and adversarial.

Each pattern is a DNF (OR of conjunctions) over conditions:
  (">", f, p)        feature f above its p-quantile
  ("<", f, p)        feature f below its p-quantile
  ("band", f, lo, hi) lo-quantile < f < hi-quantile   (non-monotonic)

Patterns are deliberately varied in depth (2..6), structure (conjunctive /
disjunctive / banded), feature scale (incl. heavy-tailed), and difficulty -- and
`xor_gated` is an adversarial interaction with ~zero marginal signal, included
to probe the known failure mode of greedy beam search. Decoy features correlate
with real signal features but appear in no pattern, to test false discovery.
"""

from __future__ import annotations

import numpy as np

# feature-index map (keep patterns on disjoint indices; decoys at 40-41)
# Thresholds are calibrated so each pattern's region is rare-but-learnable
# (~0.2-0.5% of rows), and bg is low enough that the planted signature -- not
# label noise -- is the dominant source of positives (else recall-vs-label is
# meaningless for deep patterns whose region is microscopic).
PATTERNS = [
    dict(name="deep5_chain", fire=0.92, bg=0.00004, loss=8.5, depth=5,
         dnf=[[(">", 0, .72), ("<", 1, .28), ("band", 2, .36, .64),
               (">", 3, .72), ("<", 4, .28)]]),
    dict(name="disjunctive", fire=0.88, bg=0.00004, loss=7.5, depth=3,
         dnf=[[(">", 5, .86), (">", 6, .86), ("<", 7, .14)],
              [("<", 8, .14), (">", 9, .86), ("band", 10, .40, .58)]]),
    dict(name="narrow_multiband", fire=0.88, bg=0.00004, loss=7.0, depth=3,
         dnf=[[("band", 11, .33, .47), ("band", 12, .60, .74),
               ("band", 13, .40, .54)]]),
    dict(name="deep6", fire=0.92, bg=0.00003, loss=9.0, depth=6,
         dnf=[[(">", 16, .63), (">", 17, .63), ("<", 18, .37), ("<", 19, .37),
               ("band", 20, .32, .69), (">", 21, .63)]]),
    dict(name="heavy_tailed", fire=0.88, bg=0.00004, loss=9.5, depth=4,
         dnf=[[(">", 22, .77), (">", 23, .77), ("<", 24, .23), (">", 25, .65)]]),
    dict(name="xor_gated", fire=0.92, bg=0.00002, loss=6.0, depth=3,
         dnf=[[(">", 26, .90), (">", 14, .50), ("<", 15, .50)],
              [(">", 26, .90), ("<", 14, .50), (">", 15, .50)]]),
]

SIG = 43  # number of "real" feature slots to generate fully (incl. decoys)


def planted_features(pat) -> set[int]:
    feats = set()
    for conj in pat["dnf"]:
        for cond in conj:
            feats.add(cond[1])
    return feats


def _quantiles(col, ps):
    return {p: float(np.quantile(col, p)) for p in ps}


def _cond_mask(X, cond, qcache):
    if cond[0] == ">":
        _, f, p = cond
        return X[:, f] > qcache[f][p]
    if cond[0] == "<":
        _, f, p = cond
        return X[:, f] < qcache[f][p]
    _, f, lo, hi = cond
    return (X[:, f] > qcache[f][lo]) & (X[:, f] < qcache[f][hi])


def _build_qcache(Xsig):
    qcache = {}
    for pat in PATTERNS:
        for conj in pat["dnf"]:
            for cond in conj:
                f = cond[1]
                ps = qcache.setdefault(f, {})
                for p in cond[2:]:
                    if p not in ps:
                        ps[p] = float(np.quantile(Xsig[:, f], p))
    return qcache


def _gen_signal_block(rng, n):
    """Generate the SIG 'real' features (with heavy tails + decoys)."""
    Xs = rng.standard_normal((n, SIG), dtype=np.float32)
    Xs[:, 22] = rng.lognormal(0.0, 1.0, size=n).astype(np.float32)   # amount
    Xs[:, 23] = rng.exponential(1.0, size=n).astype(np.float32)      # velocity
    # decoys: correlated with a real signal feature but used in no pattern
    Xs[:, 40] = (0.9 * Xs[:, 0] + 0.44 * rng.standard_normal(n, dtype=np.float32))
    Xs[:, 41] = (0.9 * Xs[:, 16] + 0.44 * rng.standard_normal(n, dtype=np.float32))
    return Xs


def _labels_from_patterns(Xs, rng):
    qcache = _build_qcache(Xs)
    n = Xs.shape[0]
    P = len(PATTERNS)
    Y = np.zeros((n, P), dtype=np.int64)
    loss = np.zeros((n, P), dtype=np.float64)
    for c, pat in enumerate(PATTERNS):
        member = np.zeros(n, dtype=bool)
        for conj in pat["dnf"]:
            m = np.ones(n, dtype=bool)
            for cond in conj:
                m &= _cond_mask(Xs, cond, qcache)
            member |= m
        p = np.where(member, pat["fire"], pat["bg"])
        Y[:, c] = (rng.uniform(size=n) < p).astype(np.int64)
        idx = Y[:, c] == 1
        loss[idx, c] = rng.lognormal(pat["loss"], 1.1, size=int(idx.sum()))
    return Y, loss


def make_hard(n=200_000, n_features=200, seed=0):
    """Full float generator for moderate n (returns X, Y, loss, names, planted)."""
    rng = np.random.default_rng(seed)
    X = np.empty((n, n_features), dtype=np.float32)
    X[:, :SIG] = _gen_signal_block(rng, n)
    if n_features > SIG:
        X[:, SIG:] = rng.standard_normal((n, n_features - SIG), dtype=np.float32)
    Y, loss = _labels_from_patterns(X[:, :SIG], rng)
    names = [f"f{i:04d}" for i in range(n_features)]
    planted = {pat["name"]: planted_features(pat) for pat in PATTERNS}
    type_names = [pat["name"] for pat in PATTERNS]
    return X, Y, loss, names, type_names, planted


def make_hard_binned_large(n=2_000_000, n_features=1000, n_bins=10, seed=0,
                           block=50):
    """Memory-safe fused generate->bin for large n. Returns Xbin, spec, Y, ..."""
    from .fast import BinSpec

    rng = np.random.default_rng(seed)
    qs = np.arange(1, n_bins) / n_bins
    Xbin = np.empty((n, n_features), dtype=np.int8)
    edges = [None] * n_features

    def bin_col(j, col):
        e = np.quantile(col, qs)
        edges[j] = e.astype(np.float64)
        Xbin[:, j] = np.searchsorted(e, col, side="right").astype(np.int8)

    Xs = _gen_signal_block(rng, n)            # SIG real features, kept briefly
    Y, loss = _labels_from_patterns(Xs, rng)
    for j in range(SIG):
        bin_col(j, Xs[:, j])
    del Xs

    for start in range(SIG, n_features, block):
        end = min(start + block, n_features)
        Xf = rng.standard_normal((n, end - start), dtype=np.float32)
        for jj in range(end - start):
            bin_col(start + jj, Xf[:, jj])
        del Xf

    names = [f"f{i:04d}" for i in range(n_features)]
    planted = {pat["name"]: planted_features(pat) for pat in PATTERNS}
    type_names = [pat["name"] for pat in PATTERNS]
    return Xbin, BinSpec(edges, qs, n_bins), Y, loss, names, type_names, planted
