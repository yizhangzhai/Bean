"""Scalable miner -- the version that survives 2M x 1000.

Two changes vs. the reference prototype in search.py:

1. **No dense M.** Features are pre-binned once into an int8 matrix ``Xbin``
   (N x F). All scoring is histogram/cumsum over bin indices (``np.bincount``),
   so a depth-1 scan of every threshold of every feature costs O(N*C) per
   feature with NO per-predicate mask materialization.

2. **Subset rescan for conjunctions.** To extend a rule we gather only the rows
   it already matches (its support set) and re-histogram each feature on that
   subset -- exactly recursive partitioning. Cost scales with the (small) rule
   support, not N.

Memory beyond ``Xbin`` (which is ~F bytes/row) is O(N). This is the same
histogram trick that lets LightGBM scale to billions of rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scoring import Score  # noqa: F401  (re-exported convenience)


@dataclass
class BinSpec:
    edges: list[np.ndarray]      # edges[f] = interior quantile cut values
    pct: np.ndarray              # (B-1,) percentile of each edge, e.g. .1,.2,...
    n_bins: int


def fit_bins(X: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, BinSpec]:
    """Bin every feature by its training quantiles -> (Xbin int8, BinSpec).

    Bin index b means edges[b-1] <= x < edges[b]. Threshold split index k in
    0..B-2 corresponds to the cut at percentile (k+1)/n_bins.
    """
    n, f = X.shape
    qs = np.arange(1, n_bins) / n_bins
    Xbin = np.empty((n, f), dtype=np.int8)
    edges: list[np.ndarray] = []
    for j in range(f):
        col = X[:, j]
        e = np.quantile(col, qs)
        edges.append(e.astype(np.float64))
        Xbin[:, j] = np.searchsorted(e, col, side="right").astype(np.int8)
    return Xbin, BinSpec(edges, qs, n_bins)


@dataclass
class FastRule:
    preds: tuple[tuple[int, str, int], ...]   # (feature, op, split_k)
    support: int
    caught: np.ndarray
    lift: np.ndarray
    value: float
    mask: np.ndarray = field(repr=False, default=None)

    def features_ops(self) -> set[tuple[int, str]]:
        return {(f, op) for f, op, _ in self.preds}

    def features(self) -> set[int]:
        return {f for f, _, _ in self.preds}

    def key(self):
        return frozenset(self.preds)

    def label(self, feature_names, spec: BinSpec) -> str:
        parts = []
        for f, op, k in self.preds:
            pct = int(round(spec.pct[k] * 100))
            parts.append(f"{feature_names[f]} {op} p{pct:02d}")
        return "  AND  ".join(parts)


def rule_mask(preds, Xbin) -> np.ndarray:
    m = np.ones(Xbin.shape[0], dtype=bool)
    for f, op, k in preds:
        col = Xbin[:, f]
        m &= (col > k) if op == ">" else (col <= k)
    return m


def _hist_scores(xb, Yw, base, n_bins, min_support, n_for_base):
    """Return candidate (op, k, support, lift[C]) for one feature's bin column.

    xb: int8 bin indices (subset or full). Yw: (len(xb), C) weights.
    """
    B = n_bins
    counts = np.bincount(xb, minlength=B).astype(np.float64)      # per bin
    cc = np.cumsum(counts)                                         # <= k counts
    C = Yw.shape[1]
    wsum = np.empty((B, C))
    for c in range(C):
        wsum[:, c] = np.bincount(xb, weights=Yw[:, c], minlength=B)
    wc = np.cumsum(wsum, axis=0)                                   # <= k weighted
    total_w = wc[-1]
    N = xb.shape[0]
    out = []
    for k in range(B - 1):
        s_below, c_below = cc[k], wc[k]
        s_above, c_above = N - cc[k], total_w - wc[k]
        for op, s, cgt in (("<", s_below, c_below), (">", s_above, c_above)):
            if s >= min_support:
                lift = np.where(base > 0, (cgt / s) / base, 0.0)
                out.append((op, k, int(s), lift))
    return out


def fast_beam_search(
    Xbin: np.ndarray,
    Yw: np.ndarray,
    base: np.ndarray,
    objective,
    spec: BinSpec,
    *,
    beam_width: int = 8,
    max_depth: int = 3,
    min_support: int = 40,
    min_gain: float = 0.10,
) -> list[FastRule]:
    N, F = Xbin.shape
    nb = spec.n_bins

    # ---- depth 1: histogram scan over every feature ----
    cands: list[FastRule] = []
    for f in range(F):
        for op, k, s, lift in _hist_scores(Xbin[:, f], Yw, base, nb, min_support, N):
            cands.append(FastRule(((f, op, k),), s, lift * base, lift,
                                  float(objective(lift))))
    cands.sort(key=lambda r: r.value, reverse=True)
    beam = cands[:beam_width]
    for r in beam:
        r.mask = rule_mask(r.preds, Xbin)
    pool = {r.key(): r for r in beam}

    # ---- grow via subset rescan ----
    for _depth in range(2, max_depth + 1):
        nxt: list[FastRule] = []
        for r in beam:
            idx = np.flatnonzero(r.mask)
            subYw = Yw[idx]
            used = r.features_ops()
            for f in range(F):
                xb = Xbin[idx, f]
                for op, k, s, lift in _hist_scores(xb, subYw, base, nb,
                                                   min_support, N):
                    if (f, op) in used:
                        continue
                    val = float(objective(lift))
                    if val <= r.value * (1.0 + min_gain):
                        continue
                    preds = r.preds + ((f, op, k),)
                    nr = FastRule(preds, s, lift * base, lift, val)
                    nxt.append(nr)
        if not nxt:
            break
        best = {}
        for r in nxt:
            kkey = r.key()
            if kkey not in best or r.value > best[kkey].value:
                best[kkey] = r
        beam = sorted(best.values(), key=lambda r: r.value, reverse=True)[:beam_width]
        for r in beam:
            r.mask = rule_mask(r.preds, Xbin)
            if r.key() not in pool or r.value > pool[r.key()].value:
                pool[r.key()] = r

    return sorted(pool.values(), key=lambda r: r.value, reverse=True)
