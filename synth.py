"""Synthetic labeled data -- one configurable generator for the whole project.

Consolidates the scattered experiment generators into a single function. A positive
label is fired by N hidden "modus operandi" (patterns); each pattern is a hidden
rule we can check the miner against. Supports the full mess used in the stress
tests: axis conjunctions of varying depth, categorical equality/subset, two-sided
bands, heavy-tailed feature distributions, correlated decoys, missing values,
heavy-tailed pattern sizes, feature overlap, and (optionally) NON-AXIS patterns
(ring / ratio / periodic) that need feature engineering.

    fs = make_data(n=200_000, n_features=120, n_patterns=40, bad_rate=0.02)
    fs.X, fs.y, fs.categorical, fs.patterns
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GEO_C = (20.0, 20.0)


@dataclass
class DataSet:
    X: np.ndarray            # (n, F) float32, may contain NaN
    y: np.ndarray            # (n,) int64 in {0,1}
    categorical: list        # categorical feature indices
    patterns: list           # per-pattern dicts: kind/depth/realized/mask/...
    names: list              # feature names

    @property
    def mineable(self):      # patterns with enough cases to be statistically findable
        return [p for p in self.patterns if p["realized"] >= 150]


def make_data(n=200_000, n_features=120, n_patterns=40, bad_rate=0.02, *,
               nonaxis=False, missing=0.10, fire=0.85, seed=0) -> DataSet:
    rng = np.random.default_rng(seed)
    n_cat = max(6, n_features // 12)
    cat_idx = list(range(n_features - n_cat, n_features))
    cat_card = {j: int(rng.integers(4, 9)) for j in cat_idx}
    F_num = n_features - n_cat

    # ---- features: diverse distributions ----
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    for j in range(min(4, F_num)):
        X[:, j] = rng.lognormal(0.0, 1.0, n).astype(np.float32)      # heavy tail
    for j in cat_idx:
        X[:, j] = rng.integers(0, cat_card[j], n).astype(np.float32)
    if F_num > 9:                                                    # correlated decoys
        X[:, 8] = 0.95 * X[:, 6] + 0.30 * rng.standard_normal(n)
        X[:, 9] = 0.95 * X[:, 7] + 0.30 * rng.standard_normal(n)

    patterns, region = [], []

    # ---- optional non-axis patterns on reserved columns 0..4 ----
    if nonaxis and F_num >= 5:
        X[:, 0] = rng.uniform(0, 40, n); X[:, 1] = rng.uniform(0, 40, n)
        X[:, 2] = rng.uniform(1, 100, n); X[:, 3] = rng.uniform(10, 100, n)
        X[:, 4] = rng.uniform(0, 1000, n)
        rr = np.hypot(X[:, 0] - GEO_C[0], X[:, 1] - GEO_C[1])
        ratio, tmod = X[:, 2] / X[:, 3], np.mod(X[:, 4], 24)
        patterns.append(dict(kind="ring")); region.append((rr > 4.5) & (rr < 5.5))
        patterns.append(dict(kind="ratio")); region.append(ratio > np.quantile(ratio, .98))
        patterns.append(dict(kind="periodic")); region.append((tmod >= 2) & (tmod <= 2.5))
        pool = list(range(5, F_num)) + cat_idx
    else:
        pool = list(range(F_num)) + cat_idx

    # ---- axis / categorical patterns ----
    n_axis = n_patterns - len(patterns)
    sizes = rng.integers(150, 2500, n_axis).astype(float)
    sizes = np.maximum(20, sizes / sizes.sum() * bad_rate * n / fire).astype(int)
    depths = rng.choice([2, 3, 4, 5, 6, 8, 11], n_axis,
                        p=[.22, .20, .17, .13, .12, .09, .07])
    for p in range(n_axis):
        d, Np = int(depths[p]), int(sizes[p])
        q = (max(Np, 20) / n) ** (1.0 / d)
        feats = rng.choice(pool, d, replace=False)
        m = np.ones(n, dtype=bool)
        flags = {"cat": False, "band": False}
        for f in feats:
            if f in cat_card:                                       # categorical
                k = int(rng.integers(1, 3))                          # subset of 1-2 codes
                m &= np.isin(X[:, f], rng.choice(cat_card[f], k, replace=False))
                flags["cat"] = True
            else:
                col = X[:, f]
                u = rng.random()
                if u < 0.4:
                    m &= col > np.quantile(col, 1 - q)
                elif u < 0.75:
                    m &= col < np.quantile(col, q)
                else:
                    lo, hi = 0.5 - q / 2, 0.5 + q / 2
                    m &= (col > np.quantile(col, lo)) & (col < np.quantile(col, hi))
                    flags["band"] = True
        patterns.append(dict(kind="axis", depth=d, size=Np, feats=list(feats), **flags))
        region.append(m)

    # ---- disjoint by priority (largest first), fire, background noise ----
    order = np.argsort([-int(r.sum()) for r in region])
    mo_id = np.full(n, -1, dtype=np.int64)
    for p in order:
        mo_id[region[p] & (mo_id < 0)] = p
    y = np.zeros(n, dtype=np.int64)
    for p in range(len(region)):
        hit = (mo_id == p) & (rng.random(n) < fire)
        patterns[p]["mask"] = hit
        patterns[p]["realized"] = int(hit.sum())
        y |= hit.astype(np.int64)
    y |= ((mo_id < 0) & (rng.random(n) < 0.0008)).astype(np.int64)   # background positives

    # ---- inject missingness (numeric cols only, never the non-axis 0..4) ----
    if missing > 0:
        lo = 5 if nonaxis else 0
        cols = rng.choice(range(lo, n_features), int(0.15 * (n_features - lo)), replace=False)
        for j in cols:
            X[rng.random(n) < missing * 0.5, j] = np.nan

    names = [f"f{i:03d}" for i in range(n_features)]
    return DataSet(X, y, cat_idx, patterns, names)
