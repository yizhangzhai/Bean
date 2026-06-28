"""Genericity test: a STRUCTURALLY DIFFERENT data generator, run through the
EXACT same recover_deep hyperparameters as stress.py (no re-tuning). If capture
holds, the design is generic; if it needs new knobs, it was overfit to make_stress.

Deliberate differences from make_stress:
  * feature distributions: uniform / exponential / bimodal-Gaussian / Poisson-counts
    / normal  (make_stress was mostly normal + a little lognormal)
  * overlap structure: SHARED-ANCHOR hierarchies -- ~40% of patterns build on one of
    a few "fraud-prone" anchor features  (make_stress used a flat random pool)
  * pattern sizes: BIMODAL -- a few big campaigns + a band of medium + a rare tail
    (make_stress used a single zipf)
  * condition mix: high-tail / low-tail / wide-band / categorical, different ratios
  * bad rate 3% (vs 2%), 70 patterns, 400K x 180, seed offset -> fresh data
  * heavier background fraud (0.1%) and 10% missingness

Run:  python -m experiments.stress2 [n] [n_features] [n_patterns]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from featgap import recover_deep
from experiments.stress import encode, report, strata, peak_gb


def make_stress2(n, n_features, n_patterns, bad_rate, seed):
    rng = np.random.default_rng(seed)
    n_cat = max(10, n_features // 10)
    cat_idx = list(range(n_features - n_cat, n_features))
    cat_card = {j: int(rng.integers(3, 10)) for j in cat_idx}
    F_num = n_features - n_cat

    X = np.empty((n, n_features), dtype=np.float32)
    for j in range(F_num):                              # diverse distributions
        r = j % 5
        if r == 0:
            X[:, j] = rng.uniform(0, 1, n)
        elif r == 1:
            X[:, j] = rng.exponential(1.0, n)
        elif r == 2:
            comp = rng.random(n) < 0.5
            X[:, j] = np.where(comp, rng.normal(-2, 1, n), rng.normal(2.5, 1.2, n))
        elif r == 3:
            X[:, j] = rng.poisson(4, n).astype(np.float32)
        else:
            X[:, j] = rng.standard_normal(n)
    for j in cat_idx:
        X[:, j] = rng.integers(0, cat_card[j], n).astype(np.float32)
    X[:, 7] = 0.95 * X[:, 0] + 0.30 * rng.standard_normal(n)   # correlated decoys
    X[:, 8] = 0.95 * X[:, 1] + 0.30 * rng.standard_normal(n)

    anchors = list(rng.choice(F_num, 5, replace=False))
    fire = 0.8
    # bimodal sizes: campaigns + medium + tail, scaled to the fraud budget
    sizes = np.concatenate([
        rng.integers(1000, 3000, max(1, n_patterns // 12)),
        rng.integers(150, 550, max(1, n_patterns // 3)),
        rng.integers(15, 120, n_patterns)])[:n_patterns]
    rng.shuffle(sizes)
    sizes = np.maximum(10, (sizes / sizes.sum() * bad_rate * n / fire)).astype(int)
    depths = rng.choice([2, 3, 4, 5, 6, 7, 9], n_patterns,
                        p=[.20, .20, .18, .15, .12, .09, .06])

    patterns, region = [], []
    for p in range(n_patterns):
        d, Np = int(depths[p]), int(sizes[p])
        q = (max(Np, 10) / n) ** (1.0 / d)
        feats = []
        if rng.random() < 0.4:                          # shared-anchor overlap
            feats.append(int(rng.choice(anchors)))
        while len(feats) < d:
            f = int(rng.integers(0, n_features))
            if f not in feats:
                feats.append(f)
        m = np.ones(n, dtype=bool)
        flags = {"cat": False, "band": False}
        for f in feats:
            if f in cat_card:
                m &= (X[:, f] == int(rng.integers(0, cat_card[f])))
                flags["cat"] = True
            else:
                col = X[:, f]
                u = rng.random()
                if u < 0.35:
                    m &= col > np.quantile(col, 1 - q)
                elif u < 0.60:
                    m &= col < np.quantile(col, q)
                else:
                    lo, hi = 0.5 - q / 2, 0.5 + q / 2
                    m &= (col > np.quantile(col, lo)) & (col < np.quantile(col, hi))
                    flags["band"] = True
        patterns.append(dict(depth=d, size=Np, feats=feats, disj=False, **flags))
        region.append(m)

    order = np.argsort([-pt["size"] for pt in patterns])
    mo_id = np.full(n, -1, dtype=np.int64)
    for p in order:
        mo_id[region[p] & (mo_id < 0)] = p
    y = np.zeros(n, dtype=np.int64)
    mo_masks = []
    for p in range(n_patterns):
        hit = (mo_id == p) & (rng.random(n) < fire)
        mo_masks.append(hit)
        y |= hit.astype(np.int64)
    y |= ((mo_id < 0) & (rng.random(n) < 0.001)).astype(np.int64)   # background

    miss = set(int(j) for j in rng.choice(n_features, int(0.15 * n_features), replace=False))
    for j in miss:
        X[rng.random(n) < 0.10, j] = np.nan
    for p, pt in enumerate(patterns):
        pt["missing"] = any(f in miss for f in pt["feats"])
        pt["realized"] = int(mo_masks[p].sum())
    return X, y, mo_masks, patterns, cat_idx


def run(n=400_000, n_features=180, n_patterns=70, bad_rate=0.03, seed=1):
    t_all = time.perf_counter()
    print(f"\n{'='*92}\nGENERICITY TEST (new generator)  n={n:,}  F={n_features}  "
          f"patterns={n_patterns}  bad_rate={bad_rate:.0%}\n{'='*92}")
    X, y, mo, patterns, cat_idx = make_stress2(n, n_features, n_patterns, bad_rate, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Xtr_b, Xva_b, spec = encode(X[tr], X[va], cat_idx, 100_000, seed + 7)
    ytr, yva = y[tr], y[va]
    mo_va = [m[va] for m in mo]
    minable = sum(1 for p in patterns if p["realized"] >= 150)
    print(f"  frauds={int(y.sum()):,} ({100*y.mean():.2f}%)  missing="
          f"{int(np.isnan(X).sum()):,}  cat feats={len(cat_idx)}  "
          f"patterns >=150 cases: {minable}/{n_patterns}")

    # IDENTICAL recover_deep config to stress.py -- no re-tuning
    empty = np.zeros(len(ytr), dtype=bool)
    t0 = time.perf_counter()
    deep, _ = recover_deep(Xtr_b, Xva_b, spec, ytr, yva, empty, max_rounds=30,
                           top_k=22, seed_n=250, n_seeds=8, n_jobs=6,
                           min_round_gain=120, target_precision=0.6,
                           min_accept_precision=0.12, max_misses=2,
                           min_recall=0.004, min_support=20, beam_width=64,
                           max_depth=18, seed=seed, verbose=True)
    print(f"  [recover_deep {time.perf_counter()-t0:.0f}s]  {len(deep)} rules")
    rec, prec, pcov = report("recover_deep", deep, Xva_b, yva, mo_va)
    strata(patterns, pcov)
    print(f"\n  TOTAL {time.perf_counter()-t_all:.0f}s   peak RSS {peak_gb():.2f} GB")


if __name__ == "__main__":
    a = sys.argv[1:]
    run(n=int(a[0]) if len(a) > 0 else 400_000,
        n_features=int(a[1]) if len(a) > 1 else 180,
        n_patterns=int(a[2]) if len(a) > 2 else 70)
