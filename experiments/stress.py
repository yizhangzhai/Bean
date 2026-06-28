"""Robustness stress test: 500K rows, 100 complex patterns, 2% bad rate.

Throws a realistic mess at the framework at once and mines it end to end:

  * 100 patterns, depths 2..15, sizes heavy-tailed (a few common, a long rare tail)
  * features drawn from a SHARED pool -> patterns overlap (no disjoint blocks)
  * condition types: one-sided, two-sided bands, categorical equality
  * feature kinds: numeric, heavy-tailed (lognormal), categorical, correlated decoys
  * missing values (NaN) injected into ~15% of features (incl. signal ones)
  * label noise: fire<1 inside patterns + background fraud
  * a few DISJUNCTIVE (OR) patterns
  * high-dimensional noise (only a fraction of features matter)

Pipeline: encode (missing -> own bin) -> baseline F1 beam -> recover_deep
(LightGBM detect -> restricted refine -> sequential covering). Reports capture
rate stratified by depth / rarity / corner-case, overall recall/precision, time
and peak RSS.

Run:  python -m experiments.stress [n] [n_features] [n_patterns]
"""

from __future__ import annotations

import sys
import time
import platform
import resource

import numpy as np

from arp.fast import rule_mask, BinSpec
from arp.targeted import targeted_beam_search
from featgap import uncovered_positives, recover_deep

N_BINS = 16
MISSING = N_BINS            # dedicated bin code for NaN (codes 0..15 are real)


def peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if platform.system() == "Darwin" else 1e6)


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def make_stress(n, n_features, n_patterns, bad_rate, seed):
    rng = np.random.default_rng(seed)
    n_cat = max(8, n_features // 12)
    cat_idx = list(range(n_features - n_cat, n_features))
    cat_card = {j: int(rng.integers(3, 9)) for j in cat_idx}
    heavy_idx = list(range(4))

    X = rng.standard_normal((n, n_features)).astype(np.float32)
    for j in heavy_idx:
        X[:, j] = rng.lognormal(0.0, 1.0, n).astype(np.float32)
    for j in cat_idx:
        X[:, j] = rng.integers(0, cat_card[j], n).astype(np.float32)
    # correlated decoys: 10,11 shadow 8,9 (tests detector picking the true one)
    X[:, 10] = 0.97 * X[:, 8] + 0.24 * rng.standard_normal(n)
    X[:, 11] = 0.97 * X[:, 9] + 0.24 * rng.standard_normal(n)

    # heavy-tailed pattern sizes (zipf), normalized to the fraud budget
    fire = 0.85
    z = 1.0 / np.arange(1, n_patterns + 1) ** 1.05
    sizes = np.maximum(8, (z / z.sum() * (bad_rate * n / fire)).astype(int))
    depths = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15], n_patterns,
                        p=np.array([.16, .16, .15, .12, .11, .11, .08, .06, .05]))
    disj = set(rng.choice(n_patterns, 5, replace=False))   # 5 OR-patterns

    num_idx = [j for j in range(n_features) if j not in set(cat_idx)]

    def one_clause(d, Np):
        q = (max(Np, 8) / n) ** (1.0 / d)
        feats = rng.choice(n_features, d, replace=False)
        m = np.ones(n, dtype=bool)
        flags = {"cat": False, "band": False}
        for f in feats:
            if f in cat_card:
                c = int(rng.integers(0, cat_card[f]))
                m &= (X[:, f] == c)
                flags["cat"] = True
            else:
                col = X[:, f]
                u = rng.random()
                if u < 0.4:
                    m &= col > np.nanquantile(col, 1 - q)
                elif u < 0.8:
                    m &= col < np.nanquantile(col, q)
                else:
                    lo, hi = 0.5 - q / 2, 0.5 + q / 2
                    m &= (col > np.nanquantile(col, lo)) & (col < np.nanquantile(col, hi))
                    flags["band"] = True
        return m, list(feats), flags

    patterns, region = [], []
    for p in range(n_patterns):
        d, Np = int(depths[p]), int(sizes[p])
        m, feats, flags = one_clause(d, Np)
        if p in disj:                                   # OR with a second clause
            m2, feats2, _ = one_clause(max(2, d - 1), Np)
            m = m | m2
            feats = feats + feats2
        patterns.append(dict(depth=d, size=Np, feats=feats,
                             disj=(p in disj), **flags))
        region.append(m)

    # disjoint by priority (largest patterns first keep their cases)
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
    y |= ((mo_id < 0) & (rng.random(n) < 0.0004)).astype(np.int64)   # background fraud

    # inject missingness AFTER labels (data-quality nuisance, incl. signal feats)
    miss_feats = set(int(j) for j in rng.choice(n_features, int(0.15 * n_features),
                                                replace=False))
    for j in miss_feats:
        X[rng.random(n) < 0.08, j] = np.nan
    for p, pt in enumerate(patterns):
        pt["missing"] = any(f in miss_feats for f in pt["feats"])
        pt["realized"] = int(mo_masks[p].sum())
    return X, y, mo_masks, patterns, cat_idx


# --------------------------------------------------------------------------- #
# encoding: numeric -> quantile bins, categorical -> code, NaN -> MISSING bin
# --------------------------------------------------------------------------- #
def encode(Xtr, Xva, cat_idx, sample, seed):
    F = Xtr.shape[1]
    btr = np.empty((F, Xtr.shape[0]), dtype=np.int8)
    bva = np.empty((F, Xva.shape[0]), dtype=np.int8)
    qs = np.arange(1, N_BINS) / N_BINS
    edges = [None] * F
    cat = set(cat_idx)
    rng = np.random.default_rng(seed)
    for f in range(F):
        ct, cv = Xtr[:, f], Xva[:, f]
        if f in cat:
            edges[f] = np.arange(1, N_BINS) - 0.5
            btr[f] = np.where(np.isnan(ct), MISSING, ct).astype(np.int8)
            bva[f] = np.where(np.isnan(cv), MISSING, cv).astype(np.int8)
        else:
            v = ct[~np.isnan(ct)]
            if v.size > sample:
                v = v[rng.integers(0, v.size, sample)]
            e = np.quantile(v, qs).astype(np.float64)
            edges[f] = e
            bt = np.searchsorted(e, ct, side="right")
            bt[np.isnan(ct)] = MISSING
            bv = np.searchsorted(e, cv, side="right")
            bv[np.isnan(cv)] = MISSING
            btr[f], bva[f] = bt.astype(np.int8), bv.astype(np.int8)
    return btr.T, bva.T, BinSpec(edges, qs, N_BINS + 1)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def cover(rule_preds, Xb):
    c = np.zeros(Xb.shape[0], dtype=bool)
    for preds in rule_preds:
        c |= rule_mask(preds, Xb)
    return c


def report(tag, rule_preds, Xva, yva, mo_va):
    cov = cover(rule_preds, Xva)
    pos = yva == 1
    tp = int((cov & pos).sum())
    rec = tp / max(1, int(pos.sum()))
    prec = tp / max(1, int(cov.sum()))
    pcov = np.array([(cov & m).sum() / max(1, m.sum()) for m in mo_va])
    print(f"  {tag:24s} recall={rec:.2f}  precision={prec:.2f}  "
          f"rules={len(rule_preds)}  flagged={int(cov.sum()):,}")
    return rec, prec, pcov


def strata(patterns, pcov, captured_thr=0.5):
    cap = pcov >= captured_thr
    tot_fr = sum(p["realized"] for p in patterns)
    cap_fr = sum(p["realized"] for p, c in zip(patterns, cap) if c)
    print(f"\n  CAPTURED {int(cap.sum())}/{len(patterns)} patterns "
          f"(coverage>=50%), accounting for {cap_fr/max(1,tot_fr):.0%} of all fraud")

    def bucket(key, buckets, label):
        print(f"\n  capture rate by {label}:")
        for name, lo, hi in buckets:
            ids = [i for i, p in enumerate(patterns) if lo <= key(p) < hi]
            if not ids:
                continue
            nc = sum(cap[i] for i in ids)
            fr = sum(patterns[i]["realized"] for i in ids)
            cfr = sum(patterns[i]["realized"] for i in ids if cap[i])
            print(f"    {name:14s} {nc:>3}/{len(ids):<3} patterns  "
                  f"({cfr/max(1,fr):>4.0%} of their fraud)")

    bucket(lambda p: p["realized"], [("<50", 0, 50), ("50-150", 50, 150),
           ("150-400", 150, 400), ("400-1500", 400, 1500), (">=1500", 1500, 9e9)],
           "realized size (cases)")
    bucket(lambda p: p["depth"], [("depth 2-4", 2, 5), ("depth 5-8", 5, 9),
           ("depth 9-15", 9, 16)], "depth")

    print("\n  capture rate by corner case (among patterns big enough to mine, >=150):")
    big = [i for i, p in enumerate(patterns) if p["realized"] >= 150]
    for name, key in [("categorical", "cat"), ("two-sided band", "band"),
                      ("missing feats", "missing"), ("disjunctive OR", "disj")]:
        ids = [i for i in big if patterns[i][key]]
        if ids:
            nc = sum(cap[i] for i in ids)
            print(f"    {name:16s} {nc:>3}/{len(ids):<3} captured")


# --------------------------------------------------------------------------- #
def run(n=500_000, n_features=200, n_patterns=100, bad_rate=0.02, seed=0):
    t_all = time.perf_counter()
    print(f"\n{'='*92}\nSTRESS TEST  n={n:,}  F={n_features}  patterns={n_patterns}  "
          f"bad_rate={bad_rate:.0%}\n{'='*92}")
    t0 = time.perf_counter()
    X, y, mo_masks, patterns, cat_idx = make_stress(n, n_features, n_patterns,
                                                    bad_rate, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Xtr_b, Xva_b, spec = encode(X[tr], X[va], cat_idx, sample=100_000, seed=seed + 7)
    ytr, yva = y[tr], y[va]
    mo_va = [m[va] for m in mo_masks]
    realized = sum(p["realized"] for p in patterns)
    minable = sum(1 for p in patterns if p["realized"] >= 150)
    print(f"  frauds={int(y.sum()):,} ({100*y.mean():.2f}%)  realized pattern "
          f"cases={realized:,}  patterns with >=150 cases: {minable}/{n_patterns}")
    print(f"  missing values: {int(np.isnan(X).sum()):,}  categorical feats: "
          f"{len(cat_idx)}  [gen+encode {time.perf_counter()-t0:.0f}s]")

    min_support = 30
    # ---- baseline F1 beam ----
    t0 = time.perf_counter()
    base, _ = targeted_beam_search(
        Xtr_b, ytr.reshape(-1, 1), 0, spec, min_recall=0.004, target_precision=0.4,
        min_support=min_support, beam_width=40, max_depth=12,
        Xbin_val=Xva_b, Y_val=yva.reshape(-1, 1), gap_tol=None)
    base_preds = [r.preds for r in base]
    print(f"\n  [baseline F1 beam {time.perf_counter()-t0:.0f}s]")
    rec0, prec0, pcov0 = report("0. baseline", base_preds, Xva_b, yva, mo_va)

    # ---- recover_deep as PURE sequential covering (from scratch) so isolation
    #      handles shallow patterns too, not just the baseline's residual ----
    t0 = time.perf_counter()
    empty = np.zeros(len(ytr), dtype=bool)
    deep, infos = recover_deep(Xtr_b, Xva_b, spec, ytr, yva, empty, max_rounds=30,
                               top_k=22, seed_n=250, n_seeds=8, n_jobs=6,
                               target_precision=0.6, min_accept_precision=0.12,
                               max_misses=4, min_recall=0.004, min_support=20,
                               beam_width=64, max_depth=18, seed=seed, verbose=True)
    print(f"  [recover_deep(scratch) {time.perf_counter()-t0:.0f}s]  "
          f"{len(deep)} rules over {len(infos)} captured rounds")
    rec1, prec1, pcov1 = report("1. sequential covering", deep, Xva_b, yva, mo_va)

    strata(patterns, np.maximum(pcov0, pcov1))
    print(f"\n  TOTAL {time.perf_counter()-t_all:.0f}s   peak RSS {peak_gb():.2f} GB")


if __name__ == "__main__":
    a = sys.argv[1:]
    run(n=int(a[0]) if len(a) > 0 else 500_000,
        n_features=int(a[1]) if len(a) > 1 else 200,
        n_patterns=int(a[2]) if len(a) > 2 else 100)
