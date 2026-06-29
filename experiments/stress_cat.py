"""Categorical SUBSET capture: does target-rank (Fisher trick) encoding let the
fast bin pipeline recover `cat in {non-contiguous codes}` rules that code-bin
(nominal-as-ordinal) cannot?

A/B: same data, same recover_deep -- only the categorical encoding differs.
  code-bin     : category code IS the bin (arbitrary order) -> only contiguous
                 bands, so a non-contiguous risky set is unreachable
  target-rank  : categories re-coded by smoothed fraud rate -> a threshold split
                 selects the highest-rate categories = the optimal subset

Run:  python -m experiments.stress_cat
"""

from __future__ import annotations

import time
import numpy as np

from arp.fast import rule_mask, BinSpec
from arp.encode import target_rank
from featgap import recover_deep
from experiments.stress import peak_gb

N_BINS = 16
MISSING = N_BINS


def make_cat(n, n_features, seed, n_sub=6, n_num=6):
    rng = np.random.default_rng(seed)
    n_cat = 8
    cat_idx = list(range(n_features - n_cat, n_features))
    card = {j: 8 for j in cat_idx}
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    for j in cat_idx:
        X[:, j] = rng.integers(0, card[j], n).astype(np.float32)
    fire = 0.85
    patterns, region = [], []
    # SUBSET patterns: cat in {3 non-contiguous codes} AND two numeric tails
    for _ in range(n_sub):
        cf = int(rng.choice(cat_idx))
        subset = sorted(rng.choice(8, 3, replace=False).tolist())
        nf = rng.choice(range(n_features - n_cat), 2, replace=False)
        m = np.isin(X[:, cf], subset)
        for f in nf:
            m &= X[:, f] > np.quantile(X[:, f], 0.88)
        patterns.append(dict(kind="subset", cat=cf, subset=subset))
        region.append(m)
    # NUMERIC patterns: depth 2-4 conjunctions
    for _ in range(n_num):
        d = int(rng.choice([2, 3, 4]))
        q = (700 / n) ** (1.0 / d)
        feats = rng.choice(range(n_features - n_cat), d, replace=False)
        m = np.ones(n, dtype=bool)
        for f in feats:
            m &= (X[:, f] > np.quantile(X[:, f], 1 - q)) if rng.random() < .5 \
                else (X[:, f] < np.quantile(X[:, f], q))
        patterns.append(dict(kind="numeric"))
        region.append(m)

    order = np.argsort([-r.sum() for r in region])
    mo_id = np.full(n, -1, dtype=np.int64)
    for p in order:
        mo_id[region[p] & (mo_id < 0)] = p
    y = np.zeros(n, dtype=np.int64)
    mo_masks = []
    for p in range(len(region)):
        hit = (mo_id == p) & (rng.random(n) < fire)
        mo_masks.append(hit); y |= hit.astype(np.int64)
    y |= ((mo_id < 0) & (rng.random(n) < 0.001)).astype(np.int64)
    for p, pt in enumerate(patterns):
        pt["realized"] = int(mo_masks[p].sum())
    return X, y, mo_masks, patterns, cat_idx


def encode(Xtr, Xva, cat_idx, ytr, seed, cat_rank):
    F = Xtr.shape[1]
    btr = np.empty((F, Xtr.shape[0]), dtype=np.int8)
    bva = np.empty((F, Xva.shape[0]), dtype=np.int8)
    qs = np.arange(1, N_BINS) / N_BINS
    edges, cat = [None] * F, set(cat_idx)
    rng = np.random.default_rng(seed)
    for f in range(F):
        ct, cv = Xtr[:, f], Xva[:, f]
        if f in cat:
            edges[f] = np.arange(1, N_BINS) - 0.5
            if cat_rank:
                vm = ~np.isnan(ct)
                rk = target_rank(ct[vm].astype(np.int64), ytr[vm], n_codes=8)
                tr_codes = np.where(np.isnan(ct), 0, ct).astype(np.int64)
                va_codes = np.where(np.isnan(cv), 0, cv).astype(np.int64)
                btr[f] = np.where(np.isnan(ct), MISSING, rk[tr_codes]).astype(np.int8)
                bva[f] = np.where(np.isnan(cv), MISSING, rk[va_codes]).astype(np.int8)
            else:
                btr[f] = np.where(np.isnan(ct), MISSING, ct).astype(np.int8)
                bva[f] = np.where(np.isnan(cv), MISSING, cv).astype(np.int8)
        else:
            v = ct[~np.isnan(ct)]
            e = np.quantile(v if v.size < 100_000 else v[rng.integers(0, v.size, 100_000)],
                            qs).astype(np.float64)
            edges[f] = e
            bt = np.searchsorted(e, ct, side="right"); bt[np.isnan(ct)] = MISSING
            bv = np.searchsorted(e, cv, side="right"); bv[np.isnan(cv)] = MISSING
            btr[f], bva[f] = bt.astype(np.int8), bv.astype(np.int8)
    return btr.T, bva.T, BinSpec(edges, qs, N_BINS + 1)


def mine_and_eval(Xtr_b, Xva_b, spec, ytr, yva, mo_va, patterns, cat_idx, tag):
    deep, _ = recover_deep(Xtr_b, Xva_b, spec, ytr, yva,
                           np.zeros(len(ytr), dtype=bool), max_rounds=20, top_k=18,
                           seed_n=250, n_seeds=6, n_jobs=6, min_round_gain=80,
                           target_precision=0.6, min_accept_precision=0.12,
                           max_misses=2, min_recall=0.004, min_support=20,
                           beam_width=64, max_depth=12, categorical=cat_idx, seed=0,
                           verbose=False)
    cov = np.zeros(len(yva), dtype=bool)
    for pr in deep:
        cov |= rule_mask(pr, Xva_b)
    pcov = [(cov & m).sum() / max(1, m.sum()) for m in mo_va]
    sub = [pcov[p] for p, pt in enumerate(patterns) if pt["kind"] == "subset"]
    num = [pcov[p] for p, pt in enumerate(patterns) if pt["kind"] == "numeric"]
    rec = (cov & (yva == 1)).sum() / max(1, (yva == 1).sum())
    print(f"  {tag:16s} rules={len(deep):>3}  recall={rec:.2f}  | "
          f"SUBSET captured {sum(c>=0.5 for c in sub)}/{len(sub)} "
          f"(mean cov {np.mean(sub):.2f})  | NUMERIC {sum(c>=0.5 for c in num)}/{len(num)}")


def run(n=200_000, n_features=60, seed=0):
    print(f"\n{'='*88}\nCATEGORICAL SUBSET A/B  n={n:,}  F={n_features}\n{'='*88}")
    X, y, mo, patterns, cat_idx = make_cat(n, n_features, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n); cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    ytr, yva = y[tr], y[va]
    mo_va = [m[va] for m in mo]
    print(f"  frauds={int(y.sum()):,} ({100*y.mean():.2f}%)  6 subset + 6 numeric "
          f"patterns; subset examples: " +
          ", ".join(f"cat{pt['cat']}∈{pt['subset']}" for pt in patterns
                    if pt["kind"] == "subset")[:120])
    t0 = time.perf_counter()
    for cat_rank, tag in [(False, "code-bin"), (True, "target-rank")]:
        Xtr_b, Xva_b, spec = encode(X[tr], X[va], cat_idx, ytr, seed + 7, cat_rank)
        mine_and_eval(Xtr_b, Xva_b, spec, ytr, yva, mo_va, patterns, cat_idx, tag)
    print(f"\n  [{time.perf_counter()-t0:.0f}s]  peak RSS {peak_gb():.2f} GB")


if __name__ == "__main__":
    run()
