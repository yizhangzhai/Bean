"""Multi-arm parallel covering: partition the frauds, cover each arm independently.

The serial limiter in sequential covering is the cross-pattern dependency (remove
one pattern so the next surfaces). Break it by PARTITIONING the frauds upfront into
K pattern-aware groups -- each arm then covers its own group, independently and in
parallel. Bonus: each arm sees ~1/K of the patterns, so its residual is far less of
a "soup", which is exactly what defeated isolation at 100 patterns.

  one detector fit -> KMeans on top features -> K fraud clusters (arms)
  -> recover_deep per arm (independent) -> merge + dedup

Arms are run sequentially here but are independent, so the parallel wall-clock is
~max(arm time); we report both. The question under test: does partitioning keep (or
raise) capture vs the single-stream 0.44 baseline?

Run:  python -m experiments.stress_arms [n] [n_features] [n_patterns] [K]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from featgap import recover_deep
from experiments.stress import make_stress, encode, report, strata, peak_gb


def partition(Xtr_b, ytr, K, seed, top_m=60, sub_n=80_000):
    """Pattern-aware fraud partition: one detector fit -> top features -> KMeans
    on the fraud rows. Returns (fraud_idx, top_features, kmeans)."""
    import lightgbm as lgb
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(ytr), min(sub_n, len(ytr)), replace=False)
    det = lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.08,
                             class_weight="balanced", n_jobs=-1, random_state=seed,
                             importance_type="gain", verbosity=-1)
    det.fit(np.asarray(Xtr_b[sub]).astype(np.float32), ytr[sub])
    top = np.argsort(det.feature_importances_)[::-1][:top_m]
    fr = np.flatnonzero(ytr == 1)
    Xf = np.asarray(Xtr_b[fr][:, top]).astype(np.float32)
    km = KMeans(K, n_init=4, random_state=seed).fit(Xf)
    return fr, top, km


def run(n=500_000, n_features=200, n_patterns=100, K=6, seed=0):
    t_all = time.perf_counter()
    print(f"\n{'='*92}\nMULTI-ARM COVERING  n={n:,}  F={n_features}  "
          f"patterns={n_patterns}  arms={K}\n{'='*92}")
    X, y, mo, patterns, cat_idx = make_stress(n, n_features, n_patterns, 0.02, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Xtr_b, Xva_b, spec = encode(X[tr], X[va], cat_idx, 100_000, seed + 7)
    ytr, yva = y[tr], y[va]
    mo_va = [m[va] for m in mo]
    print(f"  frauds={int(y.sum()):,} ({100*y.mean():.2f}%)  "
          f"patterns >=150 cases: {sum(1 for p in patterns if p['realized']>=150)}")

    # ---- partition ----
    t0 = time.perf_counter()
    fr_tr, top, km = partition(Xtr_b, ytr, K, seed)
    cl_tr = km.predict(np.asarray(Xtr_b[fr_tr][:, top]).astype(np.float32))
    fr_va = np.flatnonzero(yva == 1)
    cl_va = km.predict(np.asarray(Xva_b[fr_va][:, top]).astype(np.float32))
    t_part = time.perf_counter() - t0
    sizes = [int((cl_tr == k).sum()) for k in range(K)]
    print(f"  [partition {t_part:.0f}s]  arm fraud sizes: {sizes}")

    # ---- per-arm covering (independent -> parallelizable) ----
    all_preds, arm_times, arm_rules = [], [], []
    empty = np.zeros(len(ytr), dtype=bool)
    for k in range(K):
        ytr_k = np.zeros(len(ytr), dtype=np.int64); ytr_k[fr_tr[cl_tr == k]] = 1
        yva_k = np.zeros(len(yva), dtype=np.int64); yva_k[fr_va[cl_va == k]] = 1
        t0 = time.perf_counter()
        deep, _ = recover_deep(
            Xtr_b, Xva_b, spec, ytr_k, yva_k, empty, max_rounds=60, top_k=22,
            seed_n=250, target_precision=0.6, min_accept_precision=0.12,
            max_misses=10, min_recall=0.01, min_support=20, beam_width=64,
            max_depth=18, seed=seed, verbose=False)
        dt = time.perf_counter() - t0
        arm_times.append(dt); arm_rules.append(len(deep)); all_preds += deep
        print(f"  arm {k}: {sizes[k]:>5} frauds -> {len(deep):>2} rules  [{dt:.0f}s]")

    # ---- merge + dedup ----
    seen, merged = set(), []
    for p in all_preds:
        key = frozenset(p)
        if key not in seen:
            seen.add(key); merged.append(p)

    print(f"\n  merged {len(all_preds)} -> {len(merged)} rules after dedup")
    rec, prec, pcov = report("multi-arm", merged, Xva_b, yva, mo_va)
    strata(patterns, pcov)

    seq = sum(arm_times)
    par = t_part + max(arm_times)
    print(f"\n  TIME: partition {t_part:.0f}s + arms(sequential) {seq:.0f}s = "
          f"{t_part+seq:.0f}s total")
    print(f"        parallel estimate (max arm + partition): {par:.0f}s "
          f"({K} arms)   [baseline single-stream ~853s @ recall 0.44]")
    print(f"  peak RSS {peak_gb():.2f} GB   wall {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    run(n=int(a[0]) if len(a) > 0 else 500_000,
        n_features=int(a[1]) if len(a) > 1 else 200,
        n_patterns=int(a[2]) if len(a) > 2 else 100,
        K=int(a[3]) if len(a) > 3 else 6)
