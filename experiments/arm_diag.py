"""Fragmentation diagnostic: do the K arms split mineable patterns, or keep each
pattern mostly in one arm? If concentration is high, capture loss is from arm
settings; if low, the partition is fragmenting patterns (-> soft assignment)."""

from __future__ import annotations

import sys
import numpy as np

from experiments.stress import make_stress, encode
from experiments.stress_arms import partition


def run(n=500_000, n_features=200, n_patterns=100, K=6, seed=0, method="leaf"):
    X, y, mo, patterns, cat_idx = make_stress(n, n_features, n_patterns, 0.02, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    tr = perm[:int(n * 0.67)]
    va = perm[int(n * 0.67):]
    Xtr_b, Xva_b, spec = encode(X[tr], X[va], cat_idx, 100_000, seed + 7)
    ytr = y[tr]
    fr_tr, assign = partition(Xtr_b, ytr, K, seed, method=method)
    cl_tr = assign(Xtr_b, fr_tr)
    cluster_of = np.full(len(tr), -1, dtype=np.int64)
    cluster_of[fr_tr] = cl_tr

    print(f"\nFRAGMENTATION DIAGNOSTIC  (K={K} arms)")
    print(f"  for each mineable pattern: how its frauds spread across arms\n")
    print(f"  {'pat':>4} {'depth':>5} {'size':>5} | {'concentration':>13} {'arms>=15%':>9} | "
          f"per-arm counts")
    concs = []
    for p, pt in enumerate(patterns):
        if pt["realized"] < 150:
            continue
        members = np.flatnonzero(mo[p][tr])
        cls = cluster_of[members]
        cls = cls[cls >= 0]
        if len(cls) == 0:
            continue
        bc = np.bincount(cls, minlength=K)
        conc = bc.max() / len(cls)
        concs.append(conc)
        arms_used = int((bc >= 0.15 * len(cls)).sum())
        print(f"  {p:>4} {pt['depth']:>5} {pt['realized']:>5} | "
              f"{conc:>12.0%} {arms_used:>9} | {list(bc)}")
    concs = np.array(concs)
    print(f"\n  median dominant-cluster concentration: {np.median(concs):.0%}")
    print(f"  patterns with concentration < 60% (fragmented): "
          f"{int((concs < 0.6).sum())}/{len(concs)}")
    verdict = ("FRAGMENTED -> fix the partition (soft/overlapping assignment)"
               if np.median(concs) < 0.7 else
               "CONCENTRATED -> partition is fine; capture loss is from arm settings")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    a = sys.argv[1:]
    run(K=int(a[0]) if len(a) > 0 else 6,
        method=a[1] if len(a) > 1 else "leaf")
