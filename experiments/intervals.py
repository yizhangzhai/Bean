"""Can it find value INTERVALS like 1 < A < 10?

Features are Uniform(0, 20), so a value interval maps exactly to a percentile
band (e.g. 1 < A < 10  ==  p05 < A < p50). With n_bins=20 the planted bounds
fall on bin edges, so recovery -- translated back to ACTUAL VALUES via
spec.edges -- should reproduce the planted interval.

Patterns (defined in real units):
  interval_A      : 1 < f0 < 10                       (a pure one-feature band)
  interval_plus   : 5 < f1 < 8  AND  f2 > 15
  double_interval : 2 < f3 < 6  AND  12 < f4 < 18
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np

from arp.fast import fit_bins
from arp.targeted import targeted_beam_search

RANGE = 20.0
PATTERNS = [
    dict(name="interval_A", fire=.80, bg=.004,
         conds=[("int", 0, 1.0, 10.0)]),
    dict(name="interval_plus", fire=.88, bg=.002,
         conds=[("int", 1, 5.0, 8.0), ("gt", 2, 15.0)]),
    dict(name="double_interval", fire=.88, bg=.002,
         conds=[("int", 3, 2.0, 6.0), ("int", 4, 12.0, 18.0)]),
]


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1e9 if sys.platform == "darwin" else 1e6)


def make(n, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, RANGE, size=(n, n_features)).astype(np.float32)
    Y = np.zeros((n, len(PATTERNS)), dtype=np.int64)
    for c, pat in enumerate(PATTERNS):
        m = np.ones(n, dtype=bool)
        for cond in pat["conds"]:
            if cond[0] == "int":
                _, f, lo, hi = cond
                m &= (X[:, f] > lo) & (X[:, f] < hi)
            else:
                _, f, v = cond
                m &= X[:, f] > v
        p = np.where(m, pat["fire"], pat["bg"])
        Y[:, c] = (rng.uniform(size=n) < p).astype(np.int64)
    return X, Y


def feature_bounds(rule, f, spec):
    """Translate a rule's predicates on feature f back to (lo, hi) in real units."""
    lo, hi = -np.inf, np.inf
    for pf, op, k in rule.preds:
        if pf == f:
            v = float(spec.edges[f][k])
            if op == ">":
                lo = max(lo, v)
            else:
                hi = min(hi, v)
    return lo, hi


def planted_intervals(pat):
    out = {}
    for cond in pat["conds"]:
        if cond[0] == "int":
            out[cond[1]] = (cond[2], cond[3])
        else:
            out[cond[1]] = (cond[2], np.inf)
    return out


def run(n=200_000, n_features=40, n_bins=20, seed=0, tol=1.0, target_precision=0.75):
    t0 = time.perf_counter()
    X, Y = make(n, n_features, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Xtr, spec = fit_bins(X[tr], n_bins=n_bins)
    Xva, _ = fit_bins(X[va], n_bins=n_bins)
    names = [f"f{i:02d}" for i in range(n_features)]
    t_prep = time.perf_counter() - t0

    print(f"\n{'='*100}\nVALUE-INTERVAL RECOVERY  n={n:,} F={n_features} "
          f"bins={n_bins}  (features ~ Uniform(0,{RANGE:.0f}))\n{'='*100}")
    print(f"prep={t_prep:.1f}s  base rates={np.round(Y.mean(0)*100,3)}%  "
          f"target_precision={target_precision} (high enough to force BOTH "
          f"interval bounds)\n")

    t0 = time.perf_counter()
    npass = 0
    for c, pat in enumerate(PATTERNS):
        rules, _ = targeted_beam_search(
            Xtr, Y[tr], c, spec, min_recall=0.15, target_precision=target_precision,
            min_support=40, beam_width=16, max_depth=4,
            Xbin_val=Xva, Y_val=Y[va], gap_tol=0.25)
        plant = planted_intervals(pat)
        print(f"--- {pat['name']} ---")
        if not rules:
            print("    MISS (no rule met targets)\n")
            continue
        best = max(rules, key=lambda r: r.val_precision * r.val_recall)
        ok = True
        for f, (lo, hi) in plant.items():
            rlo, rhi = feature_bounds(best, f, spec)
            lo_ok = abs(rlo - lo) <= tol if np.isfinite(lo) else not np.isfinite(rlo)
            hi_ok = abs(rhi - hi) <= tol if np.isfinite(hi) else not np.isfinite(rhi)
            ok = ok and lo_ok and hi_ok
            want = (f"{lo:.1f} < {names[f]} < {hi:.1f}" if np.isfinite(hi)
                    else f"{names[f]} > {lo:.1f}")
            got = (f"{rlo:.2f} < {names[f]} < {rhi:.2f}" if np.isfinite(rhi)
                   else f"{names[f]} > {rlo:.2f}")
            print(f"    planted:  {want:28s}  recovered:  {got}")
        print(f"    valP={best.val_precision:.2f} valR={best.val_recall:.2f}  "
              f"-> {'OK' if ok else 'OFF'}\n")
        npass += ok
    t_mine = time.perf_counter() - t0
    print("-" * 100)
    print(f"recovered {npass}/{len(PATTERNS)} interval patterns   mine={t_mine:.1f}s   "
          f"peak RSS={peak_gb():.2f}GB")


if __name__ == "__main__":
    run()
