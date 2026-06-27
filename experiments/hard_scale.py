"""Time + space of mining the HARD patterns as data grows.

Uses the memory-safe fused generate->bin generator and the recall-aware
targeted search (the path that actually recovers deep patterns). Reports
per-stage time, peak RSS, and how many of the 6 patterns are still recovered.

    python experiments/hard_scale.py --n 1000000 --features 500
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np

from arp.hard_data import (make_hard_binned_large, PATTERNS, planted_features)
from arp.targeted import targeted_beam_search

DECOYS = {40, 41}


def peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if sys.platform == "darwin" else 1e6)


def all_signal():
    s = set()
    for p in PATTERNS:
        s |= planted_features(p)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--features", type=int, default=500)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    N, F = a.n, a.features
    SIG = all_signal()

    print(f"\n{'='*78}\nHARD-PATTERN SCALE  n={N:,} F={F} bins={a.bins} "
          f"depth<={a.depth} beam={a.beam}\n{'='*78}")

    t = time.perf_counter()
    Xbin, spec, Y, loss, names, tnames, planted = make_hard_binned_large(
        n=N, n_features=F, n_bins=a.bins, seed=a.seed)
    t_gb = time.perf_counter() - t
    print(f"gen+bin   {t_gb:7.2f}s   Xbin={Xbin.nbytes/1e9:.2f}GB int8   "
          f"base rates={np.round(Y.mean(0)*100,3)}%")

    # 67/33 split on rows
    rng = np.random.default_rng(a.seed + 1)
    perm = rng.permutation(N)
    cut = int(N * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Xtr, Xva, Ytr, Yva = Xbin[tr], Xbin[va], Y[tr], Y[va]

    t = time.perf_counter()
    n_full = n_part = 0
    rows = []
    for c, pat in enumerate(PATTERNS):
        rules, trace = targeted_beam_search(
            Xtr, Ytr, c, spec, min_recall=0.25, target_precision=0.5,
            min_support=40, beam_width=a.beam, max_depth=a.depth,
            Xbin_val=Xva, Y_val=Yva, gap_tol=0.20)
        branches = [frozenset(cond[1] for cond in conj) for conj in pat["dnf"]]

        def covers(bf):
            for r in rules:
                rf = {f for f, _, _ in r.preds}
                if bf <= rf and not (rf - SIG):
                    return True
            return False
        ncov = sum(covers(b) for b in branches)
        status = "OK" if ncov == len(branches) else ("PARTIAL" if ncov else "WEAK")
        n_full += status == "OK"
        n_part += status == "PARTIAL"
        rows.append((pat["name"], status, ncov, len(branches), len(rules)))
    t_mine = time.perf_counter() - t

    print(f"mine      {t_mine:7.2f}s   ({len(PATTERNS)} patterns, recall-aware "
          f"targeted, with val)\n")
    for name, status, ncov, nb, nr in rows:
        print(f"    {status:8s} {name:18s} branches {ncov}/{nb}   ({nr} rules)")
    print(f"\nrecovered: {n_full} full + {n_part} partial of {len(PATTERNS)}   "
          f"peak RSS={peak_gb():.2f}GB   total={t_gb+t_mine:.1f}s")


if __name__ == "__main__":
    main()
