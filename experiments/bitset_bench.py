"""Compare mining paths on identical binned data:

  A) histogram subset-rescan  (arp.fast.fast_beam_search)
  B) coarse-to-fine + bitset   (arp.bitset.coarse_to_fine_mine)

Same recovery expected; B should be much faster on `mine` at large F because
it prunes features (coarse rank) and scores conjunctions by AND+popcount.

    python experiments/bitset_bench.py --n 1000000 --features 500
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from arp.data import make_binned_fraud_data_large, GROUND_TRUTH_FEATURES
from arp.fast import fast_beam_search
from arp.bitset import coarse_to_fine_mine
from arp.scoring import base_rates, objective_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--features", type=int, default=500)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    N, F = args.n, args.features

    print(f"\n{'='*72}\nBITSET vs HISTOGRAM  n={N:,}  F={F}  bins={args.bins}  "
          f"top_k={args.top_k}  sample={args.sample:,}\n{'='*72}")
    Xbin, spec, Y, loss, fnames, tnames, gt = make_binned_fraud_data_large(
        n=N, n_features=F, n_bins=args.bins, seed=args.seed)
    Yw = Y.astype(np.float64)
    base = base_rates(Yw, N)

    # ---- A) histogram subset-rescan ----
    tA = time.perf_counter()
    hitA = 0
    labA = []
    for c, tn in enumerate(tnames):
        obj = lambda lift, c=c: objective_single(lift, c)
        rules = fast_beam_search(Xbin, Yw, base, obj, spec, beam_width=8,
                                 max_depth=3, min_support=40)
        top = rules[0]
        hitA += GROUND_TRUTH_FEATURES[tn].issubset(top.features())
        labA.append((tn, top.label(fnames, spec)))
    tA = time.perf_counter() - tA

    # ---- B) coarse-to-fine + bitset ----
    tB = time.perf_counter()
    hitB = 0
    labB = []
    times = {"rank": 0.0, "build_bits": 0.0, "search": 0.0}
    for c, tn in enumerate(tnames):
        obj = lambda lift, c=c: objective_single(lift, c)
        rules, tm = coarse_to_fine_mine(
            Xbin, Y, base, obj, c, spec, top_k=args.top_k,
            sample_rows=args.sample, coarse_step=2, beam_width=8,
            max_depth=3, min_support=40)
        for k in times:
            times[k] += tm[k]
        top = rules[0]
        hitB += GROUND_TRUTH_FEATURES[tn].issubset(top.features())
        labB.append((tn, top.label(fnames, spec)))
    tB = time.perf_counter() - tB

    print(f"\nA) histogram subset-rescan : {tA:7.2f}s   recovered {hitA}/3")
    for tn, lab in labA:
        print(f"     {tn:18s} {lab}")
    print(f"\nB) coarse-to-fine + bitset : {tB:7.2f}s   recovered {hitB}/3")
    print(f"     breakdown: rank={times['rank']:.1f}s  "
          f"build_bits={times['build_bits']:.1f}s  search={times['search']:.1f}s "
          f"(summed over 3 types)")
    for tn, lab in labB:
        print(f"     {tn:18s} {lab}")
    print(f"\nspeedup on mine: {tA/tB:.1f}x   (same recovery: {hitA}=={hitB})")


if __name__ == "__main__":
    main()
