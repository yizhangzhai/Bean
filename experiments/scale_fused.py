"""Scale benchmark with the FUSED generate+bin path (memory-pressure fix).

Never materializes the 8GB float X: generates each column block, bins to int8,
discards. Peak memory ~= Xbin + one block. Use this to get the *true* cost of
binning at 2M x 1000, vs. the swap-bound 78-min fit_bins of the naive path.

    python experiments/scale_fused.py --n 2000000 --features 1000
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np

from arp.data import make_binned_fraud_data_large, GROUND_TRUTH_FEATURES
from arp.fast import fast_beam_search
from arp.scoring import base_rates, objective_single


def peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if sys.platform == "darwin" else 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--features", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    N, F = args.n, args.features

    print(f"\n{'='*72}\nSCALE (fused gen+bin)  n={N:,}  features={F}  "
          f"bins={args.bins}\n{'='*72}")

    t = time.perf_counter()
    Xbin, spec, Y, loss, fnames, tnames, gt = make_binned_fraud_data_large(
        n=N, n_features=F, n_bins=args.bins, seed=args.seed, block=50)
    t_genbin = time.perf_counter() - t
    print(f"gen+bin   {t_genbin:7.2f}s   Xbin={Xbin.nbytes/1e9:.2f}GB int8   "
          f"({N*F/t_genbin/1e6:.0f}M cells/s)   fraud={np.round(Y.mean(0)*100,2)}")

    Yw = Y.astype(np.float64)
    base = base_rates(Yw, N)

    obj0 = lambda lift: objective_single(lift, 0)
    t = time.perf_counter()
    fast_beam_search(Xbin, Yw, base, obj0, spec, beam_width=8, max_depth=1,
                     min_support=40)
    t_d1 = time.perf_counter() - t
    print(f"depth1    {t_d1:7.2f}s   ({N*F/t_d1/1e6:.0f}M cells/s)")

    t = time.perf_counter()
    n_hit = 0
    rows = []
    for c, tname in enumerate(tnames):
        obj = lambda lift, c=c: objective_single(lift, c)
        rules = fast_beam_search(Xbin, Yw, base, obj, spec, beam_width=8,
                                 max_depth=3, min_support=40)
        top = rules[0]
        hit = GROUND_TRUTH_FEATURES[tname].issubset(top.features())
        n_hit += hit
        rows.append((tname, hit, top.label(fnames, spec)))
    t_mine = time.perf_counter() - t
    print(f"mine      {t_mine:7.2f}s   3 types x depth-3 beam")
    for tname, hit, lab in rows:
        print(f"    [{'OK ' if hit else 'MISS'}] {tname:18s} {lab}")
    print(f"\nrecovered {n_hit}/3   peak RSS={peak_gb():.1f}GB   "
          f"total={t_genbin+t_d1+t_mine:.1f}s")


if __name__ == "__main__":
    main()
