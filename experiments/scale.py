"""Scalability benchmark: time each stage as N and F grow, up to 2M x 1000.

Run ONE config per process (clean peak-RSS reading):
    python experiments/scale.py --n 2000000 --features 1000 --bins 10

Stages timed:
  gen      synthetic data (float32, block-filled)
  fit_bins quantile-bin all features -> int8 Xbin   (X freed afterward)
  depth1   histogram scan of EVERY threshold of EVERY feature (the core scan)
  mine     full per-type beam mine, depth 3

Reports throughput (cells/sec = N*F/time) and peak RSS.
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time

import numpy as np

from arp.data import make_fraud_data_large, GROUND_TRUTH_FEATURES
from arp.fast import fast_beam_search, fit_bins
from arp.scoring import base_rates, objective_single


def peak_gb() -> float:
    # macOS: ru_maxrss is bytes; Linux: kilobytes
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    div = 1e9 if sys.platform == "darwin" else 1e6
    return rss / div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--features", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    N, F = args.n, args.features
    cells = N * F

    print(f"\n{'='*72}")
    print(f"SCALE  n={N:,}  features={F}  bins={args.bins}  cells={cells:,}")
    print(f"{'='*72}")

    t = time.perf_counter()
    data = make_fraud_data_large(n=N, n_features=F, seed=args.seed,
                                 block=min(100, F))
    t_gen = time.perf_counter() - t
    print(f"gen       {t_gen:7.2f}s   X={data.X.nbytes/1e9:.1f}GB float32   "
          f"fraud rates={np.round(data.Y.mean(0)*100,2)}")

    Yw = data.Y.astype(np.float64)
    base = base_rates(Yw, N)

    t = time.perf_counter()
    Xbin, spec = fit_bins(data.X, n_bins=args.bins)
    t_bin = time.perf_counter() - t
    print(f"fit_bins  {t_bin:7.2f}s   Xbin={Xbin.nbytes/1e9:.2f}GB int8   "
          f"({cells/t_bin/1e6:.0f}M cells/s)")

    # free the big float matrix -- mining only needs Xbin
    data.X = None
    gc.collect()

    # depth-1 only: histogram scan of all thresholds of all features
    obj0 = lambda lift: objective_single(lift, 0)
    t = time.perf_counter()
    d1 = fast_beam_search(Xbin, Yw, base, obj0, spec, beam_width=8,
                          max_depth=1, min_support=40)
    t_d1 = time.perf_counter() - t
    print(f"depth1    {t_d1:7.2f}s   scanned {F*(args.bins-1)*2:,} predicates   "
          f"({cells/t_d1/1e6:.0f}M cells/s)   top single-lift={d1[0].lift[0]:.1f}")

    # full per-type mine (depth 3) + recovery check
    t = time.perf_counter()
    n_hit = 0
    rows = []
    for c, tname in enumerate(data.type_names):
        obj = lambda lift, c=c: objective_single(lift, c)
        rules = fast_beam_search(Xbin, Yw, base, obj, spec, beam_width=8,
                                 max_depth=3, min_support=40)
        top = rules[0]
        planted = GROUND_TRUTH_FEATURES[tname]
        hit = planted.issubset(top.features())
        n_hit += hit
        rows.append((tname, hit, top.label(data.feature_names, spec)))
    t_mine = time.perf_counter() - t
    print(f"mine      {t_mine:7.2f}s   3 types x depth-3 beam")
    for tname, hit, lab in rows:
        print(f"    [{'OK ' if hit else 'MISS'}] {tname:18s} {lab}")
    print(f"\nrecovered {n_hit}/3 planted signatures   peak RSS={peak_gb():.1f}GB   "
          f"total={t_gen+t_bin+t_d1+t_mine:.1f}s")


if __name__ == "__main__":
    main()
