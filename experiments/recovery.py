"""Does the miner recover known planted patterns? And how fast?

Plants 3 signatures in features {0,1},{2,3},{4,5} among many noise features,
then mines per type and checks whether each rule's features cover the planted
ground truth. Reports per-stage timing.
"""

from __future__ import annotations

import time

import numpy as np

from arp.data import (
    GROUND_TRUTH_FEATURES, make_fraud_data_large, train_val_split,
)
from arp.fast import fast_beam_search, fit_bins
from arp.scoring import base_rates, objective_single


def run(n=100_000, n_features=50, n_bins=10, seed=0):
    t0 = time.perf_counter()
    data = make_fraud_data_large(n=n, n_features=n_features, seed=seed, block=50)
    t_gen = time.perf_counter() - t0

    tr, va = train_val_split(data, seed=seed + 1)
    Wtr, Wva = data.Y[tr].astype(float), data.Y[va].astype(float)
    base_tr = base_rates(Wtr, len(tr))
    base_va = base_rates(Wva, len(va))

    t0 = time.perf_counter()
    Xbin_tr, spec = fit_bins(data.X[tr], n_bins=n_bins)
    t_bin = time.perf_counter() - t0
    Xbin_va, _ = fit_bins(data.X[va], n_bins=n_bins)

    print(f"\n{'='*72}\nRECOVERY TEST  n={n:,}  features={n_features}  "
          f"bins={n_bins}\n{'='*72}")
    print(f"data-gen: {t_gen:.2f}s   fit_bins(train): {t_bin:.2f}s")
    print(f"{'type':18s} {'planted':14s} {'recovered?':10s} "
          f"{'val-lift':>9s} {'support':>8s}  mined rule")
    print("-" * 100)

    n_hit = 0
    t0 = time.perf_counter()
    for c, tname in enumerate(data.type_names):
        obj = lambda lift, c=c: objective_single(lift, c)
        rules = fast_beam_search(Xbin_tr, Wtr, base_tr, obj, spec,
                                 beam_width=8, max_depth=3, min_support=40)
        top = rules[0]
        # validation lift
        m_va = np.ones(len(va), dtype=bool)
        for f, op, k in top.preds:
            col = Xbin_va[:, f]
            m_va &= (col > k) if op == ">" else (col <= k)
        s_va = int(m_va.sum())
        caught_va = Wva[m_va].sum(axis=0)
        lift_va = (caught_va[c] / s_va) / base_va[c] if s_va else 0.0

        planted = GROUND_TRUTH_FEATURES[tname]
        found = top.features()
        hit = planted.issubset(found)
        n_hit += hit
        print(f"{tname:18s} {str(sorted(planted)):14s} "
              f"{'YES' if hit else 'no':10s} {lift_va:9.1f} {s_va:8d}  "
              f"{top.label(data.feature_names, spec)}")
    t_mine = time.perf_counter() - t0

    print("-" * 100)
    print(f"recovered {n_hit}/{len(data.type_names)} planted signatures   "
          f"| mining (3 types, depth 3): {t_mine:.2f}s")
    return n_hit, len(data.type_names)


if __name__ == "__main__":
    run()
