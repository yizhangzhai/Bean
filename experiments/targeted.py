"""Precision/recall-targeted growth, with train/val gap control.

Shows, per fraud type, the rule that satisfies (precision>=target,
recall>=floor) and the depth it stopped at -- and that the recall floor /
precision target / val-gap kill the spurious deep overfit conditions seen in
the plain lift-maximizing search.
"""

from __future__ import annotations

import numpy as np

from arp.data import make_binned_fraud_data_large, train_val_split, GROUND_TRUTH_FEATURES
from arp.targeted import targeted_beam_search


def run(n=200_000, n_features=80, n_bins=10, seed=0,
        target_precision=0.5, min_recall=0.30, gap_tol=0.15):
    Xbin, spec, Y, loss, fnames, tnames, gt = make_binned_fraud_data_large(
        n=n, n_features=n_features, n_bins=n_bins, seed=seed)
    # train/val split on the binned matrix
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Xtr, Xva, Ytr, Yva = Xbin[tr], Xbin[va], Y[tr], Y[va]

    print(f"\n{'='*78}\nTARGETED GROWTH  n={n:,} F={n_features}  "
          f"target_precision={target_precision} min_recall={min_recall} "
          f"gap_tol={gap_tol}\n{'='*78}")
    print("(recall floor is an admissible prune: recall only falls with depth)\n")

    for c, tn in enumerate(tnames):
        rules, trace = targeted_beam_search(
            Xtr, Ytr, c, spec,
            min_recall=min_recall, target_precision=target_precision,
            min_support=40, beam_width=8, max_depth=4,
            Xbin_val=Xva, Y_val=Yva, gap_tol=gap_tol)
        print(f"--- {tn}  (planted features {sorted(GROUND_TRUTH_FEATURES[tn])}) ---")
        print(f"    {trace[-1]}")
        if not rules:
            print("    no rule met the targets (guardrails rejected overfit paths)\n")
            continue
        # report the best-recall accepted rule that also generalizes
        for r in rules[:2]:
            ok = GROUND_TRUTH_FEATURES[tn].issubset({f for f, _, _ in r.preds})
            print(f"    depth={r.depth}  P={r.precision:.2f} R={r.recall:.2f} "
                  f"| val P={r.val_precision:.2f} R={r.val_recall:.2f} "
                  f"gap={r.gap:+.2f}  [{'recovers GT' if ok else 'partial'}]")
            print(f"      stop: {r.stop_reason}")
            print(f"      rule: {r.label(fnames, spec)}")
        print()


if __name__ == "__main__":
    run()
