"""Does ranking the beam by F1 (instead of precision) recover deeper patterns?

Same deep data as deep_bench; runs the targeted beam with rank_by='precision'
vs rank_by='f1', identical otherwise, and compares coverage + recovery. F1
rewards the high-recall prefixes of deep conjunctions that precision buries --
the question is whether that is enough to survive the beam.

Run:  python -m experiments.deep_rank [small|large]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from arp.encode import encode_split_cm
from arp.fast import rule_mask
from arp.targeted import targeted_beam_search

from experiments.deep_bench import make_deep, DEPTHS, N_BINS, score_recovery

SCALES = {"small": (200_000, 500), "large": (2_000_000, 1000)}


def coverage(rules, Xva, yv, mo_va):
    cov = np.zeros(len(yv), dtype=bool)
    for r in rules:
        cov |= rule_mask(r.preds, Xva)
    pos = yv == 1
    tp = int((cov & pos).sum())
    rec = tp / max(1, int(pos.sum()))
    prec = tp / max(1, int(cov.sum()))
    pm = [float((cov & m).sum() / max(1, int(m.sum()))) for m in mo_va]
    return rec, prec, pm, int(cov.sum())


def run(n, n_features, seed=0):
    print(f"\n{'='*92}\nBEAM RANKING: precision vs F1   n={n:,}  F={n_features}  "
          f"depths={DEPTHS}\n{'='*92}")
    S, y, mo, planted, names, tr, va, n_sig = make_deep(n, n_features, seed)
    ytr, yva = y[tr], y[va]
    mo_va = [m[va] for m in mo]
    min_support = max(40, n // 4000)

    def make_column(f):
        if f < n_sig:
            return S[:, f]
        return np.random.default_rng(seed * 100003 + f).standard_normal(n).astype(np.float32)

    Xtr, Xva, spec = encode_split_cm(make_column, n_features, tr, va,
                                     n_bins=N_BINS, sample=100_000, seed=seed + 7)

    for mode in ("precision", "f1"):
        t0 = time.perf_counter()
        rules, _ = targeted_beam_search(
            Xtr, ytr.reshape(-1, 1), 0, spec, min_recall=0.03,
            target_precision=0.6, min_support=min_support, beam_width=48,
            max_depth=16, Xbin_val=Xva, Y_val=yva.reshape(-1, 1),
            gap_tol=None, rank_by=mode)
        rec, prec, pm, fl = coverage(rules, Xva, yva, mo_va)
        recov = [score_recovery(rules, planted[p])["recovered"] for p in range(len(DEPTHS))]
        rcv = " ".join(f"{r}/{d}" for r, d in zip(recov, DEPTHS))
        pms = "  ".join(f"d{d}={pm[p]:.2f}" for p, d in enumerate(DEPTHS))
        print(f"\n  rank_by={mode:9s} [{time.perf_counter()-t0:.0f}s]  rules={len(rules)}  "
              f"recall={rec:.2f}  precision={prec:.2f}  flagged={fl:,}")
        print(f"    per-pattern recall: {pms}")
        print(f"    exact recovered:    {rcv}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    run(*SCALES[which])
