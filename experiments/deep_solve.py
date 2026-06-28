"""Solve the deep-conjunction recall bottleneck with joint-detect -> refine.

Pipeline:
  0. baseline   F1 beam over all features              (catches shallow patterns)
  1. recover_deep   sequential covering: RandomForest detects each remaining
                    pattern's feature block on the residual, an F1 beam refines
                    the conjunction within that block, subtract its coverage, repeat

Reports recall / precision / per-pattern recall and how many of each planted
conjunction's conditions the recovered INTERPRETABLE rules contain.

Run:  python -m experiments.deep_solve [small|large]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from arp.encode import encode_split_cm
from arp.fast import rule_mask
from arp.targeted import targeted_beam_search
from featgap import uncovered_positives, recover_deep

from experiments.deep_bench import make_deep, DEPTHS, N_BINS, score_recovery

SCALES = {"small": (200_000, 500), "large": (2_000_000, 1000)}


def coverage(rule_preds, Xva, yv, mo_va):
    cov = np.zeros(len(yv), dtype=bool)
    for preds in rule_preds:
        cov |= rule_mask(preds, Xva)
    pos = yv == 1
    tp = int((cov & pos).sum())
    rec = tp / max(1, int(pos.sum()))
    prec = tp / max(1, int(cov.sum()))
    pm = [float((cov & m).sum() / max(1, int(m.sum()))) for m in mo_va]
    return rec, prec, pm, int(cov.sum())


def run(n, n_features, seed=0):
    print(f"\n{'='*92}\nSOLVE DEEP RECALL: joint-detect -> refine   "
          f"n={n:,}  F={n_features}  depths={DEPTHS}\n{'='*92}")
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

    # ---- 0. baseline (F1 beam over all features) ----
    t0 = time.perf_counter()
    base, _ = targeted_beam_search(
        Xtr, ytr.reshape(-1, 1), 0, spec, min_recall=0.03, target_precision=0.6,
        min_support=min_support, beam_width=48, max_depth=16,
        Xbin_val=Xva, Y_val=yva.reshape(-1, 1), gap_tol=None, rank_by="f1")
    base_preds = [r.preds for r in base]
    rec, prec, pm, fl = coverage(base_preds, Xva, yva, mo_va)
    print(f"\n  0. baseline (F1 beam)   recall={rec:.2f}  precision={prec:.2f}  "
          f"rules={len(base)}  [{time.perf_counter()-t0:.0f}s]")
    print("     per-pattern recall: " +
          "  ".join(f"d{d}={pm[p]:.2f}" for p, d in enumerate(DEPTHS)))

    # ---- 1. recover_deep on the residual ----
    t0 = time.perf_counter()
    _, cov_tr = uncovered_positives(base, Xtr, ytr)
    deep, infos = recover_deep(Xtr, Xva, spec, ytr, yva, cov_tr,
                               max_rounds=6, top_k=40, target_precision=0.7,
                               min_support=min_support, beam_width=64, max_depth=18,
                               seed=seed)
    all_preds = base_preds + deep
    rec, prec, pm, fl = coverage(all_preds, Xva, yva, mo_va)
    print(f"\n  1. + recover_deep       recall={rec:.2f}  precision={prec:.2f}  "
          f"deep rules={len(deep)}  [{time.perf_counter()-t0:.0f}s]")
    print("     per-pattern recall: " +
          "  ".join(f"d{d}={pm[p]:.2f}" for p, d in enumerate(DEPTHS)))

    # ---- recovery of the exact conjunctions (interpretable) ----
    class _Shim:
        def __init__(s, preds):
            s.preds = preds
            s.val_precision = s.val_recall = float("nan")
    deep_shims = [_Shim(p) for p in deep]
    print("\n     exact conjunction recovery by the deep rules:")
    for p, d in enumerate(DEPTHS):
        srb = score_recovery([_Shim(pp) for pp in base_preds], planted[p])
        srd = score_recovery(deep_shims, planted[p])
        best = max(srb["recovered"], srd["recovered"])
        print(f"       depth {d:>2}: recovered {best}/{d} conditions"
              f"  (baseline {srb['recovered']}, deep {srd['recovered']})")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    run(*SCALES[which])
