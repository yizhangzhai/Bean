"""featgap.gap -- identify the gap, and the cheapest diagnostic.

The gap is the set of positives no rule covers. Before engineering anything,
the most informative (and free) check is to *re-mine the residual with the same
engine*: if an axis-aligned rule still covers it, the gap was greedy myopia
(widen/deepen the search), not a missing feature; if nothing meets targets, the
gap is genuinely non-axis and warrants feature engineering.

Depends on `arp` (one-directional): the feature-engineering layer sits ON TOP of
the core miner, never the other way around.
"""

from __future__ import annotations

import numpy as np

from arp.fast import rule_mask
from arp.targeted import targeted_beam_search


def uncovered_positives(rules_or_masks, Xbin, y):
    """Return (gap_mask, covered_mask). Accepts rules (with .preds) or masks."""
    covered = np.zeros(len(y), dtype=bool)
    for r in rules_or_masks:
        m = r if isinstance(r, np.ndarray) else rule_mask(r.preds, Xbin)
        covered |= m
    return (y == 1) & ~covered, covered


def remine_residual(Xbin, y, covered, spec, *, target_precision=0.5,
                    min_recall=0.10, min_support=40, beam_width=16, max_depth=4,
                    gap_tol=0.25, myopia_recall=0.5, seed=0):
    """Step-1 diagnostic: is the gap still coverable by an AXIS-aligned rule?

    Re-mines the residual (uncovered frauds vs non-fraud, dropping already
    covered frauds). Verdict by how MUCH of the residual axis rules recover:
      no rule              -> genuine non-axis gap (engineer)
      rule, low recall     -> axis rules only patch corners; gap mostly non-axis
      rule, recall >= myopia_recall -> greedy myopia (widen/deepen the search)
    Returns a dict with the verdict and the best residual rule, plus the union
    recall over all residual rules (how much of the gap axis rules can recover).
    """
    gap = (y == 1) & ~covered
    if int(gap.sum()) < max(2 * min_support, 10):    # too small to diagnose
        return dict(gap=int(gap.sum()), fillable_by_axis=False, union_recall=0.0,
                    verdict="residual too small to diagnose", best=None,
                    precision=0.0, recall=0.0, rules=[])
    keep = ~((y == 1) & covered)                 # drop covered frauds
    Xk = Xbin[keep]
    yk = gap[keep].astype(np.int64).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(yk))
    cut = int(len(yk) * 0.67)
    tr, va = perm[:cut], perm[cut:]
    rules, _ = targeted_beam_search(
        Xk[tr], yk[tr], 0, spec, min_recall=min_recall,
        target_precision=target_precision, min_support=min_support,
        beam_width=beam_width, max_depth=max_depth,
        Xbin_val=Xk[va], Y_val=yk[va], gap_tol=gap_tol)
    if not rules:
        return dict(gap=int(gap.sum()), fillable_by_axis=False, union_recall=0.0,
                    verdict="no axis rule -> genuine non-axis gap (engineer)",
                    best=None, precision=0.0, recall=0.0, rules=[])
    # union recall of all residual rules over the residual positives (val)
    gap_va = yk[va, 0] == 1
    union = np.zeros(len(va), dtype=bool)
    for r in rules:
        union |= rule_mask(r.preds, Xk[va])
    union_recall = float((union & gap_va).sum() / max(1, gap_va.sum()))
    best = max(rules, key=lambda r: r.val_recall)
    # Myopia = ONE axis rule would have covered the gap (better search suffices).
    # If the best single rule has low recall but many boxes tile some of it, the
    # structure is non-axis (a feature is the clean fix, not more rules).
    if best.val_recall >= myopia_recall:
        verdict = (f"a single axis rule reaches R={best.val_recall:.2f} -> "
                   f"greedy myopia (widen/deepen the search)")
    else:
        verdict = (f"best single axis rule only R={best.val_recall:.2f} "
                   f"(though boxes can tile {union_recall:.0%}) -> NON-axis "
                   f"structure (engineer a feature)")
    return dict(gap=int(gap.sum()), fillable_by_axis=best.val_recall >= myopia_recall,
                union_recall=union_recall, verdict=verdict, best=best,
                precision=best.val_precision, recall=best.val_recall, rules=rules)
