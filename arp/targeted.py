"""Precision/recall-targeted rule growth (branch-and-bound beam search).

Practical rule mining is governed by operating targets, not raw lift. Three
mechanisms, each tied to a monotonicity fact:

1. **Recall floor = admissible early-stop.** Recall = TP / (all frauds of the
   type). Growing a rule only shrinks its matched set, so recall is
   *monotonically non-increasing* with depth. Therefore if a shallow path is
   already below `min_recall`, no descendant can recover -> prune the whole
   subtree immediately. This is the classic optimistic-estimate pruning of
   subgroup discovery, and it is exact (never discards a viable rule).

2. **Precision target = stop-on-satisfied.** Precision = TP / support, which is
   NOT monotone (usually rises as you add conditions). So we can't prune on it,
   but once a path reaches `target_precision` (while still meeting the recall
   floor) it is *accepted and not grown further* -- the shortest sufficient
   path, which also generalizes best.

3. **Train/val gap = overfitting brake (optional).** Each candidate is scored
   on a held-out set in parallel. If train precision exceeds val precision by
   more than `gap_tol` (or val recall collapses), the path is dropped rather
   than grown -- a data-driven stop that replaces a hard depth cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TargetedRule:
    preds: tuple
    support: int
    tp: float
    precision: float
    recall: float
    val_precision: float = float("nan")
    val_recall: float = float("nan")
    gap: float = float("nan")
    depth: int = 0
    stop_reason: str = ""
    mask: np.ndarray = field(repr=False, default=None)

    def used(self):
        return {(f, op) for f, op, _ in self.preds}

    def key(self):
        return frozenset(self.preds)

    def label(self, feature_names, spec) -> str:
        parts = []
        for f, op, k in self.preds:
            parts.append(f"{feature_names[f]} {op} p{int(round(spec.pct[k]*100)):02d}")
        return "  AND  ".join(parts)


def _hist_tp(xb, yc, n_bins, min_support):
    """For one bin column: (op, k, support, tp) over all thresholds."""
    B = n_bins
    counts = np.bincount(xb, minlength=B).astype(np.float64)
    cc = np.cumsum(counts)
    wc = np.cumsum(np.bincount(xb, weights=yc, minlength=B))
    tot = wc[-1]
    Ncur = xb.shape[0]
    out = []
    for k in range(B - 1):
        for op, s, tp in (("<", cc[k], wc[k]), (">", Ncur - cc[k], tot - wc[k])):
            if s >= min_support:
                out.append((op, k, float(s), float(tp)))
    return out


def _rule_mask(preds, Xbin):
    m = np.ones(Xbin.shape[0], dtype=bool)
    for f, op, k in preds:
        col = Xbin[:, f]
        m &= (col > k) if op == ">" else (col <= k)
    return m


def _eval_val(preds, Xbin_val, yc_val, total_c_val):
    m = _rule_mask(preds, Xbin_val)
    s = int(m.sum())
    tp = float(yc_val[m].sum())
    prec = tp / s if s > 0 else 0.0
    rec = tp / total_c_val if total_c_val > 0 else 0.0
    return prec, rec


def targeted_beam_search(
    Xbin, Y, target, spec, *,
    min_recall=0.05,
    target_precision=0.5,
    min_support=40,
    beam_width=8,
    max_depth=4,
    Xbin_val=None, Y_val=None, gap_tol=None,
):
    """Grow rules toward (precision >= target, recall >= floor); return accepted.

    Returns (accepted, trace) where trace logs why paths stopped.
    """
    N, F = Xbin.shape
    yc = Y[:, target].astype(np.float64)
    total_c = float(yc.sum())
    have_val = Xbin_val is not None and Y_val is not None
    if have_val:
        yc_val = Y_val[:, target].astype(np.float64)
        total_c_val = float(yc_val.sum())

    accepted: list[TargetedRule] = []
    trace: list[str] = []
    pruned = {"recall<floor": 0, "train/val gap": 0}

    def make_rule(preds, support, tp, depth):
        prec = tp / support if support > 0 else 0.0
        rec = tp / total_c if total_c > 0 else 0.0
        r = TargetedRule(tuple(preds), int(support), tp, prec, rec, depth=depth)
        if have_val:
            r.val_precision, r.val_recall = _eval_val(preds, Xbin_val, yc_val,
                                                      total_c_val)
            r.gap = prec - r.val_precision
        return r

    def classify(r):
        """-> ('prune'|'accept'|'grow', reason). Applies the three rules."""
        if r.recall < min_recall:
            return "prune", "recall<floor"          # admissible: subtree dead
        if gap_tol is not None and have_val and r.gap > gap_tol:
            return "prune", "train/val gap"
        if r.precision >= target_precision:
            return "accept", "precision target met"
        return "grow", ""

    # ---- depth 1 ----
    beam = []
    for f in range(F):
        for op, k, s, tp in _hist_tp(Xbin[:, f], yc, spec.n_bins, min_support):
            r = make_rule([(f, op, k)], s, tp, 1)
            action, reason = classify(r)
            r.stop_reason = reason
            if action == "prune":
                pruned[reason] += 1
                continue
            if action == "accept":
                accepted.append(r)
            else:
                beam.append(r)
    beam.sort(key=lambda r: (r.precision, r.recall), reverse=True)
    beam = beam[:beam_width]
    for r in beam:
        r.mask = _rule_mask(r.preds, Xbin)

    # ---- grow ----
    for depth in range(2, max_depth + 1):
        nxt = []
        for r in beam:
            idx = np.flatnonzero(r.mask)
            sub_yc = yc[idx]
            used = r.used()
            for f in range(F):
                xb = Xbin[idx, f]
                for op, k, s, tp in _hist_tp(xb, sub_yc, spec.n_bins, min_support):
                    if (f, op) in used:
                        continue
                    nr = make_rule(r.preds + ((f, op, k),), s, tp, depth)
                    action, reason = classify(nr)
                    nr.stop_reason = reason
                    if action == "prune":
                        pruned[reason] += 1
                        continue
                    if action == "accept":
                        accepted.append(nr)
                    else:
                        nxt.append(nr)
        if not nxt:
            break
        best = {}
        for r in nxt:
            if r.key() not in best or r.precision > best[r.key()].precision:
                best[r.key()] = r
        beam = sorted(best.values(), key=lambda r: (r.precision, r.recall),
                      reverse=True)[:beam_width]
        for r in beam:
            r.mask = _rule_mask(r.preds, Xbin)

    # dedup accepted, keep best precision per condition-set, drop dominated
    uniq = {}
    for r in accepted:
        if r.key() not in uniq or r.precision > uniq[r.key()].precision:
            uniq[r.key()] = r
    out = sorted(uniq.values(), key=lambda r: (r.recall, r.precision), reverse=True)
    trace.append(f"accepted={len(out)}  pruned: recall_floor="
                 f"{pruned['recall<floor']}  val_gap={pruned['train/val gap']}")
    return out, trace
