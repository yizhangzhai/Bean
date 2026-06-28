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

from .progress import make_progress


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


def _hist_tp(xb, yc, n_bins, min_support, active=None):
    """For one bin column: (op, k, support, tp) over thresholds (active or all)."""
    B = n_bins
    counts = np.bincount(xb, minlength=B).astype(np.float64)
    cc = np.cumsum(counts)
    wc = np.cumsum(np.bincount(xb, weights=yc, minlength=B))
    tot = wc[-1]
    Ncur = xb.shape[0]
    out = []
    ks = range(B - 1) if active is None else active
    for k in ks:
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
    max_accept=2000,
    policy=None,
    rank_by="precision",
    progress=None,
):
    """Grow rules toward (precision >= target, recall >= floor); return accepted.

    `policy` (arp.constraints.RulePolicy) enforces interpretability constraints
    during discovery -- allowed split directions / ranges / 1-way / forbidden
    feature pairs -- so every emitted rule is compliant by construction.

    Returns (accepted, trace) where trace logs why paths stopped. `max_accept`
    caps the accepted pool so a strong, common pattern can't generate tens of
    thousands of near-duplicate variants before the minimality filter runs.
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
        # train-only metrics here; val is computed lazily (it is expensive and
        # ~99% of candidates die on the recall floor before ever needing it)
        return TargetedRule(tuple(preds), int(support), tp, prec, rec, depth=depth)

    def add_val(r):
        if have_val and np.isnan(r.val_precision):
            r.val_precision, r.val_recall = _eval_val(r.preds, Xbin_val, yc_val,
                                                      total_c_val)
            r.gap = r.precision - r.val_precision
        return r

    def triage_train(r):
        """Cheap train-only split: recall floor / precision target / grow."""
        if r.recall < min_recall:
            return "prune"                          # admissible: subtree dead
        return "accept" if r.precision >= target_precision else "grow"

    def settle(accept_cands, grow_cands):
        """Apply val-gap only to survivors: accepted + selected beam."""
        for r in accept_cands:
            if len(accepted) >= max_accept:
                break
            add_val(r)
            if gap_tol is not None and have_val and r.gap > gap_tol:
                pruned["train/val gap"] += 1
            else:
                r.stop_reason = "precision target met"
                accepted.append(r)
        # rank grow candidates on train precision (minus a soft penalty for
        # discouraged feature co-occurrence), gap-check only the top beam.
        # rank_by="f1" instead balances precision & recall -- which rewards the
        # high-recall prefixes of deep conjunctions that pure precision buries.
        def rkey(r):
            pen = (policy.rule_penalty({f for f, _, _ in r.preds})
                   if policy is not None else 0.0)
            if rank_by == "f1":
                P, R = r.precision, r.recall
                base = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
            else:
                base = r.precision
            return (base - pen, r.recall)
        grow_cands.sort(key=rkey, reverse=True)
        beam = []
        for r in grow_cands:
            if len(beam) >= beam_width:
                break
            add_val(r)
            if gap_tol is not None and have_val and r.gap > gap_tol:
                pruned["train/val gap"] += 1
                continue
            r.mask = _rule_mask(r.preds, Xbin)
            beam.append(r)
        return beam

    nb = spec.n_bins
    prog = make_progress(progress, f"targeted[{target}]")
    prog.start(F=F, beam=beam_width, max_depth=max_depth,
               targets=f"P>={target_precision} R>={min_recall} gap<={gap_tol}")

    def _report(d, beam_in, beam_out, explored, pr0):
        pool = beam_out + (acc_c if isinstance(acc_c, list) else [])
        bp, br = (max((r.precision for r in pool), default=0.0),
                  max((r.recall for r in pool), default=0.0))
        prog.depth(d, beam_in=beam_in, beam_out=len(beam_out), explored=explored,
                   accepted_new=len(accepted) - pr0[2], accepted_total=len(accepted),
                   prune_recall=pruned["recall<floor"] - pr0[0],
                   prune_gap=pruned["train/val gap"] - pr0[1], best_p=bp, best_r=br)

    # ---- depth 1 ----
    pr0 = (pruned["recall<floor"], pruned["train/val gap"], len(accepted))
    acc_c, grow_c, explored = [], [], 0
    for f in range(F):
        af = spec.active_for(f)
        if af is not None and len(af) == 0:        # feature has no useful cut
            continue
        for op, k, s, tp in _hist_tp(Xbin[:, f], yc, nb, min_support, af):
            if policy is not None and not policy.pred_ok(f, op, k, nb):
                continue
            explored += 1
            r = make_rule([(f, op, k)], s, tp, 1)
            a = triage_train(r)
            if a == "prune":
                pruned["recall<floor"] += 1
            elif a == "accept":
                acc_c.append(r)
            else:
                grow_c.append(r)
    beam = settle(acc_c, grow_c)
    _report(1, 1, beam, explored, pr0)

    # ---- grow ----
    for depth in range(2, max_depth + 1):
        pr0 = (pruned["recall<floor"], pruned["train/val gap"], len(accepted))
        beam_in, explored = len(beam), 0
        acc_c, grow_c, seen = [], {}, set()
        for r in beam:
            idx = np.flatnonzero(r.mask)
            sub_yc = yc[idx]
            used = r.used()
            rfeats = {pf for pf, _ in used}
            for f in range(F):
                af = spec.active_for(f)
                if af is not None and len(af) == 0:
                    continue
                xb = Xbin[idx, f]
                for op, k, s, tp in _hist_tp(xb, sub_yc, nb, min_support, af):
                    if (f, op) in used:
                        continue
                    if policy is not None and (
                            not policy.pred_ok(f, op, k, nb)
                            or not policy.extend_ok(rfeats, f, op)):
                        continue
                    explored += 1
                    nr = make_rule(r.preds + ((f, op, k),), s, tp, depth)
                    a = triage_train(nr)
                    if a == "prune":
                        pruned["recall<floor"] += 1
                    elif a == "accept":
                        acc_c.append(nr)
                    else:
                        kk = nr.key()
                        if kk not in seen or nr.precision > grow_c[kk].precision:
                            seen.add(kk)
                            grow_c[kk] = nr
        if not acc_c and not grow_c:
            break
        beam = settle(acc_c, list(grow_c.values()))
        _report(depth, beam_in, beam, explored, pr0)
        if (not beam and not acc_c) or len(accepted) >= max_accept:
            break

    # dedup by condition-set (keep best precision)
    uniq = {}
    for r in accepted:
        if r.key() not in uniq or r.precision > uniq[r.key()].precision:
            uniq[r.key()] = r
    # minimality: a rule that strictly contains an already-accepted (simpler)
    # rule is redundant -- the shorter rule already meets the target. Keep only
    # minimal accepted rules. Collapses thousands of deeper extensions to cores.
    cand = sorted(uniq.values(), key=lambda r: (len(r.preds), -r.precision))
    kept, kept_sets = [], []
    for r in cand:
        if policy is not None and not policy.rule_ok({f for f, _, _ in r.preds}):
            continue                                  # required-with not satisfied
        fs = frozenset(r.preds)
        if any(ks <= fs for ks in kept_sets):
            continue
        kept.append(r)
        kept_sets.append(fs)
    out = sorted(kept, key=lambda r: (r.recall, r.precision), reverse=True)
    trace.append(f"accepted={len(out)} (of {len(uniq)} pre-minimality)  "
                 f"pruned: recall_floor={pruned['recall<floor']}  "
                 f"val_gap={pruned['train/val gap']}")
    prog.done(rules=len(out), prune_recall=pruned["recall<floor"],
              prune_gap=pruned["train/val gap"],
              note=f"(of {len(uniq)} pre-minimality)")
    return out, trace
