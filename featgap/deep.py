"""featgap.deep -- recover DEEP conjunctions that greedy beam search can't seed.

The diagnosis (experiments/deep_diagnose.py): a deep AND-of-weak-conditions is a
discovery wall for a marginal beam -- every prefix is a statistical twin of a
random broad rule, so no single-rule score (precision, recall, F1) can single it
out, and the one steep shallow pattern monopolizes the beam.

The signal IS there, but only JOINTLY. So we detect the relevant feature block
with a joint model (gradient-boosted / random-forest importances on the residual)
and then REFINE: run the rule search restricted to that handful of features,
where the conjunction is no longer drowned. Sequential covering peels one pattern
at a time -- fit on the residual, find the strongest remaining conjunction over
its features, remove the frauds it covers, repeat.

    detect (joint model) -> restrict (top-K features) -> refine (F1 beam) -> cover

Depends on `arp` (one-directional) and, optionally, scikit-learn for the detector.
"""

from __future__ import annotations

import numpy as np

from arp.fast import rule_mask, BinSpec
from arp.targeted import targeted_beam_search


def _importance_block(Xtr, y_resid, *, top_k, subsample, seed):
    """Joint detector: features most responsible for the residual, by RandomForest
    importance (it splits on the conjunction's features even when each is
    marginally weak). Returns the original-index feature block."""
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    idx = rng.choice(n, min(subsample, n), replace=False)
    rf = RandomForestClassifier(n_estimators=120, max_depth=None,
                                n_jobs=-1, random_state=seed, max_features="sqrt")
    rf.fit(np.asarray(Xtr[idx]).astype(np.float32), y_resid[idx])
    imp = rf.feature_importances_
    return np.argsort(imp)[::-1][:top_k], imp


def recover_deep(Xtr, Xva, spec, ytr, yva, covered_tr, *, max_rounds=6,
                 top_k=40, target_precision=0.7, min_recall=0.01, min_support=40,
                 beam_width=64, max_depth=18, subsample=80_000, seed=0,
                 verbose=True):
    """Sequential-covering recovery of deep conjunctions on the residual.

    `covered_tr` is the train mask already covered by the baseline rules. Each
    round: fit the joint detector on the still-uncovered frauds, restrict to its
    top-`top_k` features, run an F1 beam there (high precision target forces full
    conjunction growth), accept the best rule(s), and subtract their coverage.

    Returns (deep_rules, info) where each rule's predicate feature indices are in
    the ORIGINAL feature space, ready to apply with arp.fast.rule_mask.
    """
    covered = covered_tr.copy()
    deep_rules, info = [], []
    pos_all = int((ytr == 1).sum())

    for rnd in range(max_rounds):
        resid = (ytr == 1) & ~covered
        if resid.sum() < min_support:
            break
        keep = ~((ytr == 1) & covered)              # drop already-covered frauds
        Xk = Xtr[keep]
        y_resid = resid[keep].astype(int)

        block, imp = _importance_block(Xk, y_resid, top_k=top_k,
                                       subsample=subsample, seed=seed + rnd)
        # restrict to the detected block (search in this small space only)
        cols = sorted(int(c) for c in block)
        sub_spec = BinSpec([spec.edges[c] for c in cols], spec.pct, spec.n_bins)
        Xt, Xv = Xtr[:, cols], Xva[:, cols]
        y_resid_full = resid.astype(np.int64)        # target = uncovered frauds
        rules, _ = targeted_beam_search(
            Xt, y_resid_full.reshape(-1, 1), 0, sub_spec, min_recall=min_recall,
            target_precision=target_precision, min_support=min_support,
            beam_width=beam_width, max_depth=max_depth,
            Xbin_val=Xv, Y_val=(yva == 1).astype(np.int64).reshape(-1, 1),
            gap_tol=None, rank_by="f1")
        if not rules:
            if verbose:
                print(f"  [deep r{rnd}] residual={int(resid.sum()):,}  "
                      f"no rule over top-{top_k} block -> stop")
            break
        # remap to original feature indices, pick the best by NEW coverage
        best, best_new = None, 0
        for r in rules:
            preds = tuple((cols[f], op, k) for f, op, k in r.preds)
            m = rule_mask(preds, Xtr)
            new = int((m & resid).sum())
            if new > best_new:
                best, best_new = (preds, r, m), new
        if best is None or best_new < min_support:
            break
        preds, r, m = best
        deep_rules.append(preds)
        covered |= m
        depth = len(preds)
        info.append(dict(round=rnd, depth=depth, new_covered=best_new,
                         val_prec=r.val_precision, val_rec=r.val_recall))
        if verbose:
            print(f"  [deep r{rnd}] residual={int(resid.sum()):,}  detected "
                  f"block top-{top_k}  -> rule depth {depth}  "
                  f"covers +{best_new:,} frauds  (valP={r.val_precision:.2f})")
    return deep_rules, info
