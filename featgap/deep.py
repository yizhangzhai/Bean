"""featgap.deep -- recover DEEP conjunctions that greedy beam search can't seed.

The diagnosis (experiments/deep_diagnose.py): a deep AND-of-weak-conditions is a
discovery wall for a marginal beam -- every prefix is a statistical twin of a
random broad rule, so no single-rule score (precision, recall, F1) can single it
out, and the one steep shallow pattern monopolizes the beam.

The signal IS there, but only JOINTLY. So we detect the relevant feature block
with a joint model (a gradient-boosted tree on the residual -- its gain importance
surfaces the conjunction's features even when each is marginally weak) and then
REFINE: run the rule search restricted to that handful of features, where the
conjunction is no longer drowned. Sequential covering peels one pattern at a time
-- fit on the residual, find the strongest remaining conjunction over its
features, remove the frauds it covers, repeat.

    detect (joint model) -> restrict (top-K features) -> refine (F1 beam) -> cover

The detector is the SCOPE oracle, not the rule source (its own tree paths are
noisy/fragmented); the interpretable, policy-eligible rule is produced by the
refine step. Default detector is LightGBM -- histogram-based so it scales to high
dimension, handles missing values natively (NaN routed to a learned direction),
and takes categorical features directly (no one-hot) -- with a scikit-learn
RandomForest fallback if LightGBM is absent.

Depends on `arp` (one-directional); LightGBM / scikit-learn are optional.
"""

from __future__ import annotations

import numpy as np

from arp.fast import rule_mask, BinSpec
from arp.targeted import targeted_beam_search


def _importance_block(Xfit, y, *, top_k, seed, detector="lgbm", categorical=None):
    """Joint detector -> indices of the top-`top_k` features by gain importance.

    `Xfit` may contain NaN (LightGBM handles it natively). `categorical` is an
    optional list of column indices to treat as categorical (LightGBM only;
    columns must be integer-coded). Returns (block_indices, importances)."""
    if detector == "lgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                class_weight="balanced", n_jobs=-1, random_state=seed,
                importance_type="gain", verbosity=-1)
            fit_kw = {}
            if categorical:
                fit_kw["categorical_feature"] = list(categorical)
            model.fit(Xfit, y, **fit_kw)
            imp = np.asarray(model.feature_importances_, dtype=float)
            return np.argsort(imp)[::-1][:top_k], imp
        except ImportError:
            pass                                   # fall through to RandomForest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=seed,
                                max_features="sqrt", class_weight="balanced")
    rf.fit(np.nan_to_num(np.asarray(Xfit, dtype=np.float32)), y)   # RF needs no NaN
    imp = np.asarray(rf.feature_importances_, dtype=float)
    return np.argsort(imp)[::-1][:top_k], imp


def recover_deep(Xtr, Xva, spec, ytr, yva, covered_tr, *, max_rounds=6,
                 top_k=40, target_precision=0.7, precision_floor=0.2,
                 min_recall=0.01, min_support=40,
                 beam_width=64, max_depth=18, subsample=80_000, seed=0,
                 detector="lgbm", categorical=None, Xraw_tr=None, verbose=True):
    """Sequential-covering recovery of deep conjunctions on the residual.

    `covered_tr` is the train mask already covered by the baseline rules. Each
    round: fit the joint detector on the still-uncovered frauds, restrict to its
    top-`top_k` features, run an F1 beam there (high precision target forces full
    conjunction growth), accept the best rule(s), and subtract their coverage.

    With MANY overlapping patterns the detector's top-k block is a mix, so a
    high precision target can find no clean rule and stall sequential covering.
    Each round therefore RELAXES the precision target (`target_precision` down to
    `precision_floor`) until the restricted search yields a rule covering new
    frauds -- a contaminated pattern still gets a (lower-precision) rule and the
    next round proceeds on a cleaner residual.

    detector: "lgbm" (default, scalable + native missing/categorical) or "rf".
    Xraw_tr: optional raw (unbinned) train feature matrix for the DETECTOR only --
      pass it to let LightGBM exploit real NaN / categorical columns; the rule
      refine always runs on the binned `Xtr`. If None, the detector uses `Xtr`.
    categorical: column indices to treat as categorical in the detector (LightGBM).

    Returns (deep_rules, info); each rule's predicate feature indices are in the
    ORIGINAL feature space, ready to apply with arp.fast.rule_mask.
    """
    covered = covered_tr.copy()
    deep_rules, info = [], []
    src = Xraw_tr if Xraw_tr is not None else Xtr

    for rnd in range(max_rounds):
        resid = (ytr == 1) & ~covered
        if resid.sum() < min_support:
            break
        keep_idx = np.flatnonzero(~((ytr == 1) & covered))    # drop covered frauds
        rng = np.random.default_rng(seed + rnd)
        sub = rng.choice(keep_idx, min(subsample, len(keep_idx)), replace=False)
        Xfit = np.asarray(src[sub])
        if Xraw_tr is None:
            Xfit = Xfit.astype(np.float32)
        y_sub = resid[sub].astype(int)

        block, _ = _importance_block(Xfit, y_sub, top_k=top_k, seed=seed + rnd,
                                     detector=detector, categorical=categorical)
        cols = sorted(int(c) for c in block)
        sub_spec = BinSpec([spec.edges[c] for c in cols], spec.pct, spec.n_bins)
        Xt, Xv = Xtr[:, cols], Xva[:, cols]
        yv_col = (yva == 1).astype(np.int64).reshape(-1, 1)
        resid_col = resid.astype(np.int64).reshape(-1, 1)

        # relax the precision target until a rule covering new frauds appears
        schedule, tp = [], target_precision
        while tp >= precision_floor - 1e-9:
            schedule.append(round(tp, 3))
            tp *= 0.66
        best, best_new, used_tp = None, 0, None
        for tp in schedule:
            rules, _ = targeted_beam_search(
                Xt, resid_col, 0, sub_spec, min_recall=min_recall,
                target_precision=tp, min_support=min_support,
                beam_width=beam_width, max_depth=max_depth,
                Xbin_val=Xv, Y_val=yv_col, gap_tol=None, rank_by="f1")
            for r in rules:
                preds = tuple((cols[f], op, k) for f, op, k in r.preds)
                m = rule_mask(preds, Xtr)
                new = int((m & resid).sum())
                if new > best_new:
                    best, best_new, used_tp = (preds, r, m), new, tp
            if best is not None and best_new >= min_support:
                break
        if best is None or best_new < min_support:
            if verbose:
                print(f"  [deep r{rnd}] residual={int(resid.sum()):,}  "
                      f"no rule over top-{top_k} block (down to P={schedule[-1]}) "
                      f"-> stop")
            break
        preds, r, m = best
        deep_rules.append(preds)
        covered |= m
        info.append(dict(round=rnd, depth=len(preds), new_covered=best_new,
                         val_prec=r.val_precision, val_rec=r.val_recall))
        if verbose:
            print(f"  [deep r{rnd}] residual={int(resid.sum()):,}  detected "
                  f"block top-{top_k} ({detector})  -> rule depth {len(preds)} "
                  f"@P>={used_tp}  covers +{best_new:,} frauds  "
                  f"(valP={r.val_precision:.2f})")
    return deep_rules, info
