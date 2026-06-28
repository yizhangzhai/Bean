"""featgap.deep -- recover DEEP conjunctions that greedy beam search can't seed.

The diagnosis (experiments/deep_diagnose.py): a deep AND-of-weak-conditions is a
discovery wall for a marginal beam -- every prefix is a statistical twin of a
random broad rule, so no single-rule score (precision, recall, F1) can single it
out, and the one steep shallow pattern monopolizes the beam.

The signal IS there, but only JOINTLY. So we detect the relevant feature block
with a joint model (a gradient-boosted tree on the residual) and then REFINE: run
the rule search restricted to that handful of features, where the conjunction is
no longer drowned. Sequential covering peels one pattern at a time.

    detect (joint model) -> ISOLATE one pattern -> restrict -> refine -> cover

ISOLATION matters when many overlapping patterns share a feature pool: a plain
top-k importance list is then a BLEND of several patterns and the refiner builds
hybrids. Instead we score the residual with the detector, take its highest-
confidence positives as a seed (they belong to the single dominant remaining
pattern), and choose the block by which features are most CONCENTRATED in that
seed (KL divergence of their bin distribution vs the overall) -- a one-pattern
block. Each round also relaxes the precision target so a contaminated pattern
still yields a rule.

The detector is the SCOPE oracle, not the rule source (its own tree paths are
noisy/fragmented); the interpretable, policy-eligible rule is produced by the
refine step. Default detector is LightGBM -- histogram-based so it scales to high
dimension, handles missing values natively, and takes categorical features
directly -- with a scikit-learn RandomForest fallback.

Depends on `arp` (one-directional); LightGBM / scikit-learn are optional.
"""

from __future__ import annotations

import numpy as np

from arp.fast import rule_mask, BinSpec
from arp.targeted import targeted_beam_search


def _fit_detector(Xfit, y, *, seed, detector, categorical):
    """Fit the joint detector on (Xfit, y); returns a model with predict_proba."""
    if detector == "lgbm":
        try:
            import lightgbm as lgb
            m = lgb.LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                class_weight="balanced", n_jobs=-1, random_state=seed,
                verbosity=-1)
            fk = {"categorical_feature": list(categorical)} if categorical else {}
            m.fit(Xfit, y, **fk)
            return m
        except ImportError:
            pass
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=seed,
                               max_features="sqrt", class_weight="balanced")
    m.fit(np.nan_to_num(np.asarray(Xfit, dtype=np.float32)), y)
    return m


def _kl_block(Xtr, seed_rows, hist_all, n_bins, top_k, eps=1e-9):
    """Block = features whose bin distribution among the seed rows diverges most
    (KL) from their overall distribution -- i.e. the features the one dominant
    pattern actually constrains. Returns original-index columns."""
    F = Xtr.shape[1]
    kl = np.empty(F)
    for f in range(F):
        h = np.bincount(np.asarray(Xtr[seed_rows, f]), minlength=n_bins).astype(float)
        s = h.sum()
        if s == 0:
            kl[f] = 0.0
            continue
        h /= s
        ha = hist_all[f]
        nz = h > 0
        kl[f] = float(np.sum(h[nz] * np.log((h[nz] + eps) / (ha[nz] + eps))))
    return np.argsort(kl)[::-1][:top_k]


def recover_deep(Xtr, Xva, spec, ytr, yva, covered_tr, *, max_rounds=6,
                 top_k=16, seed_n=300, precision_floor=0.2, target_precision=0.7,
                 min_recall=0.01, min_support=40, beam_width=64, max_depth=18,
                 subsample=80_000, seed=0, detector="lgbm", categorical=None,
                 Xraw_tr=None, verbose=True):
    """Sequential-covering recovery of deep conjunctions on the residual.

    Each round: fit the detector on the still-uncovered frauds, ISOLATE the single
    dominant remaining pattern (highest-confidence positives -> KL-concentrated
    feature block), and run an F1 beam restricted to that block, RELAXING the
    precision target (`target_precision` -> `precision_floor`) until a rule
    covering new frauds appears. Accept it, subtract its coverage, repeat.

    detector: "lgbm" (default, scalable + native missing/categorical) or "rf".
    Xraw_tr: optional raw (unbinned) matrix for the DETECTOR only (lets LightGBM
      exploit real NaN / categorical); the refine always runs on binned `Xtr`.
    categorical: detector column indices to treat as categorical (LightGBM).

    Returns (deep_rules, info); predicate feature indices are in the ORIGINAL
    space, ready for arp.fast.rule_mask.
    """
    covered = covered_tr.copy()
    deep_rules, info = [], []
    src = Xraw_tr if Xraw_tr is not None else Xtr
    nb = spec.n_bins
    F = Xtr.shape[1]
    # overall per-feature bin distribution (KL baseline), computed once
    hist_all = []
    for f in range(F):
        h = np.bincount(np.asarray(Xtr[:, f]), minlength=nb).astype(float)
        hist_all.append(h / max(1.0, h.sum()))

    for rnd in range(max_rounds):
        resid = (ytr == 1) & ~covered
        resid_idx = np.flatnonzero(resid)
        if resid_idx.size < min_support:
            break
        keep_idx = np.flatnonzero(~((ytr == 1) & covered))
        rng = np.random.default_rng(seed + rnd)
        sub = rng.choice(keep_idx, min(subsample, len(keep_idx)), replace=False)
        Xfit = np.asarray(src[sub])
        if Xraw_tr is None:
            Xfit = Xfit.astype(np.float32)
        model = _fit_detector(Xfit, resid[sub].astype(int), seed=seed + rnd,
                              detector=detector, categorical=categorical)

        # isolate the dominant pattern: seed = top-scoring uncovered frauds
        Xsc = np.asarray(src[resid_idx])
        if Xraw_tr is None:
            Xsc = Xsc.astype(np.float32)
        scores = model.predict_proba(Xsc)[:, 1]
        seed_rows = resid_idx[np.argsort(scores)[::-1][:min(seed_n, resid_idx.size)]]
        cols = sorted(int(c) for c in _kl_block(Xtr, seed_rows, hist_all, nb, top_k))

        sub_spec = BinSpec([spec.edges[c] for c in cols], spec.pct, nb)
        Xt, Xv = Xtr[:, cols], Xva[:, cols]
        yv_col = (yva == 1).astype(np.int64).reshape(-1, 1)
        resid_col = resid.astype(np.int64).reshape(-1, 1)

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
                print(f"  [deep r{rnd}] residual={int(resid.sum()):,}  isolated "
                      f"block {cols[:6]}...  no rule (down to P={schedule[-1]}) -> stop")
            break
        preds, r, m = best
        deep_rules.append(preds)
        covered |= m
        info.append(dict(round=rnd, depth=len(preds), new_covered=best_new,
                         val_prec=r.val_precision, val_rec=r.val_recall))
        if verbose:
            print(f"  [deep r{rnd}] residual={int(resid.sum()):,}  isolated "
                  f"{top_k}-feat block -> rule depth {len(preds)} @P>={used_tp}  "
                  f"covers +{best_new:,}  (valP={r.val_precision:.2f})")
    return deep_rules, info
