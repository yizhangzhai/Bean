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
                 top_k=16, seed_n=300, target_precision=0.7,
                 min_accept_precision=0.3, max_misses=8,
                 min_recall=0.01, min_support=40, beam_width=64, max_depth=18,
                 subsample=80_000, seed=0, detector="lgbm",
                 categorical=None, Xraw_tr=None, verbose=True):
    """Sequential-covering recovery of deep conjunctions on the residual.

    Each round: fit the detector on the still-uncovered frauds, ISOLATE the single
    dominant remaining pattern (highest-confidence positives -> KL-concentrated
    feature block), and run an F1 beam restricted to that block, RELAXING the
    precision target (`target_precision` -> `min_accept_precision`) until a rule
    covering new frauds is found. A rule is accepted only if its held-out
    precision clears `min_accept_precision` (so a tail-dominated residual yields a
    dragnet, not a captured pattern); on such a MISS the seed region is marked
    tried and the next round re-seeds elsewhere -- only after `max_misses`
    consecutive misses does covering stop. This keeps hunting real patterns in a
    residual swamped by an unmineable rare tail.

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
    hist_all = []                                  # overall bin dist (KL baseline)
    for f in range(F):
        h = np.bincount(np.asarray(Xtr[:, f]), minlength=nb).astype(float)
        hist_all.append(h / max(1.0, h.sum()))
    schedule, tp = [], target_precision
    while tp >= min_accept_precision - 1e-9:
        schedule.append(round(tp, 3))
        tp *= 0.7
    yv_col = (yva == 1).astype(np.int64).reshape(-1, 1)

    def find_rule(cols, resid):
        """Relax precision target; return best (preds,r,mask,new,tp) with held-out
        precision >= min_accept_precision, else None."""
        sub_spec = BinSpec([spec.edges[c] for c in cols], spec.pct, nb)
        Xt, Xv = Xtr[:, cols], Xva[:, cols]
        resid_col = resid.astype(np.int64).reshape(-1, 1)
        best = None
        for tp in schedule:
            rules, _ = targeted_beam_search(
                Xt, resid_col, 0, sub_spec, min_recall=min_recall,
                target_precision=tp, min_support=min_support,
                beam_width=beam_width, max_depth=max_depth,
                Xbin_val=Xv, Y_val=yv_col, gap_tol=None, rank_by="f1")
            for r in rules:
                if r.val_precision < min_accept_precision:
                    continue
                preds = tuple((cols[f], op, k) for f, op, k in r.preds)
                m = rule_mask(preds, Xtr)             # full-data coverage
                new = int((m & resid).sum())
                if best is None or new > best[3]:
                    best = (preds, r, m, new, tp)
            if best is not None and best[3] >= min_support:
                return best
        return best if (best and best[3] >= min_support) else None

    # Each round refits the detector on the current residual -- the fresh ranking
    # (model stochasticity + the shrinking residual) is what surfaces the NEXT
    # pattern, so this is NOT wasted work. On a miss the seed region is marked
    # tried and we re-seed; reset on a capture (residual changed).
    tried = np.zeros(len(ytr), dtype=bool)
    misses = 0
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
        Xsc = np.asarray(src[resid_idx])
        if Xraw_tr is None:
            Xsc = Xsc.astype(np.float32)
        scores = model.predict_proba(Xsc)[:, 1]
        avail = ~tried[resid_idx]
        if avail.sum() < min_support:
            break
        av_idx, av_sc = resid_idx[avail], scores[avail]
        seed_rows = av_idx[np.argsort(av_sc)[::-1][:min(seed_n, av_idx.size)]]
        cols = sorted(int(c) for c in _kl_block(Xtr, seed_rows, hist_all, nb, top_k))

        found = find_rule(cols, resid)
        if found is None:
            tried[seed_rows] = True
            misses += 1
            if verbose:
                print(f"  [deep r{rnd}] residual={int(resid.sum()):,}  miss "
                      f"({misses}/{max_misses}) -> re-seed")
            if misses >= max_misses:
                break
            continue
        misses = 0
        tried[:] = False
        preds, r, m, new, used_tp = found
        deep_rules.append(preds)
        covered |= m
        info.append(dict(round=rnd, depth=len(preds), new_covered=new,
                         val_prec=r.val_precision, val_rec=r.val_recall))
        if verbose:
            print(f"  [deep r{rnd}] residual={int(resid.sum()):,}  rule depth "
                  f"{len(preds)} @P>={used_tp}  covers +{new:,}  "
                  f"(valP={r.val_precision:.2f})")
    return deep_rules, info
