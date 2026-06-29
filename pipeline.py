"""End-to-end rule mining: data in -> interpretable rules out.

THE entry point. One call encodes the data and runs the sequential-covering deep
miner, returning human-readable conjunctive rules with held-out precision/recall.

    from pipeline import mine_rules
    rules, ev = mine_rules(X, y, categorical=[12, 13])
    for r in rules: print(r)

CLI:
    python pipeline.py --synthetic                       # demo on generated data
    python pipeline.py --data tx.csv --label target --categorical mcc,country
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from arp.fast import BinSpec, rule_mask
from arp.encode import quantile_edges, assign_bins, target_rank
from featgap import recover_deep, propose_features
from featgap.synthesize import FORMATS

N_BINS = 16
MISSING = N_BINS                       # reserved bin code for NaN (real bins 0..15)


# --------------------------------------------------------------------------- #
# encoding: numeric -> quantile bins; categorical -> positive-rate rank; NaN -> MISSING
# --------------------------------------------------------------------------- #
def _encode(Xtr, Xva, ytr, cat_set, sample, seed):
    F = Xtr.shape[1]
    btr = np.empty((F, Xtr.shape[0]), dtype=np.int8)
    bva = np.empty((F, Xva.shape[0]), dtype=np.int8)
    qs = np.arange(1, N_BINS) / N_BINS
    edges = [None] * F
    render = [None] * F                # per feature: ("num",) or ("cat", code->rank)
    rng = np.random.default_rng(seed)
    for f in range(F):
        ct, cv = Xtr[:, f], Xva[:, f]
        if f in cat_set:
            vm = ~np.isnan(ct)
            codes = ct[vm].astype(np.int64)
            rk = target_rank(codes, ytr[vm])               # Fisher-trick rank
            edges[f] = np.arange(1, N_BINS) - 0.5
            render[f] = ("cat", rk)
            tr_c = np.where(np.isnan(ct), 0, ct).astype(np.int64)
            va_c = np.where(np.isnan(cv), 0, cv).astype(np.int64)
            va_c = np.clip(va_c, 0, len(rk) - 1)           # unseen val codes -> 0
            btr[f] = np.where(np.isnan(ct), MISSING, rk[tr_c]).astype(np.int8)
            bva[f] = np.where(np.isnan(cv), MISSING, rk[va_c]).astype(np.int8)
        else:
            v = ct[~np.isnan(ct)]
            if v.size > sample:
                v = v[rng.integers(0, v.size, sample)]
            e = np.quantile(v, qs).astype(np.float64) if v.size else qs
            edges[f], render[f] = e, ("num",)
            bt = np.searchsorted(e, ct, side="right"); bt[np.isnan(ct)] = MISSING
            bv = np.searchsorted(e, cv, side="right"); bv[np.isnan(cv)] = MISSING
            btr[f], bva[f] = bt.astype(np.int8), bv.astype(np.int8)
    return btr.T, bva.T, BinSpec(edges, qs, N_BINS + 1), render


def _render_pred(f, op, k, render, names, spec=None):
    if render[f][0] == "num":
        pct = int(round((k + 1) / N_BINS * 100))
        return f"{names[f]} {op} p{pct:02d}"
    if render[f][0] == "cmp":                          # comparison: real threshold
        edges = spec.edges[f]
        thr = float(edges[min(k, len(edges) - 1)])
        tval = "0" if abs(thr) < 1e-9 else f"{thr:.4g}"
        return f"{render[f][1]} {op} {tval}"           # e.g.  "A - B > 0"  /  "(C-D)*A/B > 0.42"
    rk = render[f][1]                                  # code -> rank
    keep = [c for c, r in enumerate(rk) if (r > k if op == ">" else r <= k)]
    if op == ">" and k == MISSING - 1:                # isolates the missing bin
        return f"{names[f]} is MISSING"
    return f"{names[f]} in {{{','.join(map(str, sorted(keep)))}}}"


@dataclass
class Rule:
    text: str
    precision: float
    recall: float
    support: int
    preds: tuple
    def __str__(self):
        return (f"[P={self.precision:.2f} R={self.recall:.3f} n={self.support:,}]  "
                f"{self.text}")


# --------------------------------------------------------------------------- #
# feature engineering: diagnose the residual, synthesize features, append them
# --------------------------------------------------------------------------- #
def _residual_cols(Xtr_raw, resid, cat_set, n_cand):
    """Top-n numeric columns most separated on the residual (Cohen-d), to bound
    the O(F^2) pair enumeration the synthesizer does."""
    score = []
    for f in range(Xtr_raw.shape[1]):
        if f in cat_set:
            continue
        v = Xtr_raw[:, f]
        ok = ~np.isnan(v)
        a, b = v[ok & resid], v[ok & ~resid]
        if a.size < 5 or b.size < 5:
            continue
        sd = v[ok].std() + 1e-9
        score.append((abs(a.mean() - b.mean()) / sd, f))
    score.sort(reverse=True)
    return [f for _, f in score[:n_cand]]


def _bin_edges(v, qs, rng, *, snap_zero=False):
    """Quantile edges for a margin column; optionally snap the nearest edge to 0
    so a comparison (margin > 0) is an EXACT cut, when 0 lies inside the range."""
    e = quantile_edges(v, qs, sample=100_000, rng=rng)
    if snap_zero and v.min() < 0 < v.max():
        e[int(np.argmin(np.abs(e)))] = 0.0
        e = np.sort(e)
    return e


def _engineer(Xtr_b, Xva_b, spec, render, names, Xtr_raw, Xva_raw, ytr,
              base_rules, cat_set, seed, opts):
    """Second-pass feature engineering. Two sources, both appended as columns then
    re-mined:
      * synthesized features from the residual of `base_rules` (`formats`/`custom`
        in propose_features) -- binned by quantiles, rendered as `name op pXX`;
      * user `compare` relations -- each (label, fn) where fn(X)->margin (LHS-RHS,
        the relation holds when margin>0); binned with 0 as an exact cut, rendered
        with the REAL threshold (`label op value`), so `A > w*B` shows the w.

    Returns (Xtr_aug, Xva_aug, spec_aug, render_aug, names_aug, added, covered_tr);
    `added` lists the new feature labels, `covered_tr` is the base-rule TRAIN
    coverage (the start point for the second mine)."""
    covered_tr = np.zeros(len(ytr), dtype=bool)
    for preds in base_rules:
        covered_tr |= rule_mask(preds, Xtr_b)
    resid = (ytr == 1) & ~covered_tr

    qs = np.arange(1, N_BINS) / N_BINS
    rng = np.random.default_rng(seed + 11)
    Xt_all = np.nan_to_num(np.asarray(Xtr_raw, dtype=np.float64))   # full matrix for compare
    Xv_all = np.nan_to_num(np.asarray(Xva_raw, dtype=np.float64))
    new_tr, new_va, new_edges, new_render, added = [], [], [], [], []

    # 1) residual-driven synthesis (ratio / sum / linear / radial / periodic / custom)
    cols = opts.get("cols")
    if cols is None:
        cols = _residual_cols(Xtr_raw, resid, cat_set, opts.get("n_cand", 6))
    if cols and int(resid.sum()) >= 30 and opts.get("formats", FORMATS):
        sub = np.nan_to_num(Xtr_raw[:, cols].astype(np.float64))
        eng = propose_features(sub, resid, [names[c] for c in cols],
                               formats=opts.get("formats", FORMATS),
                               custom=opts.get("custom"),
                               max_features=opts.get("max_features", 4))
        full_tr, full_va = Xt_all[:, cols], Xv_all[:, cols]
        for c in eng:
            vt = np.nan_to_num(np.asarray(c["transform"](full_tr), dtype=np.float64))
            vv = np.nan_to_num(np.asarray(c["transform"](full_va), dtype=np.float64))
            e = _bin_edges(vt, qs, rng)
            new_tr.append(assign_bins(vt, e)[:, None]); new_va.append(assign_bins(vv, e)[:, None])
            new_edges.append(e); new_render.append(("num",)); added.append(c["name"])

    # 2) user comparison relations: A>B, A>w*B, A<(B+C+D)*w, (C-D)<(A-B), ...
    #    each fn(X) returns the margin LHS-RHS (relation holds when > 0)
    for label, fn in opts.get("compare", []):
        vt = np.nan_to_num(np.asarray(fn(Xt_all), dtype=np.float64))
        vv = np.nan_to_num(np.asarray(fn(Xv_all), dtype=np.float64))
        e = _bin_edges(vt, qs, rng, snap_zero=True)
        new_tr.append(assign_bins(vt, e)[:, None]); new_va.append(assign_bins(vv, e)[:, None])
        new_edges.append(e); new_render.append(("cmp", label)); added.append(label)

    if not new_tr:
        return Xtr_b, Xva_b, spec, render, names, [], covered_tr
    Xtr_aug = np.concatenate([np.asarray(Xtr_b)] + new_tr, axis=1)
    Xva_aug = np.concatenate([np.asarray(Xva_b)] + new_va, axis=1)
    spec_aug = BinSpec(list(spec.edges) + new_edges, spec.pct, spec.n_bins)
    names_aug = list(names) + added
    render_aug = list(render) + new_render
    return Xtr_aug, Xva_aug, spec_aug, render_aug, names_aug, added, covered_tr


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
def mine_rules(X, y, *, categorical=None, names=None, val_frac=0.33, seed=0,
               n_jobs=1, sample=100_000, serial=False, engineer=None,
               val_gap_tol=None, verbose=False, **deep_kwargs):
    """Mine interpretable rules. X: (n, F) float (NaN allowed). y: (n,) 0/1.
    `categorical`: column indices to treat as categorical. Returns (rules, eval)
    where eval = dict(recall, precision, flagged, positives, n_rules, val_idx, val_cov).

    Rule-level controls (pass as kwargs, forwarded to recover_deep):
      min_accept_precision  -- min held-out precision a rule must clear (default 0.12)
      min_recall            -- recall floor (default 0.004)
      min_support           -- min rows a rule must match (default 20)
      max_depth             -- max conditions per rule (default 18)
      target_precision      -- precision the search aims for (default 0.6)
      policy                -- arp.constraints.RulePolicy: feature usage, 1-/2-way
                               splits, forbidden / mutually-exclusive pairs, allowed
                               directions & threshold ranges, required-with (enforced
                               DURING the search).
      val_gap_tol           -- real-time held-out brake: stop growing any conjunction
                               whose train precision exceeds its validation precision
                               by more than this (e.g. 0.1). None = validate only at
                               acceptance (default).

    Mode flags:
      serial   -- force the fully-serial peel-one-pattern-at-a-time miner
                  (n_seeds=1, n_jobs=1): most accurate, slowest. Overrides n_jobs.
      engineer -- run a second feature-engineering pass. True for defaults, or a
                  dict to configure:
                    cols        candidate columns (default top-6 numeric by residual
                                separation)
                    formats     subset of A/B, A-B, A+B, w1*A+w2*B, radial, periodic
                                (featgap.synthesize.FORMATS); () to skip synthesis
                    custom      [(name, fn) ...] user-defined numeric transforms
                    compare     [(label, fn) ...] COMPARISON relations between feature
                                expressions -- fn(X)->margin (LHS-RHS; relation holds
                                when margin>0), evaluated on the FULL matrix by
                                original index. Handles A>B, A>w*B, A<(B+C+D)*w,
                                (C-D)<(A-B), (C-D)*A/B>w, etc. Binned with 0 as an
                                exact cut; rules render with the real threshold
                                (e.g. "A - B > 0", "(C-D)*A/B > 0.42" -- the w is the
                                discovered cut).
                    max_features, n_cand
                  Engineered + comparison features are appended and the residual
                  re-mined; synthesized rules render `name op pXX`, comparison rules
                  `label op value`."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int64)
    n, F = X.shape
    cat_set = set(categorical or [])
    names = names or [f"f{i}" for i in range(F)]

    perm = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - val_frac))
    tr, va = perm[:cut], perm[cut:]
    ytr, yva = y[tr], y[va]

    Xtr_b, Xva_b, spec, render = _encode(X[tr], X[va], ytr, cat_set, sample, seed + 7)

    npos = int((ytr == 1).sum())
    defaults = dict(max_rounds=30, top_k=22, seed_n=250, n_seeds=8,
                    target_precision=0.6, min_accept_precision=0.12, max_misses=2,
                    min_recall=0.004, min_support=20, beam_width=64, max_depth=18,
                    block_score="hybrid", val_gap_tol=val_gap_tol,
                    min_round_gain=max(40, npos // 100))    # relative early-stop
    defaults.update(deep_kwargs)
    if serial:                          # most accurate: one pattern at a time
        defaults["n_seeds"] = 1
        n_jobs = 1
    deep, info = recover_deep(Xtr_b, Xva_b, spec, ytr, yva,
                              np.zeros(len(ytr), dtype=bool),
                              categorical=list(cat_set), n_jobs=n_jobs,
                              seed=seed, verbose=verbose, **defaults)

    if engineer is not None and engineer is not False:
        opts = {} if engineer is True else dict(engineer)
        Xtr_b, Xva_b, spec, render, names, added, cov_tr = _engineer(
            Xtr_b, Xva_b, spec, render, names, X[tr], X[va], ytr, deep,
            cat_set, seed, opts)
        if added:                       # re-mine the residual with engineered cols
            deep2, _ = recover_deep(Xtr_b, Xva_b, spec, ytr, yva, cov_tr,
                                    categorical=list(cat_set), n_jobs=n_jobs,
                                    seed=seed + 1, verbose=verbose, **defaults)
            deep = list(deep) + list(deep2)
            if verbose:
                print(f"  [engineer] +{len(added)} features, +{len(deep2)} rules: "
                      + ", ".join(added))

    pos = yva == 1
    cov = np.zeros(len(yva), dtype=bool)
    rules = []
    for preds in deep:
        m = rule_mask(preds, Xva_b)
        cov |= m
        s = int(m.sum())
        tp = int((m & pos).sum())
        text = "  AND  ".join(_render_pred(f, op, k, render, names, spec) for f, op, k in preds)
        rules.append(Rule(text, tp / max(1, s), tp / max(1, int(pos.sum())), s, preds))
    rules.sort(key=lambda r: r.recall, reverse=True)
    flagged = int(cov.sum())
    ev = dict(recall=float((cov & pos).sum() / max(1, int(pos.sum()))),
              precision=float((cov & pos).sum() / max(1, flagged)),
              flagged=flagged, positives=int(pos.sum()), n_rules=len(rules),
              val_idx=va, val_cov=cov)        # for per-pattern / downstream analysis
    return rules, ev


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_csv(path, label, categorical):
    import pandas as pd
    df = pd.read_csv(path)
    y = df[label].to_numpy()
    feat = [c for c in df.columns if c != label]
    cats = [c.strip() for c in categorical.split(",")] if categorical else []
    cat_idx = [feat.index(c) for c in cats if c in feat]
    X = df[feat].apply(lambda s: s.astype("category").cat.codes
                       if s.name in cats else s).to_numpy(dtype=np.float32)
    return X, y, cat_idx, feat


def main():
    ap = argparse.ArgumentParser(description="Mine interpretable rules.")
    ap.add_argument("--data", help="CSV file of features + label")
    ap.add_argument("--label", default="label", help="label column name")
    ap.add_argument("--categorical", default="", help="comma-separated categorical columns")
    ap.add_argument("--synthetic", action="store_true", help="run on generated data")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--features", type=int, default=120)
    ap.add_argument("--patterns", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--top", type=int, default=20, help="print top-N rules")
    args = ap.parse_args()

    if args.data:
        X, y, cat_idx, names = _load_csv(args.data, args.label, args.categorical)
    else:                                              # --synthetic (default fallback)
        from synth import make_data
        fs = make_data(args.n, args.features, args.patterns)
        X, y, cat_idx, names = fs.X, fs.y, fs.categorical, fs.names

    print(f"data: {X.shape[0]:,} rows x {X.shape[1]} features, "
          f"{int(y.sum()):,} positive ({100*y.mean():.2f}%), {len(cat_idx)} categorical")
    rules, ev = mine_rules(X, y, categorical=cat_idx, names=names,
                           n_jobs=args.jobs, verbose=True)
    print(f"\n{ev['n_rules']} rules  ->  recall={ev['recall']:.2f}  "
          f"precision={ev['precision']:.2f}  flagged={ev['flagged']:,}\n")
    for r in rules[:args.top]:
        print(" ", r)


if __name__ == "__main__":
    main()
