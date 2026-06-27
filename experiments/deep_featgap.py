"""Can the auto feature-engineering layer boost recall on DEEP axis conjunctions?

deep_bench showed rule mining alone catches ~11% of bads when the signatures are
deep AND-of-weak-conditions (depths 8/11/15 essentially uncovered). Here we throw
the feature-engineering toolkit at the residual and measure what each adds:

  0. baseline            rule mining only
  1. featgap GEOMETRIC   propose_features (radial / ratio / diff / periodic) + interaction screen
  2. supervised PROJECTION  L1-logistic on the residual -> one signed-sum "risk
                            direction" feature (a soft AND; still inspectable)
  3. boosted SCORE          HistGradientBoosting on the residual -> one score
                            feature (captures deep AND + the OR of patterns)

Each engineered feature is binned and the rules are RE-MINED at the same operating
point (P>=0.6, R>=0.03), so the comparison is apples-to-apples.

Expectation worth testing: featgap's geometric transforms were built for NON-axis
structure (ring/ratio/periodic) -- a deep axis AND is neither, so they should add
little; a SUPERVISED feature that aggregates many weak conditions is what moves
recall, at some interpretability cost.

Run:  python -m experiments.deep_featgap [small|large]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from arp.encode import encode_split_cm, quantile_edges, assign_bins
from arp.fast import rule_mask, BinSpec
from arp.targeted import targeted_beam_search
from featgap import uncovered_positives, propose_features, interaction_screen

from experiments.deep_bench import make_deep, DEPTHS, N_BINS

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


def mine(Xtr, Xva, spec, ytr, yva, min_support):
    rules, _ = targeted_beam_search(
        Xtr, ytr.reshape(-1, 1), 0, spec, min_recall=0.03, target_precision=0.6,
        min_support=min_support, beam_width=48, max_depth=16,
        Xbin_val=Xva, Y_val=yva.reshape(-1, 1), gap_tol=None)
    return rules


def bin_feature(v_tr, v_va, seed):
    qs = np.arange(1, N_BINS) / N_BINS
    e = quantile_edges(v_tr, qs, sample=100_000, rng=np.random.default_rng(seed))
    return assign_bins(v_tr, e), assign_bins(v_va, e), e


def stack(Xtr, Xva, spec, cols_tr, cols_va, edges):
    Xtr2 = np.concatenate([np.asarray(Xtr)] + [c[:, None] for c in cols_tr], axis=1)
    Xva2 = np.concatenate([np.asarray(Xva)] + [c[:, None] for c in cols_va], axis=1)
    spec2 = BinSpec(list(spec.edges) + list(edges), spec.pct, spec.n_bins)
    return Xtr2, Xva2, spec2


def report(tag, rec, prec, pm, flagged, n_va, dt):
    pms = "  ".join(f"d{d}={pm[p]:.2f}" for p, d in enumerate(DEPTHS))
    print(f"  {tag:24s} recall={rec:.2f}  precision={prec:.2f}  "
          f"flagged={flagged:>6,}/{n_va:,}  [{dt:.0f}s]\n"
          f"  {'':24s} per-pattern: {pms}")


def run(n, n_features, seed=0):
    print(f"\n{'='*92}\nAUTO-FE TO BOOST RECALL ON DEEP CONJUNCTIONS  "
          f"n={n:,}  F={n_features}  depths={DEPTHS}\n{'='*92}")
    S, y, mo, planted, names, tr, va, n_sig = make_deep(n, n_features, seed)
    ytr, yva = y[tr], y[va]
    mo_va = [m[va] for m in mo]
    n_va = len(va)
    min_support = max(40, n // 4000)

    def make_column(f):
        if f < n_sig:
            return S[:, f]
        return np.random.default_rng(seed * 100003 + f).standard_normal(n).astype(np.float32)

    Xtr, Xva, spec = encode_split_cm(make_column, n_features, tr, va,
                                     n_bins=N_BINS, sample=100_000, seed=seed + 7)

    # ---- 0. baseline ----
    t0 = time.perf_counter()
    base = mine(Xtr, Xva, spec, ytr, yva, min_support)
    rec, prec, pm, fl = coverage(base, Xva, yva, mo_va)
    report("0. baseline (rules only)", rec, prec, pm, fl, n_va, time.perf_counter() - t0)

    # residual (uncovered train frauds)
    gap_tr, cov_tr = uncovered_positives(base, Xtr, ytr)
    print(f"  residual: {int(gap_tr.sum()):,} uncovered train frauds "
          f"({100*gap_tr.sum()/max(1,(ytr==1).sum()):.0f}% of train fraud)")

    # ---- 1. featgap GEOMETRIC (propose_features + interaction screen) ----
    t0 = time.perf_counter()
    keep = ~((ytr == 1) & cov_tr)
    pairs, _ = interaction_screen(np.asarray(Xtr[:, :n_sig], dtype=float),
                                  gap_tr.astype(int), mask=keep, bins=8, top_k=3)
    cands = propose_features(S[tr], gap_tr, names[:n_sig], max_features=3)
    print(f"  [featgap] top interacting pairs (synergy): " +
          ", ".join(f"({names[i]},{names[j]}):{s:+.3f}" for i, j, s, *_ in pairs))
    print(f"  [featgap] proposed transforms: " +
          ", ".join(f"{c['name']}(lift={c['lift']:.1f})" for c in cands))
    cols_tr, cols_va, edges = [], [], []
    for ci, c in enumerate(cands):
        v = c["transform"](S)
        bt, bv, e = bin_feature(v[tr], v[va], seed + 100 + ci)
        cols_tr.append(bt); cols_va.append(bv); edges.append(e)
    Xt2, Xv2, sp2 = stack(Xtr, Xva, spec, cols_tr, cols_va, edges)
    g = mine(Xt2, Xv2, sp2, ytr, yva, min_support)
    rec, prec, pm, fl = coverage(g, Xv2, yva, mo_va)
    report("1. + featgap geometric", rec, prec, pm, fl, n_va, time.perf_counter() - t0)

    # ---- 2. supervised PROJECTION (L1-logistic on residual) ----
    from sklearn.linear_model import LogisticRegression
    t0 = time.perf_counter()
    Xfit = np.asarray(Xtr)[keep].astype(np.float32)
    yfit = gap_tr[keep].astype(int)
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=200)
    lr.fit(Xfit, yfit)
    nz = int((lr.coef_ != 0).sum())
    s_tr = lr.decision_function(np.asarray(Xtr).astype(np.float32))
    s_va = lr.decision_function(np.asarray(Xva).astype(np.float32))
    bt, bv, e = bin_feature(s_tr, s_va, seed + 200)
    Xt2, Xv2, sp2 = stack(Xtr, Xva, spec, [bt], [bv], [e])
    p = mine(Xt2, Xv2, sp2, ytr, yva, min_support)
    rec, prec, pm, fl = coverage(p, Xv2, yva, mo_va)
    report(f"2. + L1-logistic proj ({nz} feats)", rec, prec, pm, fl, n_va,
           time.perf_counter() - t0)

    # ---- 3. boosted SCORE (HistGradientBoosting on residual) ----
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.perf_counter()
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                         max_depth=None, max_leaf_nodes=63)
    hgb.fit(Xfit, yfit)
    s_tr = hgb.predict_proba(np.asarray(Xtr).astype(np.float32))[:, 1]
    s_va = hgb.predict_proba(np.asarray(Xva).astype(np.float32))[:, 1]
    bt, bv, e = bin_feature(s_tr, s_va, seed + 300)
    Xt2, Xv2, sp2 = stack(Xtr, Xva, spec, [bt], [bv], [e])
    h = mine(Xt2, Xv2, sp2, ytr, yva, min_support)
    rec, prec, pm, fl = coverage(h, Xv2, yva, mo_va)
    report("3. + HGB boosted score", rec, prec, pm, fl, n_va, time.perf_counter() - t0)
    print("\n  geometric transforms target NON-axis structure -> little here;")
    print("  a supervised feature that AGGREGATES many weak conditions is what")
    print("  moves recall (projection = soft AND, interpretable; boosting = full).")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    run(*SCALES[which])
