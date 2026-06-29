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
from featgap import recover_deep

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


def _render_pred(f, op, k, render, names):
    if render[f][0] == "num":
        pct = int(round((k + 1) / N_BINS * 100))
        return f"{names[f]} {op} p{pct:02d}"
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
# the pipeline
# --------------------------------------------------------------------------- #
def mine_rules(X, y, *, categorical=None, names=None, val_frac=0.33, seed=0,
               n_jobs=1, sample=100_000, verbose=False, **deep_kwargs):
    """Mine interpretable rules. X: (n, F) float (NaN allowed). y: (n,) 0/1.
    `categorical`: column indices to treat as categorical. Returns (rules, eval)
    where eval = dict(recall, precision, flagged, positives)."""
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
                    block_score="hybrid",
                    min_round_gain=max(40, npos // 100))    # relative early-stop
    defaults.update(deep_kwargs)
    deep, info = recover_deep(Xtr_b, Xva_b, spec, ytr, yva,
                              np.zeros(len(ytr), dtype=bool),
                              categorical=list(cat_set), n_jobs=n_jobs,
                              seed=seed, verbose=verbose, **defaults)

    pos = yva == 1
    cov = np.zeros(len(yva), dtype=bool)
    rules = []
    for preds in deep:
        m = rule_mask(preds, Xva_b)
        cov |= m
        s = int(m.sum())
        tp = int((m & pos).sum())
        text = "  AND  ".join(_render_pred(f, op, k, render, names) for f, op, k in preds)
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
