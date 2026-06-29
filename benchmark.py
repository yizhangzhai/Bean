"""One benchmark runner for the whole project (replaces the old experiments/).

Subcommands, each self-contained and parameterised:

    python benchmark.py capture     [n F patterns]   # multi-pattern capture quality
    python benchmark.py scale       [n F]            # encode + mine timing
    python benchmark.py deep        [n F]            # deep-conjunction recovery
    python benchmark.py featgap     [n F]            # non-axis -> feature engineering
    python benchmark.py categorical [n F]            # code-bin vs target-rank subsets
"""

from __future__ import annotations

import sys
import time
import platform
import resource

import numpy as np

from synth import make_data
from pipeline import mine_rules, _encode, N_BINS
from arp.fast import rule_mask, BinSpec
from arp.encode import quantile_edges, assign_bins
from featgap import recover_deep, propose_features


def peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if platform.system() == "Darwin" else 1e6)


def _percov(ev, patterns):
    """Per-pattern coverage on the validation split."""
    va, cov = ev["val_idx"], ev["val_cov"]
    out = []
    for p in patterns:
        m = p["mask"][va]
        out.append(float((cov & m).sum() / max(1, int(m.sum()))))
    return out


# --------------------------------------------------------------------------- #
def capture(n=400_000, F=200, patterns=100):
    print(f"\nCAPTURE  n={n:,} F={F} patterns={patterns}  (2% positive rate, full corner-case mess)")
    t0 = time.perf_counter()
    fs = make_data(n, F, patterns, bad_rate=0.02, seed=0)
    print(f"  positives={int(fs.y.sum()):,} ({100*fs.y.mean():.2f}%)  mineable(>=150 cases)="
          f"{len(fs.mineable)}/{patterns}  missing={int(np.isnan(fs.X).sum()):,}")
    rules, ev = mine_rules(fs.X, fs.y, categorical=fs.categorical, names=fs.names,
                           n_jobs=6, verbose=True)
    pc = _percov(ev, fs.patterns)
    cap = sum(c >= 0.5 for c in pc)
    capfr = sum(p["realized"] for p, c in zip(fs.patterns, pc) if c >= 0.5)
    print(f"\n  {ev['n_rules']} rules  recall={ev['recall']:.2f}  precision={ev['precision']:.2f}")
    print(f"  CAPTURED {cap}/{patterns} patterns "
          f"({capfr/max(1,sum(p['realized'] for p in fs.patterns)):.0%} of all positives)")
    for lo, hi, lab in [(150, 400, "150-400"), (400, 1500, "400-1500"), (1500, 9e9, ">=1500")]:
        ids = [i for i, p in enumerate(fs.patterns) if lo <= p["realized"] < hi]
        if ids:
            print(f"    size {lab:9s}: {sum(pc[i]>=0.5 for i in ids)}/{len(ids)} captured")
    print(f"  TOTAL {time.perf_counter()-t0:.0f}s  peak {peak_gb():.2f} GB")


# --------------------------------------------------------------------------- #
def scale(n=500_000, F=300):
    print(f"\nSCALE  n={n:,} F={F}  (encode + mine timing)")
    fs = make_data(n, F, 60, bad_rate=0.02, seed=0)
    t0 = time.perf_counter()
    rules, ev = mine_rules(fs.X, fs.y, categorical=fs.categorical, n_jobs=6, verbose=True)
    dt = time.perf_counter() - t0
    print(f"  {n:,}x{F}: {ev['n_rules']} rules  recall={ev['recall']:.2f}  "
          f"[{dt:.0f}s, {n/dt/1e3:.0f}K rows/s]  peak {peak_gb():.2f} GB")


# --------------------------------------------------------------------------- #
def deep(n=200_000, F=120):
    """Plant deep conjunctions (depth 5/8/11) and check exact recovery."""
    print(f"\nDEEP  n={n:,} F={F}  (deep-conjunction recovery)")
    fs = make_data(n, F, 30, bad_rate=0.03, seed=2)
    rules, ev = mine_rules(fs.X, fs.y, categorical=fs.categorical, names=fs.names,
                           n_jobs=6, verbose=True)
    pc = _percov(ev, fs.patterns)
    print(f"\n  {ev['n_rules']} rules  recall={ev['recall']:.2f}  precision={ev['precision']:.2f}")
    for d_lo, d_hi, lab in [(2, 5, "2-4"), (5, 9, "5-8"), (9, 99, "9+")]:
        ids = [i for i, p in enumerate(fs.patterns)
               if p["kind"] == "axis" and d_lo <= p["depth"] < d_hi and p["realized"] >= 150]
        if ids:
            print(f"    depth {lab:4s}: {sum(pc[i]>=0.5 for i in ids)}/{len(ids)} captured")
    print("  deepest recovered rule:")
    print("   ", max(rules, key=lambda r: len(r.preds)))


# --------------------------------------------------------------------------- #
def featgap(n=300_000, F=120):
    """Non-axis patterns (ring/ratio/periodic) -> rules miss them -> featgap
    synthesizes features -> re-mine captures them."""
    print(f"\nFEATGAP  n={n:,} F={F}  (non-axis -> feature engineering)")
    fs = make_data(n, F, 30, bad_rate=0.03, nonaxis=True, missing=0.0, seed=0)
    cand = [0, 1, 2, 3, 4]
    rng = np.random.default_rng(1)
    perm = rng.permutation(n); cut = int(n * 0.67); tr, va = perm[:cut], perm[cut:]
    ytr, yva = fs.y[tr], fs.y[va]
    Xtr_b, Xva_b, spec, _ = _encode(fs.X[tr], fs.X[va], ytr, set(fs.categorical), 100_000, 7)

    def cov_by_kind(preds_list, Xb, idx):
        cov = np.zeros(len(idx), dtype=bool)
        for pr in preds_list:
            cov |= rule_mask(pr, Xb)
        return {k: float((cov & fs.patterns[p]["mask"][idx]).sum() /
                         max(1, fs.patterns[p]["mask"][idx].sum()))
                for p, k in enumerate(["ring", "ratio", "periodic"])}

    deep1, _ = recover_deep(Xtr_b, Xva_b, spec, ytr, yva, np.zeros(len(ytr), bool),
                            max_rounds=20, n_seeds=8, n_jobs=6, min_round_gain=120,
                            min_support=20, min_accept_precision=0.12, seed=0, verbose=False)
    print("  stage 1 (raw bins):       ", cov_by_kind(deep1, Xva_b, va))

    cov_tr = np.zeros(len(ytr), bool)
    for pr in deep1:
        cov_tr |= rule_mask(pr, Xtr_b)
    cands = propose_features(fs.X[tr][:, cand], (ytr == 1) & ~cov_tr,
                             [f"f{c}" for c in cand], max_features=4)
    print("  featgap proposes:          " +
          ", ".join(f"{c['name']}(lift {c['lift']:.1f})" for c in cands[:3]))
    qs = np.arange(1, N_BINS) / N_BINS
    atr, ava, edg = [], [], []
    for c in cands[:4]:
        v = c["transform"](fs.X[:, cand])
        e = quantile_edges(v[tr], qs, sample=100_000, rng=np.random.default_rng(9))
        atr.append(assign_bins(v[tr], e)); ava.append(assign_bins(v[va], e)); edg.append(e)
    Xt2 = np.concatenate([np.asarray(Xtr_b)] + [a[:, None] for a in atr], axis=1)
    Xv2 = np.concatenate([np.asarray(Xva_b)] + [a[:, None] for a in ava], axis=1)
    sp2 = BinSpec(list(spec.edges) + edg, spec.pct, spec.n_bins)
    deep2, _ = recover_deep(Xt2, Xv2, sp2, ytr, yva, np.zeros(len(ytr), bool),
                            max_rounds=20, n_seeds=8, n_jobs=6, min_round_gain=120,
                            min_support=20, min_accept_precision=0.12, seed=0, verbose=False)
    print("  stage 2 (+ engineered):   ", cov_by_kind(deep2, Xv2, va))


# --------------------------------------------------------------------------- #
def categorical(n=200_000, F=60):
    """code-bin vs target-rank on categorical subset patterns."""
    print(f"\nCATEGORICAL  n={n:,} F={F}  (code-bin vs target-rank subset rules)")
    fs = make_data(n, F, 25, bad_rate=0.03, missing=0.0, seed=0)
    sub = [p for p in fs.patterns if p["kind"] == "axis" and p.get("cat") and p["realized"] >= 150]
    print(f"  {len(sub)} categorical patterns >=150 cases")
    for cat_rank, tag in [(False, "as-numeric"), (True, "target-rank")]:
        rules, ev = mine_rules(fs.X, fs.y, categorical=fs.categorical if cat_rank else [],
                               names=fs.names, n_jobs=6, verbose=False)
        pc = _percov(ev, fs.patterns)
        cap = sum(pc[fs.patterns.index(p)] >= 0.5 for p in sub)
        print(f"  {tag:12s}: {ev['n_rules']:>3} rules  recall={ev['recall']:.2f}  "
              f"cat-subset captured {cap}/{len(sub)}")


CMDS = {"capture": capture, "scale": scale, "deep": deep,
        "featgap": featgap, "categorical": categorical}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print("usage: python benchmark.py {capture|scale|deep|featgap|categorical} [args]")
        sys.exit(1)
    cmd = sys.argv[1]
    args = [int(a) for a in sys.argv[2:]]
    CMDS[cmd](*args)
