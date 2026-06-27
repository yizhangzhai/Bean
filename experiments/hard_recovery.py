"""Can the miner capture deep/disjunctive/banded/adversarial patterns?

For each planted pattern we mine (full histogram path, deep beam), then report:
  - feat: fraction of planted features recovered by the best rule
  - P/R : precision & recall of the best rule vs the pattern label (validation)
  - unionR: recall of the top-3 rules combined (matters for disjunctions)
  - decoy: did any decoy feature (40,41) sneak into the best rule?
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np

from arp.hard_data import make_hard, PATTERNS, planted_features
from arp.fast import fit_bins
from arp.targeted import targeted_beam_search

DECOYS = {40, 41}


def disjunct_feature_sets(pat):
    """Feature set of each DNF branch (a band's feature counts once)."""
    return [frozenset(cond[1] for cond in conj) for conj in pat["dnf"]]


def all_signal_features():
    s = set()
    for pat in PATTERNS:
        s |= planted_features(pat)
    return s


def peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if sys.platform == "darwin" else 1e6)


def _eval(preds, Xbin, yc, total_c):
    m = np.ones(Xbin.shape[0], dtype=bool)
    for f, op, k in preds:
        col = Xbin[:, f]
        m &= (col > k) if op == ">" else (col <= k)
    s = int(m.sum())
    tp = float(yc[m].sum())
    return m, (tp / s if s else 0.0), (tp / total_c if total_c else 0.0)


def run(n=200_000, n_features=200, n_bins=12, depth=6, beam=16, seed=0):
    t0 = time.perf_counter()
    X, Y, loss, names, tnames, planted = make_hard(n, n_features, seed)
    t_gen = time.perf_counter() - t0

    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]

    t0 = time.perf_counter()
    Xtr, spec = fit_bins(X[tr], n_bins=n_bins)
    Xva, _ = fit_bins(X[va], n_bins=n_bins)
    t_bin = time.perf_counter() - t0
    Ytr, Yva = Y[tr], Y[va]

    print(f"\n{'='*98}\nHARD PATTERN RECOVERY  n={n:,} F={n_features} bins={n_bins} "
          f"depth<={depth} beam={beam}\n{'='*98}")
    print(f"gen={t_gen:.1f}s  fit_bins(2x)={t_bin:.1f}s  "
          f"base rates={np.round(Y.mean(0)*100,3)} %")
    print("targeted search: min_recall=0.25, target_precision=0.5, gap_tol=0.20\n")
    print("(branches = DNF disjuncts covered by a CLEAN rule; clean = uses only "
          "that branch's true features)\n")
    print(f"{'pattern':18s} {'depth':>5s} {'#rule':>5s} {'branches':>9s} "
          f"{'bestP':>6s} {'bestR':>6s} {'clean':>6s}  status / best rule")
    print("-" * 132)
    noise_feats = lambda preds: {f for f, _, _ in preds} - all_signal_features() - DECOYS

    t0 = time.perf_counter()
    summary = []
    for c, pat in enumerate(PATTERNS):
        rules, trace = targeted_beam_search(
            Xtr, Ytr, c, spec, min_recall=0.25, target_precision=0.5,
            min_support=40, beam_width=beam, max_depth=depth,
            Xbin_val=Xva, Y_val=Yva, gap_tol=0.20)
        yc_va = Yva[:, c].astype(float)
        total_va = yc_va.sum()
        branches = disjunct_feature_sets(pat)

        if not rules:
            summary.append((pat["name"], "MISS"))
            print(f"{pat['name']:18s} {pat['depth']:>5d} {'0':>5s} {'0/'+str(len(branches)):>9s}"
                  f" {'-':>6s} {'-':>6s} {'-':>6s}  MISS (no rule meets targets)")
            continue

        # a branch is 'covered' iff some rule uses a superset of its true
        # features AND adds no noise/decoy feature (a genuinely clean recovery)
        def covers(bfeats):
            for r in rules:
                rf = {f for f, _, _ in r.preds}
                if bfeats <= rf and not (rf - all_signal_features()):
                    return True
            return False
        n_cov = sum(covers(b) for b in branches)

        best = max(rules, key=lambda r: r.val_precision * r.val_recall)
        clean = not (noise_feats(best.preds) or ({f for f, _, _ in best.preds} & DECOYS))
        status = ("OK" if n_cov == len(branches) else
                  "PARTIAL" if n_cov > 0 else "WEAK")
        summary.append((pat["name"], status))
        print(f"{pat['name']:18s} {best.depth:>5d} {len(rules):>5d} "
              f"{str(n_cov)+'/'+str(len(branches)):>9s} {best.val_precision:>6.2f} "
              f"{best.val_recall:>6.2f} {'yes' if clean else 'NO':>6s}  "
              f"{status}: {best.label(names, spec)}")
    t_mine = time.perf_counter() - t0

    print("-" * 132)
    ok = sum(s == "OK" for _, s in summary)
    par = sum(s == "PARTIAL" for _, s in summary)
    print(f"captured: {ok} full, {par} partial, "
          f"{len(summary)-ok-par} weak/miss  of {len(summary)} patterns")
    for name, s in summary:
        print(f"    {s:8s} {name}")
    print(f"\nmine ({len(PATTERNS)} patterns, depth<={depth}, beam={beam}): "
          f"{t_mine:.1f}s   peak RSS={peak_gb():.2f}GB   "
          f"total={t_gen+t_bin+t_mine:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--features", type=int, default=200)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--beam", type=int, default=16)
    a = ap.parse_args()
    run(a.n, a.features, a.bins, a.depth, a.beam)
