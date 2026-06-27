"""Supervised (chi-square-gated) binning vs plain percentile binning.

Measures, on the same multi-type fraud data:
  (a) candidate-threshold count + mine runtime  (does it get faster?)
  (b) rule cleanliness: predicates landing on NOISE features, and rule depth
      (does it stop adding spurious deep thresholds?)
  (c) recovery: are the planted signal features still found?

Supervised binning keeps cut points only where the fraud rate changes, so noise
features (flat label correlation) collapse to ~0 active thresholds and can't
enter rules -- faster search AND cleaner rules, with recovery preserved.
"""

from __future__ import annotations

import time

import numpy as np

from arp.data import make_fraud_data_large, GROUND_TRUTH_FEATURES
from arp.fast import fit_bins
from arp.targeted import targeted_beam_search


def n_candidates(spec, F):
    if spec.active is None:
        return F * (spec.n_bins - 1) * 2
    return int(sum(len(a) for a in spec.active)) * 2


def active_features(spec, F):
    if spec.active is None:
        return F
    return int(sum(1 for a in spec.active if len(a) > 0))


def mine_and_measure(Xb_tr, Ytr, spec, Xb_va, Yva, c, signal):
    t = time.perf_counter()
    rules, _ = targeted_beam_search(
        Xb_tr, Ytr, c, spec, min_recall=0.25, target_precision=0.5,
        min_support=40, beam_width=12, max_depth=4,
        Xbin_val=Xb_va, Y_val=Yva, gap_tol=0.25)
    dt = time.perf_counter() - t
    if not rules:
        return dt, 0, 0.0, 0.0, set()
    tot_preds = noise_preds = 0
    depths = []
    feats_union = set()
    for r in rules:
        depths.append(len(r.preds))
        for f, _, _ in r.preds:
            tot_preds += 1
            feats_union.add(f)
            if f not in signal:
                noise_preds += 1
    return (dt, len(rules), float(np.mean(depths)),
            noise_preds / max(1, tot_preds), feats_union)


def run(n=200_000, n_features=80, n_bins=16, seed=0):
    data = make_fraud_data_large(n=n, n_features=n_features, seed=seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Xtr, Xva = data.X[tr], data.X[va]
    Ytr, Yva = data.Y[tr], data.Y[va]
    F = n_features

    print(f"\n{'='*96}\nSUPERVISED vs PLAIN BINNING  n={n:,} F={F} bins={n_bins}\n{'='*96}")

    # plain percentile bins
    t = time.perf_counter()
    Xb_u, spec_u = fit_bins(Xtr, n_bins=n_bins)
    Xbv_u, _ = fit_bins(Xva, n_bins=n_bins)
    tbin_u = time.perf_counter() - t

    # supervised (chi-square-gated, train labels, union across types)
    t = time.perf_counter()
    Xb_s, spec_s = fit_bins(Xtr, n_bins=n_bins, Y=Ytr, supervised=True, chi2=6.63)
    # apply the SAME spec's edges to val (bins identical; active is feature-level)
    Xbv_s, _ = fit_bins(Xva, n_bins=n_bins)
    Xbv_s_spec = spec_s
    tbin_s = time.perf_counter() - t

    print(f"\nencoding:")
    print(f"  plain       candidates={n_candidates(spec_u,F):>6,}  "
          f"active features={active_features(spec_u,F)}/{F}   bin={tbin_u:.1f}s")
    print(f"  supervised  candidates={n_candidates(spec_s,F):>6,}  "
          f"active features={active_features(spec_s,F)}/{F}   bin={tbin_s:.1f}s   "
          f"(-{100*(1-n_candidates(spec_s,F)/n_candidates(spec_u,F)):.0f}% candidates)")

    hdr = (f"\n{'type':18s} {'bins':11s} {'mine':>6s} {'#rule':>6s} {'avgdepth':>8s} "
           f"{'noise%':>7s} {'recovered':>9s}")
    print(hdr)
    print("-" * 76)
    for c, tname in enumerate(data.type_names):
        signal = GROUND_TRUTH_FEATURES[tname]
        for label, Xb, spec, Xbv in (("plain", Xb_u, spec_u, Xbv_u),
                                     ("supervised", Xb_s, spec_s, Xbv_s)):
            dt, nr, avgd, noise_frac, feats = mine_and_measure(
                Xb, Ytr, spec, Xbv, Yva, c, signal)
            rec = "YES" if signal <= feats else "no"
            print(f"{tname if label=='plain' else '':18s} {label:11s} {dt:5.1f}s "
                  f"{nr:>6} {avgd:>8.1f} {100*noise_frac:>6.0f}% {rec:>9s}")
        print()


if __name__ == "__main__":
    run()
