"""Gap-driven feature engineering, end to end.

One fraud label fired by EITHER an axis-aligned pattern OR a RING:
    fraud if (5 < f2 < 8 AND f3 > 15)            # axis-aligned, miner can cut it
          or (2 < dist((f0,f1),(10,10)) < 4)     # a ring -- NO box can cover it

So the base miner covers the axis frauds and MISSES the ring (the gap). We then
diagnose the gap's geometry (ring/void) and synthesize a radial feature; after
re-mining, the ring rule appears and coverage closes.
"""

from __future__ import annotations

import time

import numpy as np

from arp.fast import fit_bins, rule_mask
from arp.targeted import targeted_beam_search
from arp.gapfill import uncovered_positives, propose_features

CENTER = (10.0, 10.0)


def make(n, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 20, size=(n, n_features)).astype(np.float32)
    r = np.hypot(X[:, 0] - CENTER[0], X[:, 1] - CENTER[1])
    axis = (X[:, 2] > 5) & (X[:, 2] < 8) & (X[:, 3] > 15)
    ring = (r > 2) & (r < 4)
    member = axis | ring
    p = np.where(member, 0.85, 0.004)
    y = (rng.uniform(size=n) < p).astype(np.int64)
    return X, y, ring


def apply_bins(X, spec):
    Xb = np.empty(X.shape, dtype=np.int8)
    for f in range(X.shape[1]):
        Xb[:, f] = np.searchsorted(spec.edges[f], X[:, f], side="right").astype(np.int8)
    return Xb


def mine(Xb_tr, y_tr, spec, Xb_va, y_va):
    Y = y_tr.reshape(-1, 1)
    Yv = y_va.reshape(-1, 1)
    rules, _ = targeted_beam_search(
        Xb_tr, Y, 0, spec, min_recall=0.10, target_precision=0.55,
        min_support=40, beam_width=16, max_depth=4,
        Xbin_val=Xb_va, Y_val=Yv, gap_tol=0.25)
    return rules


def coverage_recall(rules, Xb, y):
    cov = np.zeros(len(y), dtype=bool)
    for r in rules:
        cov |= rule_mask(r.preds, Xb)
    pos = y == 1
    return float((cov & pos).sum() / pos.sum()), cov


def run(n=150_000, n_features=10, n_bins=20, seed=0):
    X, y, ring = make(n, n_features, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    names = [f"f{i:02d}" for i in range(n_features)]

    print(f"\n{'='*92}\nGAP-FILLING FEATURE ENGINEERING\n{'='*92}")
    print(f"label = axis-pattern OR ring;  n={n:,}  frauds={int(y.sum()):,}  "
          f"({100*y.mean():.1f}%)  of which ring-origin "
          f"{100*ring[y==1].mean():.0f}%")

    # --- base mine on ORIGINAL features ---
    Xb_tr, spec = fit_bins(X[tr], n_bins=n_bins)
    Xb_va = apply_bins(X[va], spec)
    t = time.perf_counter()
    base_rules = mine(Xb_tr, y[tr], spec, Xb_va, y[va])
    rec_before, cov_tr = coverage_recall(base_rules, Xb_tr, y[tr])
    rec_before_va, _ = coverage_recall(base_rules, Xb_va, y[va])
    print(f"\n[1] base rules on original features ({len(base_rules)} rules, "
          f"{time.perf_counter()-t:.1f}s)")
    for r in base_rules[:3]:
        print(f"      {r.label(names, spec)}")
    print(f"    coverage recall: train={rec_before:.2f}  val={rec_before_va:.2f}"
          f"   <-- the gap is the uncovered frauds")

    # --- diagnose the gap & synthesize features ---
    gap_mask, _ = uncovered_positives([rule_mask(r.preds, Xb_tr) for r in base_rules],
                                      y[tr])
    print(f"\n[2] gap = {int(gap_mask.sum()):,} uncovered frauds; "
          f"diagnosing geometry + synthesizing features...")
    cands = propose_features(X[tr], gap_mask, names, max_features=4)
    for c in cands:
        lift, lo, hi, prec, recl = c["band"]
        topo = f" topo(ring)={c['topo']:.2f}" if c["kind"] == "radial" else ""
        print(f"      [{c['kind']:6s}] {c['name']:42s} band ({lo:.1f},{hi:.1f}) "
              f"lift={lift:.1f} recall={recl:.2f}{topo}")
    if not cands:
        print("      (no separating feature found)")
        return

    # --- augment with the top feature and re-mine ---
    top = cands[0]
    print(f"\n[3] add top feature  «{top['name']}»  and re-mine")
    Xaug_tr = np.concatenate([X[tr], top["transform"](X[tr])[:, None]], axis=1)
    Xaug_va = np.concatenate([X[va], top["transform"](X[va])[:, None]], axis=1)
    names2 = names + ["ENG"]
    Xb2_tr, spec2 = fit_bins(Xaug_tr, n_bins=n_bins)
    Xb2_va = apply_bins(Xaug_va, spec2)
    aug_rules = mine(Xb2_tr, y[tr], spec2, Xb2_va, y[va])
    rec_after, _ = coverage_recall(aug_rules, Xb2_tr, y[tr])
    rec_after_va, _ = coverage_recall(aug_rules, Xb2_va, y[va])
    eng_rules = [r for r in aug_rules if any(f == n_features for f, _, _ in r.preds)]
    print(f"    rules now using ENG feature:")
    for r in eng_rules[:3]:
        print(f"      {r.label(names2, spec2)}")
    print(f"\n[4] coverage recall  before={rec_before_va:.2f}  ->  "
          f"after={rec_after_va:.2f}   (val)   gap closed: "
          f"{100*(rec_after_va-rec_before_va):.0f} pts")


if __name__ == "__main__":
    run()
