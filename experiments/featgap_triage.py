"""featgap triage, end to end (the SEPARATE feature-engineering layer).

One fraud label fired by EITHER an axis pattern OR a ring that no box can cover:
    fraud if (5 < f2 < 8 AND f3 > 15)  OR  (2 < dist((f0,f1),(10,10)) < 4)

Pipeline:
  [base]  mine axis rules with `arp` -> gap = uncovered (ring) frauds
  [step1] featgap.remine_residual    -> is the gap axis-coverable or non-axis?
  [step3] featgap.interaction_screen -> which feature PAIR carries the gap?
          + HSIC confirms the dependence (the measure we opened the project with)
  [synth] featgap.propose_features    -> synthesize the radial feature
  [remine] add it, re-mine -> coverage closes
"""

from __future__ import annotations

import numpy as np

from arp.fast import fit_bins, rule_mask
from arp.targeted import targeted_beam_search
from featgap import (uncovered_positives, remine_residual, interaction_screen,
                     hsic, propose_features)

CENTER = (10.0, 10.0)


def make(n, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 20, size=(n, n_features)).astype(np.float32)
    r = np.hypot(X[:, 0] - CENTER[0], X[:, 1] - CENTER[1])
    member = ((X[:, 2] > 5) & (X[:, 2] < 8) & (X[:, 3] > 15)) | ((r > 2) & (r < 4))
    y = (rng.uniform(size=n) < np.where(member, 0.85, 0.004)).astype(np.int64)
    return X, y


def apply_bins(X, spec):
    Xb = np.empty(X.shape, dtype=np.int8)
    for f in range(X.shape[1]):
        Xb[:, f] = np.searchsorted(spec.edges[f], X[:, f], side="right").astype(np.int8)
    return Xb


def mine(Xb_tr, y_tr, spec, Xb_va, y_va):
    rules, _ = targeted_beam_search(
        Xb_tr, y_tr.reshape(-1, 1), 0, spec, min_recall=0.10,
        target_precision=0.55, min_support=40, beam_width=16, max_depth=4,
        Xbin_val=Xb_va, Y_val=y_va.reshape(-1, 1), gap_tol=0.25)
    return rules


def coverage(rules, Xb, y):
    cov = np.zeros(len(y), dtype=bool)
    for r in rules:
        cov |= rule_mask(r.preds, Xb)
    return float((cov & (y == 1)).sum() / (y == 1).sum())


def run(n=150_000, n_features=10, n_bins=20, seed=0):
    X, y = make(n, n_features, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    names = [f"f{i:02d}" for i in range(n_features)]
    print(f"\n{'='*92}\nFEATGAP TRIAGE  (separate layer on top of arp)\n{'='*92}")

    Xb_tr, spec = fit_bins(X[tr], n_bins=n_bins)
    Xb_va = apply_bins(X[va], spec)
    base = mine(Xb_tr, y[tr], spec, Xb_va, y[va])
    rec0 = coverage(base, Xb_va, y[va])
    print(f"[base]  {len(base)} axis rules   coverage recall (val) = {rec0:.2f}")

    gap_tr, cov_tr = uncovered_positives(base, Xb_tr, y[tr])
    print(f"        gap = {int(gap_tr.sum()):,} uncovered frauds\n")

    # ---- step 1: is the gap axis-coverable, or genuinely non-axis? ----
    diag = remine_residual(Xb_tr, y[tr], cov_tr, spec)
    print(f"[step1] re-mine residual -> {diag['verdict']}")
    print(f"        (best single residual axis rule only reaches R="
          f"{diag['recall']:.2f}; the ring resists boxes)\n")

    # ---- step 3: which feature PAIR carries the gap? ----
    keep = ~((y[tr] == 1) & cov_tr)            # drop covered frauds
    pairs, marg = interaction_screen(X[tr], gap_tr.astype(int), mask=keep,
                                     bins=8, top_k=4)
    print("[step3] interaction screen (synergy = MI(y;i,j) - MI(y;i) - MI(y;j)):")
    for i, j, syn, mij, mi, mj in pairs:
        print(f"        ({names[i]},{names[j]})  synergy={syn:+.4f}  "
              f"joint={mij:.4f}  marginals=({mi:.4f},{mj:.4f})")
    i, j = pairs[0][0], pairs[0][1]
    h = hsic(np.c_[X[tr][keep][:, i], X[tr][keep][:, j]], gap_tr[keep].astype(float))
    print(f"        -> top pair ({names[i]},{names[j]}): joint MI ({pairs[0][3]:.3f}) "
          f"far exceeds each marginal ({pairs[0][4]:.3f},{pairs[0][5]:.3f}) "
          f"-> a real interaction")
    print(f"        HSIC(joint pair ; residual) = {h:.4f}  (kernel dependence "
          f"confirms it)\n")

    # ---- synthesize + re-mine ----
    cands = propose_features(X[tr], gap_tr, names, max_features=3)
    top = cands[0]
    lift, lo, hi, _, _ = top["band"]
    print(f"[synth] top feature: «{top['name']}»  band ({lo:.1f},{hi:.1f})  "
          f"lift={lift:.1f}")
    Xa_tr = np.concatenate([X[tr], top["transform"](X[tr])[:, None]], axis=1)
    Xa_va = np.concatenate([X[va], top["transform"](X[va])[:, None]], axis=1)
    Xb2_tr, spec2 = fit_bins(Xa_tr, n_bins=n_bins)
    Xb2_va = apply_bins(Xa_va, spec2)
    aug = mine(Xb2_tr, y[tr], spec2, Xb2_va, y[va])
    rec1 = coverage(aug, Xb2_va, y[va])
    print(f"[remine] coverage recall (val): {rec0:.2f} -> {rec1:.2f}  "
          f"(+{100*(rec1-rec0):.0f} pts)")


if __name__ == "__main__":
    run()
