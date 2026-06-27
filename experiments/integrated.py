"""Integrated pipeline: rule search + feature engineering, end to end.

A diverse fraud population -- one label fired by FIVE distinct modus operandi:

  MO1 deep_axis : f5>p70 AND f6>p70 AND f7<p30          (axis -> rule search)
  MO2 band_axis : p45<f8<p65 AND f9>p75                 (axis band -> rule search)
  MO3 ring      : 4 < dist((f0,f1),(20,20)) < 6         (non-axis -> radial feat)
  MO4 ratio     : f2/f3 above its p95                   (non-axis -> ratio feat)
  MO5 periodic  : (f4 mod 24) in [2,4]                  (non-axis -> mod feat)

Pipeline:
  supervised binning -> mine rules -> measure per-MO coverage ->
  LOOP { featgap: diagnose gap, synthesize the best feature, re-mine } until
  coverage plateaus. Each round closes one non-axis MO; watch coverage climb.
"""

from __future__ import annotations

import time

import numpy as np

from arp.fast import fit_bins, rule_mask
from arp.targeted import targeted_beam_search
from featgap import (uncovered_positives, remine_residual, interaction_screen,
                     hsic, propose_features)

GEO_C = (20.0, 20.0)
MO_NAMES = ["deep_axis", "band_axis", "ring", "ratio", "periodic"]


def make(n, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, n_features)).astype(np.float32)
    X[:, 0] = rng.uniform(0, 40, n)            # geo lat
    X[:, 1] = rng.uniform(0, 40, n)            # geo lon
    X[:, 2] = rng.uniform(1, 100, n)           # amount
    X[:, 3] = rng.uniform(10, 100, n)          # income
    X[:, 4] = rng.uniform(0, 1000, n)          # timestamp
    X[:, 5:10] = rng.standard_normal((n, 5))   # behavioural

    def q(col, p):
        return np.quantile(col, p)

    r = np.hypot(X[:, 0] - GEO_C[0], X[:, 1] - GEO_C[1])
    ratio = X[:, 2] / X[:, 3]
    mos = [
        (X[:, 5] > q(X[:, 5], .70)) & (X[:, 6] > q(X[:, 6], .70)) & (X[:, 7] < q(X[:, 7], .30)),
        (X[:, 8] > q(X[:, 8], .45)) & (X[:, 8] < q(X[:, 8], .65)) & (X[:, 9] > q(X[:, 9], .75)),
        (r > 4) & (r < 6),
        ratio > q(ratio, .95),
        (np.mod(X[:, 4], 24) >= 2) & (np.mod(X[:, 4], 24) <= 4),
    ]
    fire = [0.85, 0.85, 0.85, 0.85, 0.85]
    # make MOs DISJOINT by region (priority order) so per-MO coverage is clean
    mo_id = np.full(n, -1, dtype=np.int64)
    for k, m in enumerate(mos):
        mo_id[m & (mo_id < 0)] = k
    y = np.zeros(n, dtype=np.int64)
    mo_masks = []
    for k in range(len(mos)):
        hit = (mo_id == k) & (rng.uniform(size=n) < fire[k])
        mo_masks.append(hit)
        y |= hit.astype(np.int64)
    return X, y, mo_masks


def apply_bins(X, spec):
    Xb = np.empty(X.shape, dtype=np.int8)
    for f in range(X.shape[1]):
        Xb[:, f] = np.searchsorted(spec.edges[f], X[:, f], side="right").astype(np.int8)
    return Xb


def mine(Xb_tr, ytr, spec, Xb_va, yva):
    rules, _ = targeted_beam_search(
        Xb_tr, ytr.reshape(-1, 1), 0, spec, min_recall=0.05,
        target_precision=0.5, min_support=40, beam_width=16, max_depth=4,
        Xbin_val=Xb_va, Y_val=yva.reshape(-1, 1), gap_tol=0.30)
    return rules


def covered_mask(rules, Xb):
    cov = np.zeros(Xb.shape[0], dtype=bool)
    for r in rules:
        cov |= rule_mask(r.preds, Xb)
    return cov


def per_mo_coverage(cov, mo_masks, y):
    return [float((cov & m).sum() / max(1, m.sum())) for m in mo_masks]


def run(n=200_000, n_features=40, n_bins=16, seed=0, max_rounds=4):
    X, y, mo_masks = make(n, n_features, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    names = [f"f{i:02d}" for i in range(n_features)]

    print(f"\n{'='*100}\nINTEGRATED: rule search + feature engineering\n{'='*100}")
    print(f"n={n:,}  frauds={int(y.sum()):,} ({100*y.mean():.1f}%)  "
          f"5 modus operandi:")
    for k, mn in enumerate(MO_NAMES):
        print(f"    MO{k+1} {mn:10s}: {int(mo_masks[k].sum()):,} cases")

    Xa = X.copy()                      # feature matrix grows as we engineer
    anames = list(names)
    added = []
    prev_overall = -1.0
    stale = 0

    for rnd in range(max_rounds + 1):
        Xa_tr, Xa_va = Xa[tr], Xa[va]
        Xb_tr, spec = fit_bins(Xa_tr, n_bins=n_bins, Y=y[tr].reshape(-1, 1),
                               supervised=True)
        Xb_va = apply_bins(Xa_va, spec)
        rules = mine(Xb_tr, y[tr], spec, Xb_va, y[va])
        cov_va = covered_mask(rules, Xb_va)
        mo_cov = per_mo_coverage(cov_va, [m[va] for m in mo_masks], y[va])
        overall = float((cov_va & (y[va] == 1)).sum() / (y[va] == 1).sum())

        tag = "base (axis only)" if rnd == 0 else f"after +{added[-1]}"
        print(f"\n[round {rnd}] {tag}   overall coverage = {overall:.2f}")
        print("    per-MO: " + "  ".join(
            f"{MO_NAMES[k]}={mo_cov[k]:.2f}" for k in range(5)))

        # stop only after two consecutive non-improving rounds (a spurious round
        # can precede the one that finally closes a smaller MO like the ring)
        if rnd > 0:
            stale = stale + 1 if (overall - prev_overall) < 0.01 else 0
        prev_overall = max(prev_overall, overall)
        if stale >= 2 or overall > 0.97:
            print("    coverage plateaued -> stop")
            break
        if rnd == max_rounds:
            break

        # --- featgap on the residual ---
        gap_tr, cov_tr = uncovered_positives(rules, Xb_tr, y[tr])
        if rnd == 0:
            diag = remine_residual(Xb_tr, y[tr], cov_tr, spec)
            keep = ~((y[tr] == 1) & cov_tr)
            pairs, _ = interaction_screen(Xa_tr[:, :n_features], gap_tr.astype(int),
                                          mask=keep, bins=8, top_k=1)
            i, j, syn = pairs[0][:3]
            print(f"    [diagnose] {diag['verdict']}")
            print(f"    [screen]  top interacting pair ({names[i]},{names[j]}) "
                  f"synergy={syn:+.4f}")
        cands = propose_features(Xa_tr, gap_tr, anames, max_features=3)
        if not cands:
            print("    no separating feature found -> stop")
            break
        top = cands[0]
        lo, hi = top["band"][1], top["band"][2]
        print(f"    [synth]   +«{top['name']}» [{top['kind']}] band ({lo:.1f},{hi:.1f})"
              f" lift={top['lift']:.1f}")
        Xa = np.concatenate([Xa, top["transform"](Xa)[:, None]], axis=1)
        anames.append(f"ENG{len(added)}:{top['kind']}")
        added.append(anames[-1])

    print(f"\n{'-'*100}\nengineered features added: {added}")
    print(f"final overall coverage: {overall:.2f}  "
          f"(rule search alone reached the axis MOs; feature engineering closed "
          f"the ring / ratio / periodic MOs)")


if __name__ == "__main__":
    run()
