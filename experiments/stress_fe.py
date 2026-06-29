"""Does the feature-engineering module add capture? Plant AXIS patterns + a few
NON-AXIS ones (ring / ratio / periodic) that no threshold conjunction can express,
then:
  stage 1: recover_deep on the raw bins        -> captures axis, MISSES non-axis
  stage 2: featgap on the residual -> synthesize radial/ratio/periodic features,
           append, recover_deep again           -> captures the non-axis patterns

Shows precisely where FE helps (non-axis residual) and where it can't (axis misses
are rarity/depth, not representation).

Run:  python -m experiments.stress_fe
"""

from __future__ import annotations

import time
import numpy as np

from arp.fast import rule_mask, BinSpec
from arp.encode import quantile_edges, assign_bins
from featgap import recover_deep, propose_features
from experiments.stress import encode, peak_gb

CAND = [0, 1, 2, 3, 4]              # geo(0,1), amount/income(2,3), timestamp(4)


def make_mixed(n, n_features, seed, n_axis=28):
    rng = np.random.default_rng(seed)
    n_cat = 12
    cat_idx = list(range(n_features - n_cat, n_features))
    card = {j: int(rng.integers(3, 8)) for j in cat_idx}
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    X[:, 0] = rng.uniform(0, 40, n); X[:, 1] = rng.uniform(0, 40, n)
    X[:, 2] = rng.uniform(1, 100, n); X[:, 3] = rng.uniform(10, 100, n)
    X[:, 4] = rng.uniform(0, 1000, n)
    for j in cat_idx:
        X[:, j] = rng.integers(0, card[j], n).astype(np.float32)

    r = np.hypot(X[:, 0] - 20, X[:, 1] - 20)
    ratio = X[:, 2] / X[:, 3]
    tmod = np.mod(X[:, 4], 24)
    fire = 0.85
    patterns, region = [], []                       # non-axis ~2% of rows each
    patterns.append(dict(kind="ring")); region.append((r > 4.5) & (r < 5.5))
    patterns.append(dict(kind="ratio")); region.append(ratio > np.quantile(ratio, .98))
    patterns.append(dict(kind="periodic")); region.append((tmod >= 2) & (tmod <= 2.5))

    pool = list(range(5, n_features - n_cat)) + cat_idx
    sizes = rng.integers(200, 1500, n_axis).astype(float)
    sizes = np.maximum(60, sizes / sizes.sum() * 0.03 * n / fire).astype(int)  # ~3% total
    for p in range(n_axis):
        d, Np = int(rng.choice([2, 3, 4, 5, 6])), int(sizes[p])
        q = (Np / n) ** (1.0 / d)
        m = np.ones(n, dtype=bool)
        for f in rng.choice(pool, d, replace=False):
            if f in card:
                m &= (X[:, f] == int(rng.integers(0, card[f])))
            else:
                col = X[:, f]
                m &= (col > np.quantile(col, 1 - q)) if rng.random() < 0.5 \
                    else (col < np.quantile(col, q))
        patterns.append(dict(kind="axis", depth=d, size=Np))
        region.append(m)

    fire = 0.85
    order = list(range(3)) + [3 + i for i in
                              np.argsort([-patterns[3 + i].get("size", 0) for i in range(n_axis)])]
    mo_id = np.full(n, -1, dtype=np.int64)
    for p in order:
        mo_id[region[p] & (mo_id < 0)] = p
    y = np.zeros(n, dtype=np.int64)
    mo_masks = []
    for p in range(len(region)):
        hit = (mo_id == p) & (rng.random(n) < fire)
        mo_masks.append(hit); y |= hit.astype(np.int64)
    y |= ((mo_id < 0) & (rng.random(n) < 0.0008)).astype(np.int64)
    miss = set(int(j) for j in rng.choice(range(5, n_features),
                                          int(0.12 * (n_features - 5)), replace=False))
    for j in miss:
        X[rng.random(n) < 0.08, j] = np.nan
    for p, pt in enumerate(patterns):
        pt["realized"] = int(mo_masks[p].sum())
    return X, y, mo_masks, patterns, cat_idx


def mine(Xtr_b, Xva_b, spec, ytr, yva):
    return recover_deep(Xtr_b, Xva_b, spec, ytr, yva,
                        np.zeros(len(ytr), dtype=bool), max_rounds=30, top_k=22,
                        seed_n=250, n_seeds=8, n_jobs=6, min_round_gain=120,
                        target_precision=0.6, min_accept_precision=0.12,
                        max_misses=2, min_recall=0.004, min_support=20,
                        beam_width=64, max_depth=18, seed=0, verbose=False)[0]


def evaluate(preds, Xva_b, yva, mo_va, patterns, tag):
    cov = np.zeros(len(yva), dtype=bool)
    for pr in preds:
        cov |= rule_mask(pr, Xva_b)
    pos = yva == 1
    rec = (cov & pos).sum() / max(1, pos.sum())
    prec = (cov & pos).sum() / max(1, cov.sum())
    pcov = [(cov & m).sum() / max(1, m.sum()) for m in mo_va]
    print(f"\n  {tag}: recall={rec:.2f} precision={prec:.2f} rules={len(preds)}")
    for kind in ("ring", "ratio", "periodic"):
        c = [pcov[p] for p, pt in enumerate(patterns) if pt["kind"] == kind][0]
        print(f"      NON-AXIS {kind:9s} coverage = {c:.2f}")
    ax = [pcov[p] >= 0.5 for p, pt in enumerate(patterns)
          if pt["kind"] == "axis" and pt["realized"] >= 150]
    print(f"      AXIS (>=150 cases) captured: {sum(ax)}/{len(ax)}")
    return pcov


def run(n=300_000, n_features=120, seed=0):
    t0 = time.perf_counter()
    print(f"\n{'='*92}\nFEATURE-ENGINEERING CAPTURE TEST  n={n:,}  F={n_features}\n{'='*92}")
    X, y, mo, patterns, cat_idx = make_mixed(n, n_features, seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n); cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Xtr_b, Xva_b, spec = encode(X[tr], X[va], cat_idx, 100_000, seed + 7)
    ytr, yva = y[tr], y[va]
    mo_va = [m[va] for m in mo]
    print(f"  frauds={int(y.sum()):,} ({100*y.mean():.2f}%)  3 non-axis + "
          f"{len(patterns)-3} axis patterns")

    # stage 1
    deep1 = mine(Xtr_b, Xva_b, spec, ytr, yva)
    evaluate(deep1, Xva_b, yva, mo_va, patterns, "STAGE 1 (raw bins)")

    # featgap on the train residual
    cov1 = np.zeros(len(ytr), dtype=bool)
    for pr in deep1:
        cov1 |= rule_mask(pr, Xtr_b)
    gap_tr = (ytr == 1) & ~cov1
    cands = propose_features(X[tr][:, CAND], gap_tr,
                             [f"f{c}" for c in CAND], max_features=5)
    print("\n  featgap proposes (on residual):")
    for c in cands[:4]:
        print(f"      +{c['name']:32s} [{c['kind']:8s}] lift={c['lift']:.1f}")

    # augment + stage 2
    qs = np.arange(1, spec.n_bins - 1) / (spec.n_bins - 1)   # 15 quantile cuts
    rng2 = np.random.default_rng(seed + 9)
    atr, ava, nedge = [], [], []
    for c in cands[:4]:
        v = c["transform"](X[:, CAND])
        e = quantile_edges(v[tr], qs, sample=100_000, rng=rng2)
        atr.append(assign_bins(v[tr], e)); ava.append(assign_bins(v[va], e)); nedge.append(e)
    Xtr2 = np.concatenate([np.asarray(Xtr_b)] + [a[:, None] for a in atr], axis=1)
    Xva2 = np.concatenate([np.asarray(Xva_b)] + [a[:, None] for a in ava], axis=1)
    spec2 = BinSpec(list(spec.edges) + nedge, spec.pct, spec.n_bins)
    deep2 = mine(Xtr2, Xva2, spec2, ytr, yva)
    evaluate(deep2, Xva2, yva, mo_va, patterns, "STAGE 2 (+ engineered features)")
    print(f"\n  TOTAL {time.perf_counter()-t0:.0f}s   peak RSS {peak_gb():.2f} GB")


if __name__ == "__main__":
    run()
