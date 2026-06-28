"""Diagnose WHY greedy rule search cannot replicate the deep planted patterns.

Prints (1) the exact ground-truth conjunctions, and runs three checks that
together pin the failure mechanism:

  A. PREFIX GRADIENT.  precision(first k conditions) and the per-step gain.
     Theory: precision(S_k) ~ base + 0.83 * q^(d-k), so the gain of adding the
     k-th TRUE condition is ~0.83*q^(d-k)*(1-q) -- near zero for small k.

  B. DEPTH-1 BEAM COMPOSITION.  Score every single condition (feature,op,k) by
     precision; what fills the top-`beam`? If steep shallow-pattern conditions
     crowd out the deep patterns' seeds, the deep paths never start.

  C. BEAM CAPTURE AT DEPTH 2.  Given a deep pattern's true seed, what is the
     single best 2nd condition by precision? If it is a FOREIGN (steeper)
     pattern's condition rather than the true same-block partner, greedy
     steepest-ascent abandons the deep path.

Run:  python -m experiments.deep_diagnose
"""

from __future__ import annotations

import numpy as np

from arp.encode import encode_split_cm
from experiments.deep_bench import make_deep, DEPTHS, N_BINS, TARGET_FRAC


def cond_str(f, op, k):
    pct = int(round((k + 1) / N_BINS * 100))
    return f"f{f:04d} {'>' if op=='>' else '<'} p{pct:02d}"


def precision_of(mask, y):
    s = int(mask.sum())
    return (int(y[mask].sum()) / s if s else 0.0), s


def best_extension(seed_mask, Xtr, ytr, nb, min_support, used_feats):
    """Within the seed's support, the single (f,op,k) giving max 2-cond precision."""
    idx = np.flatnonzero(seed_mask)
    sub_y = ytr[idx].astype(np.float64)
    best = None
    for f in range(Xtr.shape[1]):
        if f in used_feats:
            continue
        xb = Xtr[idx, f]
        counts = np.bincount(xb, minlength=nb).astype(np.float64)
        cc = np.cumsum(counts)
        wc = np.cumsum(np.bincount(xb, weights=sub_y, minlength=nb))
        tot, N = wc[-1], len(idx)
        for k in range(nb - 1):
            for op, s, tp in (("<", cc[k], wc[k]), (">", N - cc[k], tot - wc[k])):
                if s >= min_support:
                    p = tp / s
                    if best is None or p > best[0]:
                        best = (p, int(s), f, op, k)
    return best


def main(n=200_000, n_features=500, seed=0):
    S, y, mo, planted, names, tr, va, n_sig = make_deep(n, n_features, seed)

    def make_column(f):
        if f < n_sig:
            return S[:, f]
        return np.random.default_rng(seed * 100003 + f).standard_normal(n).astype(np.float32)

    Xtr, Xva, spec = encode_split_cm(make_column, n_features, tr, va,
                                     n_bins=N_BINS, sample=100_000, seed=seed + 7)
    ytr = y[tr]
    base = float(ytr.mean())
    min_support = max(40, n // 4000)
    nb = N_BINS
    block_of = {}                              # feature -> pattern index
    col0 = 0
    for p, d in enumerate(DEPTHS):
        for f in range(col0, col0 + d):
            block_of[f] = p
        col0 += d

    print(f"\n{'='*92}\nPREDEFINED PATTERNS  (n={n:,}, base fraud rate={base:.3f}, "
          f"each fires on ~{TARGET_FRAC:.0%} of rows)\n{'='*92}")
    for p, d in enumerate(DEPTHS):
        q = TARGET_FRAC ** (1.0 / d)
        conds = "  AND  ".join(cond_str(f, op, k) for f, op, k in planted[p])
        print(f"\n  pattern {p} (depth {d}, each condition keeps q={q:.2f} of rows):")
        print(f"    {conds}")

    # ---- A. prefix gradient ----
    from arp.fast import rule_mask
    print(f"\n{'='*92}\nA. PREFIX PRECISION & PER-STEP GAIN (held-out-equivalent: train)\n{'='*92}")
    print(f"  {'depth d':>7} | precision after k conditions  ->  (gain from the k-th)")
    for p, d in enumerate(DEPTHS):
        prev = base
        cells = []
        for k in range(1, d + 1):
            m = rule_mask(planted[p][:k], Xtr)
            pr, _ = precision_of(m, ytr)
            cells.append((k, pr, pr - prev))
            prev = pr
        pick = [1, max(1, d // 2), d - 1, d]
        s = "  ".join(f"k={k}:{[c for c in cells if c[0]==k][0][1]:.2f}"
                      f"(+{[c for c in cells if c[0]==k][0][2]:.3f})" for k in pick)
        print(f"  {d:>7} | {s}")

    # ---- B. depth-1 beam composition ----
    print(f"\n{'='*92}\nB. DEPTH-1 BEAM: what fills the top-48 single conditions?\n{'='*92}")
    cands = []
    for f in range(n_features):
        xb = Xtr[:, f]
        counts = np.bincount(xb, minlength=nb).astype(np.float64)
        cc = np.cumsum(counts)
        wc = np.cumsum(np.bincount(xb, weights=ytr.astype(np.float64), minlength=nb))
        tot, N = wc[-1], len(ytr)
        for k in range(nb - 1):
            for op, sp, tp in (("<", cc[k], wc[k]), (">", N - cc[k], tot - wc[k])):
                if sp >= min_support:
                    cands.append((tp / sp, f, op, k))
    cands.sort(reverse=True)
    top = cands[:48]
    from collections import Counter
    blk = Counter(block_of.get(f, -1) for _, f, _, _ in top)
    print(f"  top-48 single-condition precisions range "
          f"{top[0][0]:.3f} (best) .. {top[-1][0]:.3f} (48th); base={base:.3f}")
    print(f"  of the top-48, block membership: " +
          "  ".join(f"depth{DEPTHS[b]}={blk.get(b,0)}" for b in range(len(DEPTHS))) +
          f"   noise={blk.get(-1,0)}")
    # where does each pattern's BEST seed rank?
    rank = {f: i for i, (_, f, _, _) in enumerate(cands)}
    for p, d in enumerate(DEPTHS):
        seeds = [i for i, (_, f, _, _) in enumerate(cands) if block_of.get(f) == p]
        bestrank = min(seeds) if seeds else -1
        bp = cands[bestrank][0] if bestrank >= 0 else 0
        print(f"    depth-{d:>2} pattern: best single condition ranks "
              f"#{bestrank} of {len(cands):,}  (precision {bp:.3f})")

    # ---- C. beam capture at depth 2 ----
    print(f"\n{'='*92}\nC. BEAM CAPTURE: best 2nd condition after a deep pattern's true seed\n{'='*92}")
    for p, d in enumerate(DEPTHS):
        f0, op0, k0 = planted[p][0]
        seed_mask = rule_mask([(f0, op0, k0)], Xtr)
        bp, bs, bf, bop, bk = best_extension(seed_mask, Xtr, ytr, nb, min_support, {f0})
        true_partner = planted[p][1]
        is_foreign = block_of.get(bf) != p
        # precision if we instead added the TRUE partner
        tm = rule_mask([(f0, op0, k0), true_partner], Xtr)
        tp_prec, _ = precision_of(tm, ytr)
        print(f"  depth-{d:>2} seed [{cond_str(f0,op0,k0)}]: greedy best 2nd = "
              f"[{cond_str(bf,bop,bk)}] P={bp:.3f}  "
              f"({'FOREIGN block depth-'+str(DEPTHS[block_of[bf]]) if is_foreign else 'same block'})")
        print(f"           true partner [{cond_str(*true_partner)}] would give only "
              f"P={tp_prec:.3f}  -> greedy picks the steeper foreign condition"
              if is_foreign else
              f"           true partner P={tp_prec:.3f}")


if __name__ == "__main__":
    main()
