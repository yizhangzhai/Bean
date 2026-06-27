"""Interpretability constraints on CATEGORICAL predicates (mixed miner).

Shows: restrict a categorical to equality-only (no subsets), cap subset size,
forbid two categoricals from co-appearing, and a soft 'discouraged' preference
that breaks ties toward the more interpretable rule. Compliance is verified.
"""

from __future__ import annotations

import numpy as np

from arp.fast import fit_bins
from arp.mixed import Meta, mixed_targeted_search, EqPred, InPred
from arp.constraints import RulePolicy
from experiments.mixed_recovery import make, CARDS


def build(n, seed):
    Xnum, Xcat, Y = make(n, seed)
    n_bins, n_num = 12, Xnum.shape[1]
    Xb, spec = fit_bins(Xnum, n_bins=n_bins)
    M = np.concatenate([Xb.astype(np.int32), Xcat], axis=1)
    meta = Meta(
        names=[f"num{i}" for i in range(n_num)] + [f"cat{i}" for i in range(len(CARDS))],
        kind=["num"] * n_num + ["cat"] * len(CARDS),
        size=[n_bins] * n_num + CARDS)
    return M, Y, meta, spec, n_num


def mine(M, y, meta, policy=None):
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(y))
    cut = int(len(y) * 0.67)
    tr, va = perm[:cut], perm[cut:]
    return mixed_targeted_search(
        M[tr], y[tr].astype(float), meta, min_recall=0.25, target_precision=0.45,
        min_support=40, beam_width=12, max_depth=5,
        M_val=M[va], y_val=y[va].astype(float), gap_tol=0.25, policy=policy)[0]


def show(tag, rules, meta):
    print(f"  {tag}")
    for r in rules[:2]:
        print(f"     P={r.precision:.2f} R={r.recall:.2f}  {r.label(meta)}")
    if not rules:
        print("     (no compliant rule met targets)")


def run(n=200_000):
    M, Y, meta, spec, n_num = build(n, 0)
    # cat columns are unified indices n_num..  (cat2 = merchant = index n_num+2)
    cat0, cat1, cat2, cat3 = (n_num + i for i in range(4))
    # device_merchant pattern: cat1==c4 AND cat2 in {c7,c9} AND num1 > p85
    c = 1
    print(f"\n{'='*92}\nCATEGORICAL CONSTRAINTS  (target = device_merchant: "
          f"cat1==c4 & cat2 in {{c7,c9}} & num1>p85)\n{'='*92}")

    show("[1] UNCONSTRAINED", mine(M, Y[:, c], meta), meta)

    # cat2 equality-only: cannot form the {c7,c9} subset -> must use one level
    pol_eq = RulePolicy.build(cat_eq_only=[cat2])
    rules_eq = mine(M, Y[:, c], meta, pol_eq)
    show("[2] cat2 EQUALITY-ONLY (no subsets allowed)", rules_eq, meta)
    bad = [r for r in rules_eq for p in r.preds
           if isinstance(p, InPred) and p.f == cat2]
    print(f"     compliance: cat2 used as a subset? {bool(bad)} (expected False)")

    # cap subset size to 2 (true set is size 2 -> still recoverable)
    pol_sz = RulePolicy.build(max_set_size={cat2: 2})
    rules_sz = mine(M, Y[:, c], meta, pol_sz)
    show("[3] cat2 max_set_size=2", rules_sz, meta)
    over = [len(p.codes) for r in rules_sz for p in r.preds
            if isinstance(p, InPred) and p.f == cat2 and len(p.codes) > 2]
    print(f"     compliance: any cat2 set > 2 levels? {bool(over)} (expected False)")

    # forbid cat1 & cat2 co-appearing (the essential interaction) -> reroute/empty
    pol_fb = RulePolicy.build(forbidden_pairs=[(cat1, cat2)])
    rules_fb = mine(M, Y[:, c], meta, pol_fb)
    show("[4] forbid {cat1,cat2} co-occurrence", rules_fb, meta)
    both = [r for r in rules_fb
            if {cat1, cat2} <= {p.f for p in r.preds}]
    print(f"     compliance: any rule has both cat1 & cat2? {bool(both)} "
          f"(expected False)")


if __name__ == "__main__":
    run()
