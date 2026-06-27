"""Interpretability constraints enforced during discovery.

Mines the 3-type fraud data unconstrained, then under a RulePolicy, and
verifies every emitted rule obeys the policy (monotone direction, 1-way only,
threshold range, forbidden pair). Shows how a constraint *changes* the rules.
"""

from __future__ import annotations

import numpy as np

from arp.data import make_binned_fraud_data_large
from arp.constraints import RulePolicy
from arp.targeted import targeted_beam_search


def mine(Xtr, Ytr, Xva, Yva, spec, c, policy=None):
    return targeted_beam_search(
        Xtr, Ytr, c, spec, min_recall=0.25, target_precision=0.40,
        min_support=40, beam_width=12, max_depth=4,
        Xbin_val=Xva, Y_val=Yva, gap_tol=0.25, policy=policy)[0]


def check(rules, policy, spec, names):
    """Return list of violations (should be empty under enforcement)."""
    v = []
    for r in rules:
        feats = [f for f, _, _ in r.preds]
        for f, op, k in r.preds:
            c = policy.fc(f)
            if op not in c.directions:
                v.append(f"{names[f]} uses '{op}' (allowed {c.directions})")
            pct = (k + 1) / spec.n_bins
            if not (c.pct_range[0] <= pct <= c.pct_range[1] + 1e-9):
                v.append(f"{names[f]} thr p{int(pct*100)} outside {c.pct_range}")
            if not c.two_way and sum(1 for g, _, _ in r.preds if g == f) > 1:
                v.append(f"{names[f]} used 2-way but 1-way only")
        for fp in policy.forbidden_pairs:
            if fp <= set(feats):
                v.append(f"forbidden pair {sorted(fp)} co-appears")
    return v


def run(n=200_000, n_features=60, seed=0):
    Xbin, spec, Y, loss, names, tnames, gt = make_binned_fraud_data_large(
        n=n, n_features=n_features, n_bins=10, seed=seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Xtr, Xva, Ytr, Yva = Xbin[tr], Xbin[va], Y[tr], Y[va]
    C = 0  # account_takeover: true signature f01>p95 AND f00>p90

    print(f"\n{'='*92}\nINTERPRETABILITY CONSTRAINTS  (target = {tnames[C]})\n{'='*92}")

    print("\n[1] UNCONSTRAINED")
    base = mine(Xtr, Ytr, Xva, Yva, spec, C)
    for r in base[:3]:
        print(f"    P={r.precision:.2f} R={r.recall:.2f}  {r.label(names, spec)}")

    # Policy:
    #  - f00 monotone-increasing (chargeback-like): only '>', 1-way
    #  - f03 may only split in its lower half (p<=40)
    #  - f01 and f08 forbidden from co-appearing
    policy = RulePolicy.build(
        monotone_up=[0],
        ranges={3: (0.0, 0.40)},
        disable=[11],                              # f11 not allowed -> reroute
        forbidden_pairs=[(1, 8)],
    )
    print("\n[2] CONSTRAINED  (f00 monotone-up/1-way; f03 thr<=p40; disable f11; "
          "forbid {f01,f08})")
    con = mine(Xtr, Ytr, Xva, Yva, spec, C, policy=policy)
    for r in con[:3]:
        print(f"    P={r.precision:.2f} R={r.recall:.2f}  {r.label(names, spec)}")

    viol = check(con, policy, spec, names)
    print(f"\n  compliance: {'ALL RULES OBEY POLICY' if not viol else 'VIOLATIONS:'}")
    for x in viol[:10]:
        print(f"    !! {x}")

    # Demonstrate a constraint biting: forbid the true pair {f00,f01}
    policy2 = RulePolicy.build(forbidden_pairs=[(0, 1)])
    con2 = mine(Xtr, Ytr, Xva, Yva, spec, C, policy=policy2)
    print("\n[3] CONSTRAINT BITES  (forbid the true pair {f00,f01})")
    if con2:
        r = max(con2, key=lambda r: r.val_precision * r.val_recall)
        uses_both = {0, 1} <= {f for f, _, _ in r.preds}
        print(f"    best now: {r.label(names, spec)}")
        print(f"    uses both f00&f01? {uses_both}  (expected False) "
              f"-> miner routed around the forbidden interaction")
    else:
        print("    no compliant rule met targets without the f00&f01 interaction")

    # Direction filter: force f01 to '<' only (contradicts true f01>p90)
    policy3 = RulePolicy.build(monotone_down=[1])
    con3 = mine(Xtr, Ytr, Xva, Yva, spec, C, policy=policy3)
    used_f01_gt = any(f == 1 and op == ">" for r in con3 for f, op, _ in r.preds)
    print("\n[4] DIRECTION FILTER  (force f01 monotone-DOWN: '<' only)")
    print(f"    any rule uses f01 '>'? {used_f01_gt}  (expected False) "
          f"-> '>' predicates on f01 never generated")
    if con3:
        r = max(con3, key=lambda r: r.val_precision * r.val_recall)
        print(f"    best compliant rule: {r.label(names, spec)}")


if __name__ == "__main__":
    run()
