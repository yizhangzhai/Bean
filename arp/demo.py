"""End-to-end demo: mine type-balanced fraud rules and compare to a tree.

Run:  python -m arp.demo            # count-based lift
      python -m arp.demo --dollars # dollar-loss-weighted lift
"""

from __future__ import annotations

import argparse

import numpy as np

from .baselines import tree_leaf_rules
from .data import make_fraud_data, train_val_split
from .encoding import make_predicates, stack_masks
from .portfolio import build_portfolio
from .scoring import base_rates, objective_single
from .search import beam_search, evaluate_rule, refine_rule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dollars", action="store_true",
                    help="weight by dollar loss instead of counts")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--coarse-bins", type=int, default=10)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--min-support", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = make_fraud_data(n=args.n, seed=args.seed)
    tr, va = train_val_split(data, seed=args.seed + 1)
    fn = data.feature_names

    # weight matrix: dollar loss or binary counts
    W = data.loss if args.dollars else data.Y.astype(np.float64)
    Wtr, Wva = W[tr], W[va]
    weight_kind = "dollar loss" if args.dollars else "counts"

    base_tr = base_rates(Wtr, len(tr))
    base_va = base_rates(Wva, len(va))

    print(f"\n{'='*70}\nattention-rule-paths demo  ({weight_kind})\n{'='*70}")
    print(f"train={len(tr)}  val={len(va)}  features={len(fn)}  "
          f"types={data.type_names}")
    print("\nHidden ground-truth signatures:")
    for name, gt in zip(data.type_names, data.ground_truth):
        rate = data.Y[:, data.type_names.index(name)].mean()
        print(f"  {name:18s} ({rate*100:.2f}% of rows):  {gt}")

    # coarse predicate set, computed on train quantiles
    preds = make_predicates(data.X[tr], n_bins=args.coarse_bins)
    pmasks = stack_masks(preds, data.X[tr])
    print(f"\nCoarse predicate set: {len(preds)} predicates "
          f"({args.coarse_bins}-bin) over {len(fn)} features")

    # ---- per-type mining (each fraud type is its own query) ----
    all_refined = []
    for c, tname in enumerate(data.type_names):
        obj = lambda lift, c=c: objective_single(lift, c)
        rules = beam_search(
            preds, pmasks, Wtr, base_tr, obj,
            beam_width=args.beam, max_depth=args.depth,
            min_support=args.min_support,
        )
        print(f"\n--- {tname} : top mined paths (train lift) ---")
        for r in rules[:3]:
            rr = refine_rule(r, data.X[tr], Wtr, base_tr, obj,
                             min_support=args.min_support)
            rv = evaluate_rule(rr, data.X[va], Wva, base_va, obj)
            print(f"  coarse: {r.label(fn)}")
            print(f"  refined:{rr.label(fn)}")
            print(f"     train lift[{tname}]={rr.lift[c]:6.1f}  "
                  f"val lift={rv.lift[c]:6.1f}  "
                  f"val support={rv.support}")
            all_refined.append(rr)

    # ---- type-balanced portfolio (on validation) ----
    print(f"\n{'='*70}\nType-balanced portfolio (greedy maximin coverage on val)\n{'='*70}")
    port = build_portfolio(all_refined, data.X[va], Wva, max_rules=6)
    for i, (r, cov) in enumerate(zip(port.rules, port.trajectory), 1):
        print(f"  pick {i}: {r.label(fn)}")
        print(f"           min-coverage after pick = {cov.min()*100:5.1f}%  "
              f"per-type = {np.round(cov*100,1)}")
    print(f"\n  final coverage per type ({weight_kind}): "
          f"{dict(zip(data.type_names, np.round(port.covered_frac*100,1)))}")
    print(f"  worst-covered type: {port.covered_frac.min()*100:.1f}%")

    # ---- decision-tree baseline on 'any fraud' ----
    print(f"\n{'='*70}\nBaseline: sklearn decision tree on 'any fraud'\n{'='*70}")
    y_any = data.any_fraud
    leaves = tree_leaf_rules(data.X[tr], y_any[tr], fn,
                             max_depth=args.depth, min_samples_leaf=args.min_support)
    for r in leaves[:3]:
        print(f"  lift={r['lift']:5.1f} support={r['support']:5d}  "
              f"{'  AND  '.join(r['conds'])}")
    print("\n(Tree optimizes one pooled label; note it tends to chase the common "
          "type and dilute the rare/expensive ones -- the portfolio balances them.)\n")


if __name__ == "__main__":
    main()
