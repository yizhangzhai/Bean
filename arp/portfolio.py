"""Type-balanced rule portfolio via greedy maximin set cover.

Fraud types have distinct signatures, so we do NOT force one rule to be a
generalist. Instead we mine strong per-type rules, then assemble a small set
whose *worst-covered type* is as good as possible -- balancing at the portfolio
level, which is the honest place for it.

Coverage here = fraction of a type's positives (or $ loss) caught by the union
of selected rules, measured on validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .search import Rule


@dataclass
class Portfolio:
    rules: list[Rule]
    covered_frac: np.ndarray        # (C,) final coverage per type
    trajectory: list[np.ndarray]    # min-coverage after each pick


def _rule_hits(rule: Rule, X: np.ndarray) -> np.ndarray:
    mask = np.ones(X.shape[0], dtype=bool)
    for p in rule.preds:
        mask &= p.mask(X)
    return mask


def build_portfolio(
    rules: list[Rule],
    X_val: np.ndarray,
    Yw_val: np.ndarray,        # (N, C) -- counts or $ loss on validation
    *,
    max_rules: int = 8,
    min_precision_lift: float = 1.5,
) -> Portfolio:
    """Greedily pick rules to maximize the minimum per-type coverage.

    At each step choose the rule that most raises the worst-covered type, so the
    portfolio spreads across types instead of piling onto the easy one.
    """
    totals = Yw_val.sum(axis=0)                      # (C,) total per type on val
    totals = np.where(totals > 0, totals, 1.0)
    C = Yw_val.shape[1]

    masks = [_rule_hits(r, X_val) for r in rules]
    captured = np.zeros(X_val.shape[0], dtype=bool)  # union of selected rules

    chosen: list[Rule] = []
    trajectory: list[np.ndarray] = []
    used = set()

    for _ in range(max_rules):
        covered = (captured[:, None] * Yw_val).sum(axis=0) / totals  # (C,)
        worst = int(np.argmin(covered))   # currently worst-covered type
        # pick the unused rule that adds the most coverage to the worst type;
        # this always makes progress (unlike requiring the min itself to rise)
        best_i, best_gain = None, 0.0
        for i, m in enumerate(masks):
            if i in used:
                continue
            new_union = captured | m
            new_cov = (new_union[:, None] * Yw_val).sum(axis=0) / totals
            gain = new_cov[worst] - covered[worst]
            # tie-break toward rules that also lift overall coverage
            gain += 1e-3 * (new_cov.sum() - covered.sum())
            if gain > best_gain + 1e-12:
                best_gain, best_i = gain, i
        if best_i is None:
            break
        used.add(best_i)
        captured |= masks[best_i]
        chosen.append(rules[best_i])
        trajectory.append((captured[:, None] * Yw_val).sum(axis=0) / totals)

    final = (captured[:, None] * Yw_val).sum(axis=0) / totals
    return Portfolio(chosen, final, trajectory)
