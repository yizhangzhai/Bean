"""Beam search over conjunctive rules ('decision paths').

A Rule is an AND of predicates over *distinct* features. Growth is greedy and
beam-limited, exactly like a tree picking the best split at a node, except we
keep the top-K partial rules instead of committing to one.

Conjunction membership is a boolean AND of masks -- cheap and exact. Each
expansion scores ALL candidate predicates against a base rule in one matmul
(see scoring.score_many), so a depth step is O(beam * n_preds * N).

Coarse-to-fine: mine with coarse (decile) predicates, then `refine_rule`
sharpens each winning threshold to fine resolution *within its local band* and
re-validates -- so we pay full resolution only on the handful of survivors,
and a sharp fraud band that a decile average would dilute still gets found via
the best-achievable-lift refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .encoding import Predicate
from .scoring import score_many, score_one


@dataclass
class Rule:
    preds: tuple[Predicate, ...]
    support: int
    caught: np.ndarray
    lift: np.ndarray
    value: float
    mask: np.ndarray = field(repr=False, default=None)

    @property
    def features(self) -> frozenset[int]:
        return frozenset(p.feature for p in self.preds)

    def key(self) -> frozenset:
        return frozenset((p.feature, p.op, round(p.pct, 4)) for p in self.preds)

    def label(self, feature_names: list[str]) -> str:
        return "  AND  ".join(p.label(feature_names) for p in self.preds)


Objective = Callable[[np.ndarray], np.ndarray]  # lift (..., C) -> scalar(...)


def beam_search(
    preds: list[Predicate],
    pred_masks: np.ndarray,      # (n_preds, N) bool on train
    Yw: np.ndarray,              # (N, C)
    base: np.ndarray,            # (C,)
    objective: Objective,
    *,
    beam_width: int = 8,
    max_depth: int = 3,
    min_support: int = 30,
    min_gain: float = 0.10,
) -> list[Rule]:
    """Return the best rules found across all depths (deduped, score-sorted)."""
    n = Yw.shape[0]

    # ---- depth 1 ----
    support, lift = score_many(pred_masks, Yw, base, n)
    val = objective(lift)
    val = np.where(support >= min_support, val, -np.inf)
    order = np.argsort(val)[::-1][:beam_width]

    beam: list[Rule] = []
    for i in order:
        if not np.isfinite(val[i]):
            continue
        beam.append(Rule((preds[i],), int(support[i]), lift[i] * base, lift[i],
                          float(val[i]), pred_masks[i].copy()))

    pool: dict[frozenset, Rule] = {r.key(): r for r in beam}

    # ---- grow ----
    for _depth in range(2, max_depth + 1):
        candidates: list[Rule] = []
        for r in beam:
            # AND base rule mask with every predicate at once
            conj = pred_masks & r.mask[None, :]            # (n_preds, N)
            support, lift = score_many(conj, Yw, base, n)
            val = objective(lift)
            # forbid the same (feature, direction) -- but allow the opposite
            # direction on a used feature, so two-sided bands (p40<f<p55) form
            used = {(p.feature, p.op) for p in r.preds}
            dup = np.array([(p.feature, p.op) in used for p in preds])
            val = np.where((support >= min_support) & ~dup, val, -np.inf)
            # require a *relative* gain over the parent, so a third predicate
            # has to earn its place rather than shaving off a noisy subset
            val = np.where(val > r.value * (1.0 + min_gain), val, -np.inf)
            top = np.argsort(val)[::-1][:beam_width]
            for i in top:
                if not np.isfinite(val[i]):
                    continue
                candidates.append(Rule(
                    r.preds + (preds[i],), int(support[i]),
                    lift[i] * base, lift[i], float(val[i]), conj[i].copy(),
                ))
        if not candidates:
            break
        # dedup, keep best per condition-set, take top beam for next depth
        best: dict[frozenset, Rule] = {}
        for r in candidates:
            k = r.key()
            if k not in best or r.value > best[k].value:
                best[k] = r
        beam = sorted(best.values(), key=lambda r: r.value, reverse=True)[:beam_width]
        for r in beam:
            k = r.key()
            if k not in pool or r.value > pool[k].value:
                pool[k] = r

    return sorted(pool.values(), key=lambda r: r.value, reverse=True)


def refine_rule(
    rule: Rule,
    X_train: np.ndarray,
    Yw: np.ndarray,
    base: np.ndarray,
    objective: Objective,
    *,
    fine_bins: int = 100,
    min_support: int = 30,
) -> Rule:
    """Sharpen each predicate's threshold within +/- one coarse step.

    Coordinate ascent: for each predicate, scan fine percentiles in a local
    band around the coarse cut and keep the best-scoring threshold, holding the
    others fixed. Cheap because it runs on the (small) surviving rule only.
    """
    preds = list(rule.preds)
    coarse_step = 0.05  # local band half-width in percentile units
    fine_q = np.arange(1, fine_bins) / fine_bins

    for j, p in enumerate(preds):
        col = X_train[:, p.feature]
        lo, hi = p.pct - coarse_step, p.pct + coarse_step
        local_q = fine_q[(fine_q >= lo) & (fine_q <= hi)]
        best_p, best_val = p, -np.inf
        others = [q for k, q in enumerate(preds) if k != j]
        other_mask = np.ones(X_train.shape[0], dtype=bool)
        for q in others:
            other_mask &= q.mask(X_train)
        for q in local_q:
            v = float(np.quantile(col, q))
            cand = Predicate(p.feature, p.op, float(q), v)
            mask = other_mask & cand.mask(X_train)
            support, _, lift = score_one(mask, Yw, base)
            if support < min_support:
                continue
            val = float(objective(lift))
            if val > best_val:
                best_p, best_val = cand, val
        preds[j] = best_p

    mask = np.ones(X_train.shape[0], dtype=bool)
    for p in preds:
        mask &= p.mask(X_train)
    support, caught, lift = score_one(mask, Yw, base)
    return Rule(tuple(preds), support, caught, lift, float(objective(lift)), mask)


def evaluate_rule(rule: Rule, X: np.ndarray, Yw: np.ndarray, base: np.ndarray,
                  objective: Objective) -> Rule:
    """Re-score a rule on a different dataset (e.g. validation) -- honest lift."""
    mask = np.ones(X.shape[0], dtype=bool)
    for p in rule.preds:
        mask &= p.mask(X)
    support, caught, lift = score_one(mask, Yw, base)
    return Rule(rule.preds, support, caught, lift, float(objective(lift)), mask)
