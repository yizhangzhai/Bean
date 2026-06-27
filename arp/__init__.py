"""attention-rule-paths (arp): mine conjunctive 'decision paths' with a
parameter-free, label-as-query attention heuristic.

Pipeline: threshold-bin encode -> label-as-query scoring (mask @ Yw) ->
beam-search conjunctions -> coarse-to-fine refine -> per-type mine ->
type-balanced portfolio. See README.md for the design rationale.
"""

from .data import FraudData, make_fraud_data, train_val_split
from .encoding import Predicate, make_predicates, stack_masks
from .scoring import (
    base_rates,
    objective_fairness,
    objective_maximin,
    objective_single,
    objective_weighted,
    score_many,
)
from .search import Rule, beam_search, evaluate_rule, refine_rule
from .portfolio import Portfolio, build_portfolio

__all__ = [
    "FraudData", "make_fraud_data", "train_val_split",
    "Predicate", "make_predicates", "stack_masks",
    "base_rates", "score_many",
    "objective_single", "objective_weighted", "objective_maximin", "objective_fairness",
    "Rule", "beam_search", "refine_rule", "evaluate_rule",
    "Portfolio", "build_portfolio",
]
