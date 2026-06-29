"""attention-rule-paths (arp): mine conjunctive 'decision paths' with a
parameter-free, label-as-query attention heuristic.

Pipeline: threshold-bin encode -> label-as-query scoring (mask @ Yw) ->
beam-search conjunctions -> coarse-to-fine refine -> per-type mine ->
type-balanced portfolio. See README.md for the design rationale.
"""

from .data import (
    FraudData, make_fraud_data, make_fraud_data_large,
    make_binned_fraud_data_large, train_val_split,
)
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
# scalable + practical layers
from .fast import BinSpec, FastRule, fit_bins, fast_beam_search
from .encode import (quantile_edges, assign_bins, target_rank, encode_split_cm,
                     encode_matrix_cm, encode_matrix_naive)
from .bitset import coarse_to_fine_mine, bitset_beam_search
from .targeted import TargetedRule, targeted_beam_search
from .constraints import FeatureConstraint, RulePolicy
from .mixed import Meta, MixedRule, mixed_targeted_search
from .progress import Progress
# NOTE: gap-driven feature engineering lives in the separate `featgap` package
# (one-directional dependency featgap -> arp), not in the core miner.

__all__ = [
    # data
    "FraudData", "make_fraud_data", "make_fraud_data_large",
    "make_binned_fraud_data_large", "train_val_split",
    # reference scoring / search
    "Predicate", "make_predicates", "stack_masks",
    "base_rates", "score_many",
    "objective_single", "objective_weighted", "objective_maximin", "objective_fairness",
    "Rule", "beam_search", "refine_rule", "evaluate_rule",
    "Portfolio", "build_portfolio",
    # scalable miner
    "BinSpec", "FastRule", "fit_bins", "fast_beam_search",
    "coarse_to_fine_mine", "bitset_beam_search",
    # fast encoding (sampled edges + column-major + threaded)
    "quantile_edges", "assign_bins", "target_rank", "encode_split_cm",
    "encode_matrix_cm", "encode_matrix_naive",
    # targeted (precision/recall) + constraints
    "TargetedRule", "targeted_beam_search",
    "FeatureConstraint", "RulePolicy",
    # mixed numeric + categorical
    "Meta", "MixedRule", "mixed_targeted_search",
    # progress reporting
    "Progress",
]
