"""arp -- the core rule miner used by the Bean pipeline.

This is the portable runtime subset: binned encoding, the histogram beam
(`fast`), the precision/recall-targeted beam (`targeted`), rule-level constraints
(`constraints`), the categorical-native miner (`mixed`), and progress reporting.
The gap-driven feature engineering lives in the separate `featgap` package
(one-directional dependency featgap -> arp).
"""

from .scoring import Score, base_rates
from .fast import BinSpec, FastRule, fit_bins, fast_beam_search, rule_mask
from .encode import (quantile_edges, assign_bins, target_rank, encode_split_cm,
                     encode_matrix_cm, encode_matrix_naive)
from .targeted import TargetedRule, targeted_beam_search
from .constraints import FeatureConstraint, RulePolicy
from .mixed import Meta, MixedRule, mixed_targeted_search
from .progress import Progress

__all__ = [
    "Score", "base_rates",
    "BinSpec", "FastRule", "fit_bins", "fast_beam_search", "rule_mask",
    "quantile_edges", "assign_bins", "target_rank", "encode_split_cm",
    "encode_matrix_cm", "encode_matrix_naive",
    "TargetedRule", "targeted_beam_search",
    "FeatureConstraint", "RulePolicy",
    "Meta", "MixedRule", "mixed_targeted_search",
    "Progress",
]
