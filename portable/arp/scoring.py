"""Label-as-query scoring -- the parameter-free 'attention'.

The whole supervised signal is one matmul:

    caught = mask.astype(float) @ Yw        # (C,)  per fraud type

where ``Yw`` is the (N, C) weight matrix: either binary labels Y (count-based
lift) or dollar loss (value-based lift). The query is the label/loss column;
the key is the bin membership. No learned W_Q / W_K / W_V -- with identity
projections, QK^T is exactly this co-occurrence/alignment, and it is the thing
we actually want to read off as a rule's strength.

Lift_c = precision_c / base_rate_c, computed against per-type base rates so an
imbalanced or heavy-tailed type is judged on its own scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Score:
    support: int
    caught: np.ndarray      # (C,) count or $ captured per type
    lift: np.ndarray        # (C,) lift per type
    value: float            # scalar objective value used for ranking


def base_rates(Yw: np.ndarray, n: int) -> np.ndarray:
    """Per-type base rate: mean label, or mean $ loss per row."""
    return Yw.sum(axis=0) / n


def score_many(
    masks: np.ndarray,        # (n_preds, N) bool  -- candidates to score
    Yw: np.ndarray,           # (N, C) weights
    base: np.ndarray,         # (C,)
    n_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized: returns (support (n_preds,), lift (n_preds, C))."""
    m = masks.astype(np.float64)
    support = m.sum(axis=1)                      # (n_preds,)
    caught = m @ Yw                              # (n_preds, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(support[:, None] > 0, caught / support[:, None], 0.0)
        lift = np.where(base[None, :] > 0, precision / base[None, :], 0.0)
    return support, lift


def score_one(mask: np.ndarray, Yw: np.ndarray, base: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    support = int(mask.sum())
    caught = mask.astype(np.float64) @ Yw
    precision = caught / support if support > 0 else np.zeros_like(caught)
    lift = np.where(base > 0, precision / base, 0.0)
    return support, caught, lift


# ---- objectives: collapse a per-type lift vector into one ranking scalar ----

def objective_single(lift: np.ndarray, target: int) -> np.ndarray:
    """Lift on one target type (use this for per-type mining)."""
    return lift[..., target]


def objective_weighted(lift: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Business-weighted sum of per-type lift."""
    return lift @ alpha


def objective_maximin(lift: np.ndarray) -> np.ndarray:
    """Worst-covered type -- balances, but chases the rarest type (overfits)."""
    return lift.min(axis=-1)


def objective_fairness(lift: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Mean lift penalized by spread -- rewards generalist rules."""
    return lift.mean(axis=-1) - lam * lift.std(axis=-1)
