"""Threshold-bin encoding = the 'decision stump' feature space.

Each predicate is a one-sided percentile threshold on one feature, e.g.
``f03 < p10``. Thresholds are computed on the *training* values and then
applied to any dataset (train or val) by recomputing the boolean mask, so we
never leak val statistics into the cut points.

We deliberately do NOT materialize the full nested-threshold matrix M. A
predicate carries only (feature, op, percentile, value); masks are computed on
demand and stacked into a (n_preds, N) boolean matrix for fast vectorized
scoring (mask @ Y).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Predicate:
    feature: int
    op: str          # '<' or '>'
    pct: float       # percentile in [0, 1], for display / refinement
    value: float     # the actual threshold on the feature scale

    def mask(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.feature]
        return col < self.value if self.op == "<" else col > self.value

    def label(self, feature_names: list[str]) -> str:
        return f"{feature_names[self.feature]} {self.op} p{int(round(self.pct * 100)):02d}"


def quantile_grid(n_bins: int) -> np.ndarray:
    """Interior quantile cut points, e.g. n_bins=10 -> [0.1, 0.2, ..., 0.9]."""
    return np.arange(1, n_bins) / n_bins


def make_predicates(
    X_train: np.ndarray,
    n_bins: int = 10,
    both_sides: bool = True,
) -> list[Predicate]:
    """Build the candidate predicate set from training quantiles.

    n_bins=10 gives a coarse decile scan (9 cut points/feature); refinement
    later sharpens the winners. Set n_bins=100 for a flat full-resolution scan.
    """
    qs = quantile_grid(n_bins)
    preds: list[Predicate] = []
    for f in range(X_train.shape[1]):
        col = X_train[:, f]
        for q in qs:
            v = float(np.quantile(col, q))
            preds.append(Predicate(f, ">", q, v))
            if both_sides:
                preds.append(Predicate(f, "<", q, v))
    return preds


def stack_masks(preds: list[Predicate], X: np.ndarray) -> np.ndarray:
    """(n_preds, N) boolean matrix -- the rows of M we actually use."""
    out = np.empty((len(preds), X.shape[0]), dtype=bool)
    for i, p in enumerate(preds):
        out[i] = p.mask(X)
    return out
