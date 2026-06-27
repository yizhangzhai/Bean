"""Synthetic multi-type fraud data.

Each fraud *type* is generated from a hidden conjunctive rule over a couple of
features, so we have ground truth to check whether the miner recovers it.
Dollar losses are heavy-tailed (lognormal) to mirror real fraud, where a few
cases dominate total loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FraudData:
    X: np.ndarray            # (N, F) numerical features
    Y: np.ndarray            # (N, C) binary labels, one column per fraud type
    loss: np.ndarray         # (N, C) dollar loss (0 where not that fraud type)
    feature_names: list[str]
    type_names: list[str]
    ground_truth: list[str]  # human-readable hidden rule per type

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def any_fraud(self) -> np.ndarray:
        """Binary 'is this row fraud of any type' (N,)."""
        return (self.Y.sum(axis=1) > 0).astype(np.int64)


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q))


def make_fraud_data(
    n: int = 20_000,
    n_features: int = 20,
    seed: int = 0,
) -> FraudData:
    """Generate a tabular fraud dataset with 3 fraud types of differing rarity.

    The hidden signatures intentionally use sharp, localized bands and
    interactions so a single global threshold won't catch them -- this is what
    the conjunctive rule miner is meant to recover.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, n_features))
    # give a couple of features a heavier tail so percentile bins matter
    X[:, 0] = rng.lognormal(mean=0.0, sigma=1.0, size=n)   # e.g. amount
    X[:, 1] = rng.exponential(scale=1.0, size=n)           # e.g. velocity

    feature_names = [f"f{i:02d}" for i in range(n_features)]
    type_names = ["account_takeover", "collusion", "friendly_fraud"]
    C = len(type_names)

    Y = np.zeros((n, C), dtype=np.int64)
    loss = np.zeros((n, C), dtype=np.float64)

    # --- hidden signatures (the "decision paths" we hope to recover) ---
    # Type 0: sharp velocity spike AND high amount  (rare, expensive)
    sig0 = (X[:, 1] > _pct(X[:, 1], 0.95)) & (X[:, 0] > _pct(X[:, 0], 0.90))
    # Type 1: mid-band on f2 AND low f3  (interaction, non-monotonic in f2)
    sig1 = (
        (X[:, 2] > _pct(X[:, 2], 0.40)) & (X[:, 2] < _pct(X[:, 2], 0.55))
        & (X[:, 3] < _pct(X[:, 3], 0.10))
    )
    # Type 2: high f4 AND high f5  (common, cheap)
    sig2 = (X[:, 4] > _pct(X[:, 4], 0.80)) & (X[:, 5] > _pct(X[:, 5], 0.75))

    ground_truth = [
        "f01 > p95  AND  f00 > p90",
        "p40 < f02 < p55  AND  f03 < p10",
        "f04 > p80  AND  f05 > p75",
    ]

    # base firing probability inside the signature region (noisy labels)
    fire_prob = [0.85, 0.80, 0.70]
    # background false-positive rate outside the region (label noise)
    bg_prob = [0.001, 0.001, 0.004]

    for c, sig in enumerate((sig0, sig1, sig2)):
        p = np.where(sig, fire_prob[c], bg_prob[c])
        Y[:, c] = (rng.uniform(size=n) < p).astype(np.int64)

    # dollar loss: heavy-tailed, larger for the rarer/expensive types
    loss_scale = [9.0, 7.5, 6.0]  # lognormal mean (log dollars)
    for c in range(C):
        idx = Y[:, c] == 1
        loss[idx, c] = rng.lognormal(mean=loss_scale[c], sigma=1.1, size=idx.sum())

    return FraudData(
        X=X,
        Y=Y,
        loss=loss,
        feature_names=feature_names,
        type_names=type_names,
        ground_truth=ground_truth,
    )


def train_val_split(
    data: FraudData, val_frac: float = 0.33, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx). Mining on train, honest lift on val."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(data.n)
    cut = int(round(data.n * (1 - val_frac)))
    return perm[:cut], perm[cut:]
