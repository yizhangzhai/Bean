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


def make_fraud_data_large(
    n: int = 2_000_000,
    n_features: int = 1000,
    seed: int = 0,
    block: int = 100,
) -> FraudData:
    """Memory-safe large generator: float32 X filled in column blocks.

    Same three hidden signatures as make_fraud_data, planted in features 0-5;
    the remaining n_features-6 columns are pure noise -- a needle-in-haystack
    test of whether the miner finds the right few features among many.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, n_features), dtype=np.float32)
    for start in range(0, n_features, block):
        end = min(start + block, n_features)
        X[:, start:end] = rng.standard_normal((n, end - start), dtype=np.float32)
    # heavy-tailed signal features
    X[:, 0] = rng.lognormal(0.0, 1.0, size=n).astype(np.float32)
    X[:, 1] = rng.exponential(1.0, size=n).astype(np.float32)

    feature_names = [f"f{i:04d}" for i in range(n_features)]
    type_names = ["account_takeover", "collusion", "friendly_fraud"]

    def pc(j, q):
        return np.quantile(X[:, j], q)

    sig0 = (X[:, 1] > pc(1, 0.95)) & (X[:, 0] > pc(0, 0.90))
    sig1 = ((X[:, 2] > pc(2, 0.40)) & (X[:, 2] < pc(2, 0.55)) & (X[:, 3] < pc(3, 0.10)))
    sig2 = (X[:, 4] > pc(4, 0.80)) & (X[:, 5] > pc(5, 0.75))

    ground_truth = [
        "f0001 > p95  AND  f0000 > p90",
        "p40 < f0002 < p55  AND  f0003 < p10",
        "f0004 > p80  AND  f0005 > p75",
    ]
    fire_prob = [0.85, 0.80, 0.70]
    bg_prob = [0.001, 0.001, 0.004]
    loss_scale = [9.0, 7.5, 6.0]

    Y = np.zeros((n, 3), dtype=np.int64)
    loss = np.zeros((n, 3), dtype=np.float64)
    for c, sig in enumerate((sig0, sig1, sig2)):
        p = np.where(sig, fire_prob[c], bg_prob[c])
        Y[:, c] = (rng.uniform(size=n) < p).astype(np.int64)
        idx = Y[:, c] == 1
        loss[idx, c] = rng.lognormal(loss_scale[c], 1.1, size=int(idx.sum()))

    return FraudData(X, Y, loss, feature_names, type_names, ground_truth)


def make_binned_fraud_data_large(
    n: int = 2_000_000,
    n_features: int = 1000,
    n_bins: int = 10,
    seed: int = 0,
    block: int = 50,
):
    """Fused generate+bin: never holds the full float X in memory.

    Generates each column block, bins it to int8 immediately, and discards the
    floats -- so peak memory is Xbin (N*F bytes) + one block, not the 8GB float
    matrix. Returns (Xbin, BinSpec, Y, loss, feature_names, type_names,
    ground_truth). This is the memory-pressure fix for the 78-min fit_bins seen
    when the float X and Xbin were both resident at 2M x 1000.
    """
    from .fast import BinSpec

    rng = np.random.default_rng(seed)
    qs = np.arange(1, n_bins) / n_bins
    Xbin = np.empty((n, n_features), dtype=np.int8)
    edges: list[np.ndarray] = [None] * n_features

    def bin_col(j, col):
        e = np.quantile(col, qs)
        edges[j] = e.astype(np.float64)
        Xbin[:, j] = np.searchsorted(e, col, side="right").astype(np.int8)

    # signal features kept as float just long enough to define signatures
    f0 = rng.lognormal(0.0, 1.0, size=n).astype(np.float32)
    f1 = rng.exponential(1.0, size=n).astype(np.float32)
    f2, f3, f4, f5 = (rng.standard_normal(n, dtype=np.float32) for _ in range(4))

    def pc(col, q):
        return np.quantile(col, q)

    sig0 = (f1 > pc(f1, 0.95)) & (f0 > pc(f0, 0.90))
    sig1 = (f2 > pc(f2, 0.40)) & (f2 < pc(f2, 0.55)) & (f3 < pc(f3, 0.10))
    sig2 = (f4 > pc(f4, 0.80)) & (f5 > pc(f5, 0.75))

    for j, col in enumerate((f0, f1, f2, f3, f4, f5)):
        bin_col(j, col)

    type_names = ["account_takeover", "collusion", "friendly_fraud"]
    fire_prob, bg_prob, loss_scale = [0.85, 0.80, 0.70], [0.001, 0.001, 0.004], [9.0, 7.5, 6.0]
    Y = np.zeros((n, 3), dtype=np.int64)
    loss = np.zeros((n, 3), dtype=np.float64)
    for c, sig in enumerate((sig0, sig1, sig2)):
        p = np.where(sig, fire_prob[c], bg_prob[c])
        Y[:, c] = (rng.uniform(size=n) < p).astype(np.int64)
        idx = Y[:, c] == 1
        loss[idx, c] = rng.lognormal(loss_scale[c], 1.1, size=int(idx.sum()))
    del f0, f1, f2, f3, f4, f5

    # remaining noise features: generate -> bin -> discard, block by block
    for start in range(6, n_features, block):
        end = min(start + block, n_features)
        Xf = rng.standard_normal((n, end - start), dtype=np.float32)
        for jj in range(end - start):
            bin_col(start + jj, Xf[:, jj])
        del Xf

    feature_names = [f"f{i:04d}" for i in range(n_features)]
    ground_truth = [
        "f0001 > p95  AND  f0000 > p90",
        "p40 < f0002 < p55  AND  f0003 < p10",
        "f0004 > p80  AND  f0005 > p75",
    ]
    return Xbin, BinSpec(edges, qs, n_bins), Y, loss, feature_names, type_names, ground_truth


# ground-truth signal features per type, for recovery scoring
GROUND_TRUTH_FEATURES = {
    "account_takeover": {0, 1},
    "collusion": {2, 3},
    "friendly_fraud": {4, 5},
}


def train_val_split(
    data: FraudData, val_frac: float = 0.33, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx). Mining on train, honest lift on val."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(data.n)
    cut = int(round(data.n * (1 - val_frac)))
    return perm[:cut], perm[cut:]
