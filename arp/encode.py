"""Fast feature encoding: float columns -> int8 quantile bins.

Encoding (quantizing every feature into bin indices) is a one-time preprocessing
step, but on wide data it dominates wall-clock -- and almost all of that cost is
avoidable. Three fixes over a naive per-column ``np.quantile`` + strided write:

1. **Sampled edges.** Bin cut-points are percentile *statistics*; you do not need
   all N rows to locate them. A 50-100k-row sub-sample estimates each quantile to
   well within a bin width (sample-quantile standard error ~ sqrt(p(1-p)/n)/dens),
   so the per-column sort drops from O(N log N) to O(s log s) with s << N. Free:
   no measurable accuracy loss for equal-frequency bins.

2. **Column-major output.** Storing the matrix as ``(F, N)`` C-contiguous makes
   each column's write a single contiguous run instead of an N-stride scatter
   across the whole (multi-GB) matrix -- which otherwise misses cache on every
   element. Callers get ``buf.T`` (an ``(N, F)`` view) so the *miner* sees the
   usual layout, and its ``Xbin[:, f]`` column slices are contiguous too (the
   read side of the search also speeds up). Free: exact same bins.

3. **Threaded.** Columns are independent and numpy releases the GIL on
   sort/searchsorted, so a plain thread pool gives near-linear multicore speedup.
   Free: exact same bins.

The irreducible floor is one contiguous write of the N*F int8 matrix plus an
O(N) assignment per column; everything above that floor is what these remove.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def quantile_edges(col: np.ndarray, qs: np.ndarray, *, sample: int = 0,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Interior quantile cut-points of ``col`` at fractions ``qs``.

    If ``sample`` > 0 and the column is larger, estimate the cut-points from a
    random sub-sample (with replacement -- cheap, and quantiles are insensitive
    to it). Returns float64 edges.
    """
    if sample and col.shape[0] > sample:
        rng = rng or np.random.default_rng(0)
        col = col[rng.integers(0, col.shape[0], sample)]
    return np.quantile(col, qs).astype(np.float64)


def assign_bins(col: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per value (int8): index b means edges[b-1] <= x < edges[b]."""
    return np.searchsorted(edges, col, side="right").astype(np.int8)


def target_rank(codes: np.ndarray, y: np.ndarray, n_codes: int | None = None,
                *, smoothing: float = 20.0) -> np.ndarray:
    """Fisher-trick ordinal re-code of a *categorical* column: rank its categories
    by SMOOTHED fraud rate, so a threshold split (`code_rank > k`) selects the
    highest-rate categories -- the OPTIMAL subset split for a binary target. This
    lets a nominal categorical be mined as an ordinal in the fast bin/threshold
    pipeline and recover `cat in {…}` rules, not just `cat == c`.

    Smoothing (and computing on TRAIN only) controls target-encoding leakage; the
    held-out precision gate catches what slips through. Returns `rank_of_code`
    (array indexed by category code).
    """
    codes = codes.astype(np.int64)
    K = n_codes or int(codes.max()) + 1
    cnt = np.bincount(codes, minlength=K).astype(float)
    pos = np.bincount(codes, weights=y.astype(float), minlength=K)
    base = float(y.mean())
    rate = (pos + smoothing * base) / (cnt + smoothing)      # smoothed fraud rate
    rank = np.empty(K, dtype=np.int64)
    rank[np.argsort(rate)] = np.arange(K)                    # low rate -> low rank
    return rank


def _threads(n_threads):
    return n_threads or min(os.cpu_count() or 4, 8)


def encode_split_cm(make_column, n_features, tr, va, *, n_bins=16,
                    sample=100_000, n_threads=None, seed=0):
    """Encode every feature for a train/val split into column-major buffers.

    ``make_column(f) -> (N,) float`` yields column f's raw values (the caller
    owns generation / IO). Edges are fit on the TRAIN rows only (a sub-sample of
    them) -- no leakage -- and applied to both splits. Columns are encoded in
    parallel threads.

    Returns ``(Xtr_view, Xva_view, BinSpec)`` where the views are ``(n_tr, F)``
    and ``(n_va, F)`` -- transposed views of ``(F, n)`` C-contiguous buffers, so
    column slices stay contiguous for the miner.
    """
    from .fast import BinSpec
    n_tr, n_va = len(tr), len(va)
    btr = np.empty((n_features, n_tr), dtype=np.int8)
    bva = np.empty((n_features, n_va), dtype=np.int8)
    qs = np.arange(1, n_bins) / n_bins
    edges: list = [None] * n_features
    seeds = np.random.SeedSequence(seed).spawn(n_features)

    def work(f):
        col = make_column(f)
        e = quantile_edges(col[tr], qs, sample=sample,
                           rng=np.random.default_rng(seeds[f]))
        edges[f] = e
        btr[f, :] = assign_bins(col[tr], e)
        bva[f, :] = assign_bins(col[va], e)

    with ThreadPoolExecutor(_threads(n_threads)) as ex:
        list(ex.map(work, range(n_features)))
    return btr.T, bva.T, BinSpec(edges, qs, n_bins)


def encode_matrix_cm(Xf_FN, *, n_bins=16, sample=100_000, n_threads=None, seed=0):
    """Fast encode of a column-major float matrix ``Xf_FN`` of shape (F, N).

    Sampled edges + contiguous column-major write + threaded. Returns
    ``(Xbin_view (N,F), BinSpec)``.
    """
    from .fast import BinSpec
    F, N = Xf_FN.shape
    buf = np.empty((F, N), dtype=np.int8)
    qs = np.arange(1, n_bins) / n_bins
    edges: list = [None] * F
    seeds = np.random.SeedSequence(seed).spawn(F)

    def work(f):
        e = quantile_edges(Xf_FN[f], qs, sample=sample,
                           rng=np.random.default_rng(seeds[f]))
        edges[f] = e
        buf[f, :] = assign_bins(Xf_FN[f], e)

    with ThreadPoolExecutor(_threads(n_threads)) as ex:
        list(ex.map(work, range(F)))
    return buf.T, BinSpec(edges, qs, n_bins)


def encode_matrix_naive(Xf_FN, *, n_bins=16):
    """Baseline: serial, full-data quantile, strided ``(N, F)`` row-major write.

    Reads the same column-major float input as ``encode_matrix_cm`` (fair), but
    reproduces the slow path -- this is the reference the optimizations beat.
    """
    from .fast import BinSpec
    F, N = Xf_FN.shape
    out = np.empty((N, F), dtype=np.int8)             # row-major -> strided cols
    qs = np.arange(1, n_bins) / n_bins
    edges: list = []
    for f in range(F):
        e = np.quantile(Xf_FN[f], qs).astype(np.float64)   # full-data sort
        edges.append(e)
        out[:, f] = np.searchsorted(e, Xf_FN[f], side="right").astype(np.int8)
    return out, BinSpec(edges, qs, n_bins)
