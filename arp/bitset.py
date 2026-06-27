"""Coarse-to-fine feature pruning + bitset conjunction scoring.

Targets the `mine` bottleneck. Two ideas, composed:

1. **Coarse-to-fine (feature pruning).** A cheap depth-1 pass -- coarse
   thresholds, optionally on a *row sample* -- ranks features by univariate
   lift. Only the top-k features are carried into the expensive deep beam
   search. In the histogram design, depth-1 cost is dominated by the bincount
   over N, not the number of thresholds, so the leverage is cutting F (e.g.
   1000 -> 64) before expansion, not shrinking bins per se.

2. **Bitset membership (count mode).** Each surviving predicate's membership is
   packed to bits (np.packbits). A rule's mask = bitwise-AND of predicate
   bitsets; support = popcount; caught_c = popcount(mask & class_c_bits). This
   removes the subset-gather + re-histogram of the float/int path -- expansion
   becomes vectorized AND + np.bitwise_count over N/8 bytes.

   Limitation: popcount sums *binary* membership, so this accelerates
   count-based lift only. Dollar-weighted lift must stay on the histogram path
   (weights aren't binary). Mining on counts then ranking the survivors by
   dollars on validation is the usual split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fast import BinSpec, FastRule


def _popcount_rows(packed: np.ndarray) -> np.ndarray:
    """Total set bits per row of a (P, W) uint8 matrix -> (P,)."""
    return np.bitwise_count(packed).sum(axis=1, dtype=np.int64)


def rank_features(
    Xbin: np.ndarray,
    Yw: np.ndarray,
    base: np.ndarray,
    target: int,
    n_bins: int,
    *,
    coarse_step: int = 2,
    sample_rows: int | None = None,
    min_support: int = 40,
    seed: int = 0,
) -> np.ndarray:
    """Rank features by best single-threshold lift on the target type.

    coarse_step>1 skips thresholds (coarse); sample_rows subsamples instances
    (coarse in N too). Returns feature indices, best-first.
    """
    N, F = Xbin.shape
    if sample_rows and sample_rows < N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=sample_rows, replace=False)
        Xs, Ys = Xbin[idx], Yw[idx]
    else:
        Xs, Ys = Xbin, Yw
    Ns = Xs.shape[0]
    yc = Ys[:, target]
    best = np.full(F, -np.inf)
    ks = range(0, n_bins - 1, coarse_step)
    for f in range(F):
        col = Xs[:, f]
        counts = np.bincount(col, minlength=n_bins).astype(np.float64)
        cc = np.cumsum(counts)
        wc = np.cumsum(np.bincount(col, weights=yc, minlength=n_bins))
        tot = wc[-1]
        for k in ks:
            for s, cgt in ((cc[k], wc[k]), (Ns - cc[k], tot - wc[k])):
                if s >= min_support * Ns / N:
                    lift = (cgt / s) / base[target] if base[target] > 0 else 0.0
                    if lift > best[f]:
                        best[f] = lift
    return np.argsort(best)[::-1]


@dataclass
class BitPreds:
    bits: np.ndarray            # (P, W) uint8 packed membership
    meta: list                  # P x (feature, op, k)
    yc: np.ndarray              # (C, W) uint8 packed class membership
    n: int
    W: int = field(default=0)

    def __post_init__(self):
        self.W = self.bits.shape[1]


def build_bitpreds(Xbin, Y, feature_subset, n_bins, *, min_support=40) -> BitPreds:
    N = Xbin.shape[0]
    bitlist, meta = [], []
    for f in feature_subset:
        col = Xbin[:, f]
        for k in range(n_bins - 1):
            above = col > k
            below = ~above
            for op, m in (("<", below), (">", above)):
                if int(m.sum()) >= min_support:
                    bitlist.append(np.packbits(m))
                    meta.append((int(f), op, int(k)))
    bits = np.vstack(bitlist)
    yc = np.vstack([np.packbits(Y[:, c] == 1) for c in range(Y.shape[1])])
    return BitPreds(bits, meta, yc, N)


def _score_against(base_bits, bp: BitPreds, base_rates, N):
    """Score every predicate AND base_bits. Returns (support (P,), lift (P,C))."""
    conj = bp.bits & base_bits[None, :]            # (P, W)
    support = _popcount_rows(conj)                  # (P,)
    C = bp.yc.shape[0]
    caught = np.empty((conj.shape[0], C))
    for c in range(C):
        caught[:, c] = _popcount_rows(conj & bp.yc[c][None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(support[:, None] > 0, caught / support[:, None], 0.0)
        lift = np.where(base_rates[None, :] > 0, prec / base_rates[None, :], 0.0)
    return support, lift, conj


def bitset_beam_search(
    bp: BitPreds,
    base_rates: np.ndarray,
    objective,
    spec: BinSpec,
    *,
    beam_width: int = 8,
    max_depth: int = 3,
    min_support: int = 40,
    min_gain: float = 0.10,
) -> list[FastRule]:
    """Beam search entirely in bitset space (AND + popcount, no gather)."""
    N = bp.n
    W = bp.W
    full = np.full(W, 0xFF, dtype=np.uint8)         # all-ones base

    support, lift, conj = _score_against(full, bp, base_rates, N)
    val = np.where(support >= min_support, objective(lift), -np.inf)
    order = np.argsort(val)[::-1][:beam_width]

    @dataclass
    class _R:
        preds: tuple
        bits: np.ndarray
        support: int
        lift: np.ndarray
        value: float
        def used(self): return {(f, op) for f, op, _ in self.preds}
        def key(self): return frozenset(self.preds)

    beam = []
    for i in order:
        if not np.isfinite(val[i]):
            continue
        beam.append(_R((bp.meta[i],), conj[i].copy(), int(support[i]),
                       lift[i], float(val[i])))
    pool = {r.key(): r for r in beam}

    for _d in range(2, max_depth + 1):
        cand = []
        for r in beam:
            support, lift, conj = _score_against(r.bits, bp, base_rates, N)
            val = objective(lift)
            used = r.used()
            dup = np.array([(f, op) in used for f, op, _ in bp.meta])
            val = np.where((support >= min_support) & ~dup, val, -np.inf)
            val = np.where(val > r.value * (1.0 + min_gain), val, -np.inf)
            top = np.argsort(val)[::-1][:beam_width]
            for i in top:
                if not np.isfinite(val[i]):
                    continue
                cand.append(_R(r.preds + (bp.meta[i],), conj[i].copy(),
                               int(support[i]), lift[i], float(val[i])))
        if not cand:
            break
        best = {}
        for r in cand:
            k = r.key()
            if k not in best or r.value > best[k].value:
                best[k] = r
        beam = sorted(best.values(), key=lambda r: r.value, reverse=True)[:beam_width]
        for r in beam:
            if r.key() not in pool or r.value > pool[r.key()].value:
                pool[r.key()] = r

    out = []
    for r in sorted(pool.values(), key=lambda r: r.value, reverse=True):
        out.append(FastRule(tuple(r.preds), r.support, r.lift * base_rates,
                            r.lift, r.value))
    return out


def coarse_to_fine_mine(
    Xbin, Y, base, objective, target, spec, *,
    top_k=64, sample_rows=None, coarse_step=2,
    beam_width=8, max_depth=3, min_support=40,
) -> tuple[list[FastRule], dict]:
    """Full pipeline: coarse rank features -> bitset deep search on top-k.

    Returns (rules, timing_dict).
    """
    import time
    Yw = Y.astype(np.float64)
    t = time.perf_counter()
    ranked = rank_features(Xbin, Yw, base, target, spec.n_bins,
                           coarse_step=coarse_step, sample_rows=sample_rows,
                           min_support=min_support)
    keep = ranked[:top_k]
    t_rank = time.perf_counter() - t

    t = time.perf_counter()
    bp = build_bitpreds(Xbin, Y, keep, spec.n_bins, min_support=min_support)
    t_build = time.perf_counter() - t

    t = time.perf_counter()
    rules = bitset_beam_search(bp, base, objective, spec, beam_width=beam_width,
                               max_depth=max_depth, min_support=min_support)
    t_search = time.perf_counter() - t
    return rules, {"rank": t_rank, "build_bits": t_build, "search": t_search,
                   "kept_features": int(top_k), "n_predicates": len(bp.meta)}
