"""Scalable miner -- the version that survives 2M x 1000.

Two changes vs. the reference prototype in search.py:

1. **No dense M.** Features are pre-binned once into an int8 matrix ``Xbin``
   (N x F). All scoring is histogram/cumsum over bin indices (``np.bincount``),
   so a depth-1 scan of every threshold of every feature costs O(N*C) per
   feature with NO per-predicate mask materialization.

2. **Subset rescan for conjunctions.** To extend a rule we gather only the rows
   it already matches (its support set) and re-histogram each feature on that
   subset -- exactly recursive partitioning. Cost scales with the (small) rule
   support, not N.

Memory beyond ``Xbin`` (which is ~F bytes/row) is O(N). This is the same
histogram trick that lets LightGBM scale to billions of rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scoring import Score  # noqa: F401  (re-exported convenience)


@dataclass
class BinSpec:
    edges: list[np.ndarray]      # edges[f] = interior quantile cut values
    pct: np.ndarray              # (B-1,) percentile of each edge, e.g. .1,.2,...
    n_bins: int
    active: list = None          # per-feature array of ACTIVE split-k indices
                                 # (None => all thresholds active, as before)

    def active_for(self, f):
        return None if self.active is None else self.active[f]


def _chimerge_active(xb, Y, B, chi2_thresh):
    """Supervised threshold selection (ChiMerge, union across label types).

    Merge adjacent fine bins whose per-type fraud rates are not statistically
    distinguishable (chi-square < threshold for ALL types); the surviving bin
    boundaries are the active split indices. Flat/degenerate ranges collapse to
    nothing; cut points survive only where the label rate genuinely changes.
    """
    C = Y.shape[1]
    counts = np.bincount(xb, minlength=B).astype(np.float64)
    pos = [np.bincount(xb, weights=Y[:, c].astype(np.float64), minlength=B)
           for c in range(C)]
    segs = [[i, i, counts[i], [pos[c][i] for c in range(C)]] for i in range(B)]

    def chi2(a, b):                                   # max chi2 over types (union)
        na, nb = a[2], b[2]
        N = na + nb
        if N == 0:
            return 0.0
        best = 0.0
        for c in range(C):
            ap, bp = a[3][c], b[3][c]
            c1 = ap + bp
            c2 = N - c1
            if na == 0 or nb == 0 or c1 == 0 or c2 == 0:
                continue
            v = 0.0
            for o, r, cc in ((ap, na, c1), (na - ap, na, c2),
                             (bp, nb, c1), (nb - bp, nb, c2)):
                e = r * cc / N
                if e > 0:
                    v += (o - e) ** 2 / e
            best = max(best, v)
        return best

    while len(segs) > 1:
        chis = [chi2(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
        mi = int(np.argmin(chis))
        if chis[mi] >= chi2_thresh:
            break
        a, b = segs[mi], segs[mi + 1]
        segs[mi] = [a[0], b[1], a[2] + b[2],
                    [a[3][c] + b[3][c] for c in range(C)]]
        del segs[mi + 1]
    return np.array([segs[i][1] for i in range(len(segs) - 1)], dtype=np.int64)


def fit_bins(X: np.ndarray, n_bins: int = 10, *, Y=None, supervised: bool = False,
             chi2: float = 6.63) -> tuple[np.ndarray, BinSpec]:
    """Bin every feature by its training quantiles -> (Xbin int8, BinSpec).

    Bin index b means edges[b-1] <= x < edges[b]. Threshold split index k in
    0..B-2 corresponds to the cut at percentile (k+1)/n_bins.

    If `supervised` and `Y` (N x C labels) are given, also compute a per-feature
    ACTIVE threshold mask via chi-square-gated bin merging (ChiMerge): thresholds
    in flat / uninformative ranges are dropped so the search never generates
    them. Lossless for structure; for the label it keeps exactly the cut points
    where the fraud rate changes (which, for conjunctions, are the boundaries
    that matter -- see README). Compute on TRAIN only to avoid leakage.
    """
    n, f = X.shape
    qs = np.arange(1, n_bins) / n_bins
    Xbin = np.empty((n, f), dtype=np.int8)
    edges: list[np.ndarray] = []
    active = [] if (supervised and Y is not None) else None
    for j in range(f):
        col = X[:, j]
        e = np.quantile(col, qs)
        edges.append(e.astype(np.float64))
        xb = np.searchsorted(e, col, side="right").astype(np.int8)
        Xbin[:, j] = xb
        if active is not None:
            active.append(_chimerge_active(xb, Y, n_bins, chi2))
    return Xbin, BinSpec(edges, qs, n_bins, active)


@dataclass
class FastRule:
    preds: tuple[tuple[int, str, int], ...]   # (feature, op, split_k)
    support: int
    caught: np.ndarray
    lift: np.ndarray
    value: float
    mask: np.ndarray = field(repr=False, default=None)

    def features_ops(self) -> set[tuple[int, str]]:
        return {(f, op) for f, op, _ in self.preds}

    def features(self) -> set[int]:
        return {f for f, _, _ in self.preds}

    def key(self):
        return frozenset(self.preds)

    def label(self, feature_names, spec: BinSpec) -> str:
        parts = []
        for f, op, k in self.preds:
            pct = int(round(spec.pct[k] * 100))
            parts.append(f"{feature_names[f]} {op} p{pct:02d}")
        return "  AND  ".join(parts)


def rule_mask(preds, Xbin) -> np.ndarray:
    m = np.ones(Xbin.shape[0], dtype=bool)
    for f, op, k in preds:
        col = Xbin[:, f]
        m &= (col > k) if op == ">" else (col <= k)
    return m


def _hist_scores(xb, Yw, base, n_bins, min_support, n_for_base, active=None):
    """Return candidate (op, k, support, lift[C]) for one feature's bin column.

    xb: int8 bin indices (subset or full). Yw: (len(xb), C) weights.
    `active`: optional array of split indices k to consider (else all 0..B-2).
    """
    B = n_bins
    counts = np.bincount(xb, minlength=B).astype(np.float64)      # per bin
    cc = np.cumsum(counts)                                         # <= k counts
    C = Yw.shape[1]
    wsum = np.empty((B, C))
    for c in range(C):
        wsum[:, c] = np.bincount(xb, weights=Yw[:, c], minlength=B)
    wc = np.cumsum(wsum, axis=0)                                   # <= k weighted
    total_w = wc[-1]
    N = xb.shape[0]
    out = []
    ks = range(B - 1) if active is None else active
    for k in ks:
        s_below, c_below = cc[k], wc[k]
        s_above, c_above = N - cc[k], total_w - wc[k]
        for op, s, cgt in (("<", s_below, c_below), (">", s_above, c_above)):
            if s >= min_support:
                lift = np.where(base > 0, (cgt / s) / base, 0.0)
                out.append((op, k, int(s), lift))
    return out


def fast_beam_search(
    Xbin: np.ndarray,
    Yw: np.ndarray,
    base: np.ndarray,
    objective,
    spec: BinSpec,
    *,
    beam_width: int = 8,
    max_depth: int = 3,
    min_support: int = 40,
    min_gain: float = 0.10,
    progress=None,
) -> list[FastRule]:
    from .progress import make_progress
    N, F = Xbin.shape
    nb = spec.n_bins
    prog = make_progress(progress, "fast")
    prog.start(N=f"{N:,}", F=F, beam=beam_width, max_depth=max_depth)

    # ---- depth 1: histogram scan over every feature ----
    cands: list[FastRule] = []
    tick = max(1, F // 5)
    for f in range(F):
        af = spec.active_for(f)
        if af is not None and len(af) == 0:
            continue
        for op, k, s, lift in _hist_scores(Xbin[:, f], Yw, base, nb, min_support, N, af):
            cands.append(FastRule(((f, op, k),), s, lift * base, lift,
                                  float(objective(lift))))
        if prog.enabled and (f + 1) % tick == 0:
            prog.tick(f"depth-1 scan {f + 1:,}/{F:,} features  "
                      f"({len(cands):,} candidates)  {prog.elapsed():.1f}s")
    cands.sort(key=lambda r: r.value, reverse=True)
    beam = cands[:beam_width]
    for r in beam:
        r.mask = rule_mask(r.preds, Xbin)
    pool = {r.key(): r for r in beam}
    prog.depth(1, beam_in=1, beam_out=len(beam), explored=len(cands),
               accepted_total=len(pool),
               best_p=beam[0].value if beam else 0.0, best_r=None)

    # ---- grow via subset rescan ----
    for _depth in range(2, max_depth + 1):
        explored = 0
        nxt: list[FastRule] = []
        for r in beam:
            idx = np.flatnonzero(r.mask)
            subYw = Yw[idx]
            used = r.features_ops()
            for f in range(F):
                af = spec.active_for(f)
                if af is not None and len(af) == 0:
                    continue
                xb = Xbin[idx, f]
                for op, k, s, lift in _hist_scores(xb, subYw, base, nb,
                                                   min_support, N, af):
                    if (f, op) in used:
                        continue
                    explored += 1
                    val = float(objective(lift))
                    if val <= r.value * (1.0 + min_gain):
                        continue
                    preds = r.preds + ((f, op, k),)
                    nr = FastRule(preds, s, lift * base, lift, val)
                    nxt.append(nr)
        if not nxt:
            break
        best = {}
        for r in nxt:
            kkey = r.key()
            if kkey not in best or r.value > best[kkey].value:
                best[kkey] = r
        beam_in = len(beam)
        beam = sorted(best.values(), key=lambda r: r.value, reverse=True)[:beam_width]
        for r in beam:
            r.mask = rule_mask(r.preds, Xbin)
            if r.key() not in pool or r.value > pool[r.key()].value:
                pool[r.key()] = r
        prog.depth(_depth, beam_in=beam_in, beam_out=len(beam), explored=explored,
                   accepted_total=len(pool),
                   best_p=beam[0].value if beam else 0.0, best_r=None)

    out = sorted(pool.values(), key=lambda r: r.value, reverse=True)
    prog.done(rules=len(out), note=f"top score={out[0].value:.1f}" if out else "")
    return out
