"""featgap.synthesize -- candidate feature construction for the gap.

Given the residual (uncovered positives), diagnose its geometry and synthesize
features that separate it:

  * topology-lite ring/void detector (an H1-like hole == "use a radial coord")
  * candidate transforms (radial / ratio / diff) ranked by how well a single
    band on them isolates the residual positives.

The topology step is a lightweight stand-in for persistent homology (ripser /
gudhi give the rigorous H1); here a center-vs-shell density test detects the
ring signature with no extra dependency.
"""

from __future__ import annotations

import numpy as np


def best_band(v, lbl, n_bins=24, min_support=30, min_recall=0.25):
    """Best two-sided band on feature v isolating lbl==1. Returns
    (lift, lo, hi, precision, recall) in real units, or None."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(v, qs)
    b = np.searchsorted(edges, v)
    B = n_bins
    cnt = np.bincount(b, minlength=B).astype(float)
    pos = np.bincount(b, weights=lbl.astype(float), minlength=B)
    base = lbl.mean()
    cc = np.concatenate([[0.0], np.cumsum(cnt)])
    cp = np.concatenate([[0.0], np.cumsum(pos)])
    tot = lbl.sum()
    best = None
    for a in range(B):
        for c in range(a, B):
            s = cc[c + 1] - cc[a]
            p = cp[c + 1] - cp[a]
            if s < min_support:
                continue
            rec = p / (tot + 1e-9)
            if rec < min_recall:
                continue
            lift = (p / s) / base if base > 0 else 0.0
            if best is None or lift > best[0]:
                lo = edges[a - 1] if a > 0 else -np.inf
                hi = edges[c] if c < B - 1 else np.inf
                best = (lift, float(lo), float(hi), p / s, rec)
    return best


def ring_score(xi, xj, gap_mask, min_support=30, nb=24):
    """Topology-lite H1 detector: do gap positives form a ring around a void?

    Uses an AREA-NORMALIZED radial density peak/void test (robust to background
    contamination, unlike a global std-of-radius tightness): for points uniform
    in 2D, count per radial shell grows like r, so count/r is flat; a ring makes
    count/r spike at its radius while the interior stays empty. Returns
    (score in [0,1], center, peak-radius). High == a genuine ring/void.
    """
    res = np.c_[xi[gap_mask], xj[gap_mask]]
    if len(res) < min_support:
        return 0.0, None, 0.0
    center = np.median(res, axis=0)                       # robust to outliers
    r_res = np.hypot(res[:, 0] - center[0], res[:, 1] - center[1])
    rmax = float(np.percentile(r_res, 99))
    if rmax <= 1e-6:
        return 0.0, center, 0.0
    edges = np.linspace(0, rmax, nb + 1)
    hist, _ = np.histogram(r_res, bins=edges)
    rc = (edges[:-1] + edges[1:]) / 2
    dens = hist / np.maximum(rc, 1e-9)                     # area-normalized
    pk = int(np.argmax(dens))
    R = float(rc[pk])
    if pk < 2 or hist[pk] < min_support:                  # peak at r~0 = a blob
        return 0.0, center, R
    inner = float(dens[: max(1, pk // 2)].mean())         # density inside the peak
    void = dens[pk] / (inner + 1e-9)
    return max(0.0, 1.0 - 1.0 / void), center, R          # ->1 as peak >> interior


def _fit_linear_weights(a, b, lbl):
    """w1, w2 for w1*a + w2*b that best separates lbl (logistic fit, standardized
    so the weights are comparable; returns weights in RAW units). Falls back to
    the mean-difference direction if sklearn is unavailable."""
    sa, sb = a.std() + 1e-9, b.std() + 1e-9
    za, zb = (a - a.mean()) / sa, (b - b.mean()) / sb
    try:
        from sklearn.linear_model import LogisticRegression
        w = LogisticRegression(max_iter=200, class_weight="balanced").fit(
            np.c_[za, zb], lbl).coef_[0]
    except Exception:                                 # mean-difference direction
        m = lbl > 0
        w = np.array([za[m].mean() - za[~m].mean(), zb[m].mean() - zb[~m].mean()])
    n = np.hypot(*w) + 1e-9
    return float(w[0] / n / sa), float(w[1] / n / sb)  # back to raw units, unit norm


# the built-in format families; FORMATS names what propose_features can build
FORMATS = ("radial", "ratio", "diff", "sum", "linear", "periodic")


def propose_features(X, gap_mask, names, *, ring_thresh=0.5, max_features=5,
                     formats=FORMATS, custom=None,
                     periods=(7, 12, 24, 30, 60, 100, 168), eps=1e-6):
    """Rank candidate engineered features that separate the gap positives.

    Families (select via `formats`; default = all), each scored by best_band lift
    on the residual:
      * "radial"   topology-guided dist((A,B); center) where a ring/void is found
      * "ratio"    A / B            (asymmetric -- both orders)
      * "diff"     A - B            (symmetric under a band -- i<j only)
      * "sum"      A + B            (symmetric -- i<j only)
      * "linear"   w1*A + w2*B      (weights fit by logistic regression on the gap)
      * "periodic" A mod P          (x mod P over candidate periods; the winning
                                     period falls out of the lift ranking)

    `custom`: optional list of (name, callable) where callable(X)->1-D array, for
    fully user-defined formats (any expression of the columns). These are scored
    and ranked alongside the built-ins.

    Returns a list of dicts: name, transform (callable X->1-D), kind, lift, band.
    """
    F = X.shape[1]
    lbl = gap_mask.astype(float)
    want = set(formats)
    cands = []

    # 1) topology-guided radial features
    if "radial" in want:
        for i in range(F):
            for j in range(i + 1, F):
                sc, center, R = ring_score(X[:, i], X[:, j], gap_mask)
                if sc >= ring_thresh:
                    tf = (lambda A, i=i, j=j, c=center:
                          np.hypot(A[:, i] - c[0], A[:, j] - c[1]))
                    b = best_band(tf(X), lbl)
                    if b:
                        cands.append(dict(
                            name=f"dist({names[i]},{names[j]}; c=({center[0]:.1f},{center[1]:.1f}))",
                            transform=tf, kind="radial", lift=b[0], band=b, topo=sc))

    # 2) algebraic interactions. ratio asymmetric (both orders); diff/sum
    # symmetric under a two-sided band so i<j only; linear weights are fit.
    for i in range(F):
        for j in range(F):
            if i == j:
                continue
            pairs = []
            if "ratio" in want:
                pairs.append(("ratio", "/",
                              lambda A, i=i, j=j: A[:, i] / (np.abs(A[:, j]) + eps)))
            if i < j and "diff" in want:
                pairs.append(("diff", "-", lambda A, i=i, j=j: A[:, i] - A[:, j]))
            if i < j and "sum" in want:
                pairs.append(("sum", "+", lambda A, i=i, j=j: A[:, i] + A[:, j]))
            if i < j and "linear" in want:
                w1, w2 = _fit_linear_weights(X[:, i], X[:, j], lbl)
                tf = lambda A, i=i, j=j, w1=w1, w2=w2: w1 * A[:, i] + w2 * A[:, j]
                cands.append(dict(name=f"{w1:+.2f}*{names[i]} {w2:+.2f}*{names[j]}",
                                  transform=tf, kind="linear",
                                  lift=(best_band(tf(X), lbl) or (0,))[0],
                                  band=best_band(tf(X), lbl), topo=0.0))
            for kind, sym, tf in pairs:
                b = best_band(tf(X), lbl)
                if b:
                    cands.append(dict(name=f"{names[i]} {sym} {names[j]}",
                                      transform=tf, kind=kind, lift=b[0], band=b, topo=0.0))

    # 3) periodic transforms: x mod P for each candidate period
    if "periodic" in want:
        for i in range(F):
            rng = float(X[:, i].max() - X[:, i].min())
            for P in periods:
                if P >= rng or P <= 0:
                    continue                          # period must fit the range
                tf = lambda A, i=i, P=P: np.mod(A[:, i], P)
                b = best_band(tf(X), lbl)
                if b:
                    cands.append(dict(name=f"{names[i]} mod {P}", transform=tf,
                                      kind="periodic", lift=b[0], band=b, topo=0.0))

    # 4) user-defined formats (any expression of the columns)
    for name, fn in (custom or []):
        b = best_band(np.asarray(fn(X), dtype=float), lbl)
        if b:
            cands.append(dict(name=name, transform=fn, kind="custom",
                              lift=b[0], band=b, topo=0.0))

    cands = [c for c in cands if c.get("band")]
    cands.sort(key=lambda c: c["lift"], reverse=True)
    # dedup by name, keep best
    seen, out = set(), []
    for c in cands:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        out.append(c)
        if len(out) >= max_features:
            break
    return out
