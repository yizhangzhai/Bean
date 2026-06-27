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


def ring_score(xi, xj, gap_mask, min_support=30):
    """Topology-lite H1 detector: do gap positives form a ring around a void?

    Returns (score in [0,1], center, radius). High when residual positives sit
    in a tight shell whose interior is empty of THEM but populated by other
    points -- the hallmark of a 1-D hole that a radial feature collapses.
    """
    res = np.c_[xi[gap_mask], xj[gap_mask]]
    if len(res) < min_support:
        return 0.0, None, 0.0
    center = np.median(res, axis=0)
    r_all = np.hypot(xi - center[0], xj - center[1])
    r_res = r_all[gap_mask]
    R = float(np.median(r_res))
    if R <= 1e-6:
        return 0.0, None, 0.0
    inside_frac = float((r_res < 0.5 * R).mean())        # ring -> few inside
    tight = max(0.0, 1.0 - float(np.std(r_res)) / R)     # tight shell
    others_inside = int((r_all[~gap_mask] < 0.5 * R).sum())
    occupied = 1.0 if others_inside >= min_support else 0.3  # void, not empty space
    return (1.0 - inside_frac) * tight * occupied, center, R


def propose_features(X, gap_mask, names, *, ring_thresh=0.5, max_features=5,
                     eps=1e-6):
    """Rank candidate engineered features that separate the gap positives.

    Topology-guided radial features first (where a ring is detected), then
    generic ratio / diff / product transforms -- all scored by best_band lift
    on the residual.  Returns list of dicts.
    """
    F = X.shape[1]
    lbl = gap_mask.astype(float)
    cands = []

    # 1) topology-guided radial features
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

    # 2) generic algebraic transforms
    for i in range(F):
        for j in range(F):
            if i == j:
                continue
            for kind, tf in (
                    ("ratio", lambda A, i=i, j=j: A[:, i] / (np.abs(A[:, j]) + eps)),
                    ("diff", lambda A, i=i, j=j: A[:, i] - A[:, j])):
                b = best_band(tf(X), lbl)
                if b:
                    cands.append(dict(
                        name=f"{names[i]} {('/' if kind=='ratio' else '-')} {names[j]}",
                        transform=tf, kind=kind, lift=b[0], band=b, topo=0.0))

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
