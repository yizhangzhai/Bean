"""Deep-logic recovery + fast-encoding benchmark, at 200K x 500 and 2M x 1000.

Two questions:

  A. ENCODING SPEED. How much do the three "free" fixes (sampled edges,
     column-major write, threading) cut the encode time vs the naive
     per-column full-quantile strided path?  -> encode_microbench()

  B. DEEP-LOGIC RECOVERY. Plant fraud whose signatures are DEEP conjunctions --
     rules of depth 5, 8, 11, 15 -- each condition individually weak, the fraud
     only in the full AND. How deep along each planted path can a greedy beam
     climb before the (near-flat) prefix signal disappears? Does more data (2M)
     let it climb deeper than 200K?  -> deep_experiment()

Each depth-d pattern lives on its own disjoint block of d features. Per-condition
kept-fraction is q = target_frac**(1/d), so every pattern -- shallow or deep --
fires on ~the same fraction of rows (and so is equally "minable" by support); the
ONLY thing that varies is how many conditions must stack before the region
concentrates. Thresholds are planted ON bin edges so exact recovery is possible.

Run:
    python -m experiments.deep_bench encode      # just the encode A/B
    python -m experiments.deep_bench small       # 200K x 500 deep recovery
    python -m experiments.deep_bench large        # 2M x 1000 deep recovery
    python -m experiments.deep_bench all
"""

from __future__ import annotations

import sys
import time
import platform
import resource

import numpy as np

from arp.encode import encode_split_cm, encode_matrix_cm, encode_matrix_naive
from arp.targeted import targeted_beam_search

DEPTHS = [5, 8, 11, 15]
TARGET_FRAC = 0.02            # each pattern fires on ~2% of rows, any depth
FIRE = 0.90
N_BINS = 16


def peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if platform.system() == "Darwin" else 1e6)


# --------------------------------------------------------------------------- #
# A. encode microbenchmark (held column-major float matrix)
# --------------------------------------------------------------------------- #
def encode_microbench():
    print(f"\n{'='*92}\nA. ENCODE MICROBENCH  (naive full-quantile/strided  vs  "
          f"sampled/column-major/threaded)\n{'='*92}")
    print(f"{'shape':>14} | {'naive':>8} {'fast':>8} {'speedup':>8} | "
          f"{'cells/s naive':>13} {'cells/s fast':>13} | {'bin agree':>9}")
    print("-" * 92)
    for (N, F) in [(200_000, 500), (2_000_000, 400)]:
        rng = np.random.default_rng(0)
        Xf = rng.standard_normal((F, N), dtype=np.float32)     # column-major floats
        t0 = time.perf_counter()
        bn, _ = encode_matrix_naive(Xf, n_bins=N_BINS)
        t_naive = time.perf_counter() - t0
        t0 = time.perf_counter()
        bf, _ = encode_matrix_cm(Xf, n_bins=N_BINS, sample=100_000)
        t_fast = time.perf_counter() - t0
        agree = float((bn == bf).mean())
        cells = N * F
        print(f"{N//1000}K x {F:>4} | {t_naive:7.2f}s {t_fast:7.2f}s "
              f"{t_naive/t_fast:7.1f}x | {cells/t_naive/1e6:12.0f}M "
              f"{cells/t_fast/1e6:12.0f}M | {agree:8.4f}")
        del Xf, bn, bf
    print("  (bin agree < 1.0 only because sampled edges land a hair off the "
          "full-data quantiles -> a few boundary rows shift one bin)")


# --------------------------------------------------------------------------- #
# B. deep-pattern generator
# --------------------------------------------------------------------------- #
def _plant_condition(col, op, q, n_bins):
    """Plant one condition keeping fraction q; return (mask, (op, k)) with the
    bin split k it maps to (thresholds sit on bin edges -> recoverable)."""
    if op == ">":
        thr = np.quantile(col, 1.0 - q)
        k = int(round((1.0 - q) * n_bins)) - 1
        mask = col > thr
    else:
        thr = np.quantile(col, q)
        k = int(round(q * n_bins)) - 1
        mask = col < thr
    return mask, (op, max(0, min(n_bins - 2, k)))


def make_deep(n, n_features, seed):
    """Signal cols 0..(sum DEPTHS-1) carry the deep patterns; rest are noise.

    Returns (S_signal floats, y, mo_masks, planted, names, tr, va, n_signal).
    `planted[p]` = list of (feature, op, k) ground-truth conditions for pattern p.
    """
    rng = np.random.default_rng(seed)
    n_signal = sum(DEPTHS)
    S = rng.standard_normal((n, n_signal), dtype=np.float32)

    planted = []
    region = []                                  # boolean region (all conds AND)
    col0 = 0
    for d in DEPTHS:
        q = TARGET_FRAC ** (1.0 / d)
        m = np.ones(n, dtype=bool)
        preds = []
        for j in range(d):
            f = col0 + j
            op = ">" if (j % 2 == 0) else "<"
            cm, (op, k) = _plant_condition(S[:, f], op, q, N_BINS)
            m &= cm
            preds.append((f, op, k))
        planted.append(preds)
        region.append(m)
        col0 += d

    # disjoint by priority so each pattern's recall is unambiguous
    mo_id = np.full(n, -1, dtype=np.int64)
    for p, m in enumerate(region):
        mo_id[m & (mo_id < 0)] = p
    y = np.zeros(n, dtype=np.int64)
    mo_masks = []
    for p in range(len(DEPTHS)):
        hit = (mo_id == p) & (rng.uniform(size=n) < FIRE)
        mo_masks.append(hit)
        y |= hit.astype(np.int64)
    bg = (mo_id < 0) & (rng.uniform(size=n) < 0.001)
    y |= bg.astype(np.int64)

    names = [f"f{i:04d}" for i in range(n_features)]
    perm = np.random.default_rng(seed + 1).permutation(n)
    cut = int(n * 0.67)
    return S, y, mo_masks, planted, names, perm[:cut], perm[cut:], n_signal


# --------------------------------------------------------------------------- #
# recovery scoring
# --------------------------------------------------------------------------- #
def score_recovery(rules, planted_preds):
    """Best accepted rule for a planted pattern: how many of its d conditions
    are recovered (same feature+op, bin split within +-1), is it a clean prefix
    (no foreign conditions), and the rule's held-out precision/recall."""
    pset = {(f, op): k for f, op, k in planted_preds}
    pfeats = {f for f, _, _ in planted_preds}
    best = None
    for r in rules:
        rfeats = {f for f, _, _ in r.preds}
        if not (rfeats & pfeats):
            continue
        matched = sum(1 for f, op, k in r.preds
                      if (f, op) in pset and abs(pset[(f, op)] - k) <= 1)
        foreign = len(r.preds) - matched
        cand = (matched, -foreign, r)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return dict(recovered=0, depth=len(planted_preds), rule_len=0,
                    foreign=0, prec=0.0, rec=0.0)
    matched, neg_foreign, r = best
    return dict(recovered=matched, depth=len(planted_preds), rule_len=len(r.preds),
                foreign=-neg_foreign, prec=r.val_precision, rec=r.val_recall)


def prefix_curve(planted_p, Xva, yv):
    """Oracle: held-out precision/recall of the first-k planted conditions, for
    k=1..d. Shows the deep rule is real & high-precision (the full conjunction),
    while every PREFIX sits near the base rate -- the mechanistic reason a greedy
    beam, which ranks partial rules by precision, is blind to the deep path."""
    from arp.fast import rule_mask
    tot = max(1, int(yv.sum()))
    curve = []
    for k in range(1, len(planted_p) + 1):
        m = rule_mask(planted_p[:k], Xva)
        s = int(m.sum())
        tp = int(yv[m].sum())
        curve.append((k, tp / max(1, s), tp / tot))
    return curve


# --------------------------------------------------------------------------- #
# B. deep experiment, one scale
# --------------------------------------------------------------------------- #
def deep_experiment(n, n_features, *, seed=0, target_precision=0.6,
                    min_recall=0.03, beam_width=48, max_depth=16):
    print(f"\n{'='*92}\nB. DEEP-LOGIC RECOVERY  n={n:,}  features={n_features}   "
          f"depths={DEPTHS}\n{'='*92}")
    log = {}
    t0 = time.perf_counter()
    S, y, mo_masks, planted, names, tr, va, n_signal = make_deep(n, n_features, seed)
    log["generate"] = time.perf_counter() - t0
    rng_noise = seed * 100003

    def make_column(f):
        if f < n_signal:
            return S[:, f]
        return np.random.default_rng(rng_noise + f).standard_normal(n).astype(np.float32)

    # ---- fast encode (sampled edges + column-major + threaded), train/val ----
    t0 = time.perf_counter()
    Xtr, Xva, spec = encode_split_cm(make_column, n_features, tr, va,
                                     n_bins=N_BINS, sample=100_000, seed=seed + 7)
    log["encode"] = time.perf_counter() - t0
    frauds = int(y.sum())
    print(f"    frauds={frauds:,} ({100*y.mean():.1f}%)   signal cols={n_signal}   "
          f"min_support tuned per scale")
    for p, d in enumerate(DEPTHS):
        sz = int(mo_masks[p].sum())
        print(f"      pattern depth {d:2d}: {sz:>7,} cases "
              f"({100*sz/max(1,frauds):4.1f}% of fraud)")
    print(f"    [generate {log['generate']:.1f}s]  [encode {log['encode']:.1f}s "
          f"(threaded, sampled edges, column-major)]")

    # ---- mine: high precision target + deep cap forces conjunction growth ----
    min_support = max(40, n // 4000)
    Y = y.reshape(-1, 1)
    print(f"    mining (beam={beam_width}, max_depth={max_depth}, "
          f"P_target={target_precision}, R_floor={min_recall}, "
          f"min_support={min_support}) ...")
    t0 = time.perf_counter()
    rules, _ = targeted_beam_search(
        Xtr, Y[tr], 0, spec, min_recall=min_recall,
        target_precision=target_precision, min_support=min_support,
        beam_width=beam_width, max_depth=max_depth,
        Xbin_val=Xva, Y_val=Y[va], gap_tol=None, progress=True)
    log["mine"] = time.perf_counter() - t0
    print(f"    [mine {log['mine']:.1f}s]   accepted rules: {len(rules)}   "
          f"peak RSS {peak_gb():.2f} GB")

    # ---- COVERAGE / ACCURACY of the whole rule set (do the rules catch the
    #      bads, even where the exact deep conjunction was not recovered?) ----
    from arp.fast import rule_mask
    cov = np.zeros(len(va), dtype=bool)
    for r in rules:
        cov |= rule_mask(r.preds, Xva)
    pos = y[va] == 1
    tp = int((cov & pos).sum())
    recall = tp / max(1, int(pos.sum()))
    precision = tp / max(1, int(cov.sum()))
    accuracy = float((cov == pos).mean())
    pmrec = [float((cov & m[va]).sum() / max(1, int(m[va].sum()))) for m in mo_masks]
    print(f"\n    RULE-SET COVERAGE (val):  recall={recall:.2f}  "
          f"precision={precision:.2f}  accuracy={accuracy:.3f}  "
          f"(flagged {int(cov.sum()):,}/{len(va):,})")
    print("    per-pattern recall:  " +
          "  ".join(f"d{d}={pmrec[p]:.2f}" for p, d in enumerate(DEPTHS)))

    # ---- blind discovery + oracle prefix-precision curve ----
    base = float(y[va].mean())
    print(f"\n    base fraud rate (val) = {base:.3f}")
    print(f"    {'planted':>7} | {'BLIND beam recovery':>22} | "
          f"{'ORACLE prefix precision @ k conditions':>40}")
    print(f"    {'depth d':>7} | {'recovered':>9} {'valP':>5} {'verdict':>5} | "
          f"{'k=1':>6} {'k=d/2':>6} {'k=d-1':>6} {'k=d':>6}  (full rule valR)")
    print("    " + "-" * 86)
    recs = []
    for p, d in enumerate(DEPTHS):
        sr = score_recovery(rules, planted[p])
        recs.append(sr)
        cv = prefix_curve(planted[p], Xva, y[va])
        pk = {k: pr for k, pr, _ in cv}
        full_rec = cv[-1][2]
        mid = (d + 1) // 2
        v = ("FULL" if sr["recovered"] >= d else
             "part" if sr["recovered"] >= 2 else "died")
        print(f"    {d:>7} | {sr['recovered']:>4}/{d:<4} {sr['prec']:>5.2f} {v:>5} | "
              f"{pk[1]:>6.2f} {pk[mid]:>6.2f} {pk[d-1]:>6.2f} {pk[d]:>6.2f}  "
              f"(R={full_rec:.2f})")
    print("    -> the full depth-d rule is real (k=d precision ~0.9) but its "
          "prefixes hug the base rate until the last 1-2 conditions:")
    print("       no marginal gradient for a greedy beam to follow == "
          "deep AND of weak conditions is a discovery (not representation) wall.")

    total = sum(log.values())
    log["_summary"] = dict(n=n, n_features=n_features, frauds=frauds,
                           encode=log["encode"], mine=log["mine"],
                           generate=log["generate"], total=total,
                           peak_gb=peak_gb(), n_rules=len(rules),
                           recovered=[r["recovered"] for r in recs],
                           recall=recall, precision=precision, accuracy=accuracy,
                           per_pattern=pmrec)
    return log


def print_summary(rows):
    print(f"\n\n{'='*92}\nSUMMARY\n{'='*92}")
    hdr = (f"{'scale':>14} | {'gen':>6} {'encode':>7} {'mine':>7} {'TOTAL':>7} | "
           f"{'peakRSS':>8} | {'rules':>6} | recovered depth (of planted)")
    print(hdr)
    print("-" * len(hdr))
    for log in rows:
        s = log["_summary"]
        pm = " ".join(f"d{d}={r:.2f}" for r, d in zip(s["per_pattern"], DEPTHS))
        print(f"{s['n']//1000}K x {s['n_features']:>4} | {s['generate']:6.1f} "
              f"{s['encode']:7.1f} {s['mine']:7.1f} {s['total']:7.1f} | "
              f"{s['peak_gb']:7.2f}G | recall={s['recall']:.2f} prec={s['precision']:.2f} | {pm}")


SCALES = {"small": (200_000, 500), "large": (2_000_000, 1000)}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "encode"
    prec = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    if which == "encode":
        encode_microbench()
    elif which == "all":
        encode_microbench()
        rows = [deep_experiment(*SCALES[k], target_precision=prec)
                for k in ("small", "large")]
        print_summary(rows)
    else:
        rows = [deep_experiment(*SCALES[which], target_precision=prec)]
        print_summary(rows)
