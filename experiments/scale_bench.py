"""Scale benchmark: sophisticated fraud patterns at 200K x 500 and 2M x 1000.

Measures BOTH rule quality and the time/space cost of every stage, at the
operating point the user asked for:  min rule precision = 0.20, min recall = 0.05.

The population is fired by SIX distinct modus operandi, three axis-expressible
(recoverable by rule search alone) and three non-axis (need the featgap layer):

  MO1 deep_axis  : f0>p60 & f1>p60 & f2<p40 & f3>p55     depth-4 conjunction
  MO2 band_axis  : p45<f4<p60 & f5>p80                   two-sided band + tail
  MO3 dbl_intvl  : p30<f6<p50 & p45<f7<p65               two two-sided intervals
  MO4 ring       : 4 < dist((f8,f9),(20,20)) < 6         radial (non-axis)
  MO5 ratio      : f10/f11 above its p96                 ratio  (non-axis)
  MO6 periodic   : (f12 mod 24) in [2,3]                 periodic (non-axis)

Only 16 columns carry signal; the other ~F-16 are pure noise -- a needle-in-
haystack at 500 / 1000 features. The generator is memory-safe: it fills the
int8 bin matrix block by block and NEVER materializes the full float X (which
would be 8 GB at 2M x 1000).

Space is reported as the process RSS high-water mark (resource.getrusage), so it
includes everything: the int8 matrix, the kept signal floats, and transient
noise blocks. Time is wall-clock per stage.

Run:
    python -m experiments.scale_bench small      # 200K x 500
    python -m experiments.scale_bench large      # 2M x 1000   (heavier)
    python -m experiments.scale_bench both
"""

from __future__ import annotations

import sys
import time
import platform
import resource

import numpy as np

from arp.fast import BinSpec, rule_mask
from arp.targeted import targeted_beam_search
from featgap import uncovered_positives, propose_features

GEO_C = (20.0, 20.0)
NS = 16                                   # number of structured (signal) columns
MO_NAMES = ["deep_axis", "band_axis", "dbl_intvl", "ring", "ratio", "periodic"]


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #
def peak_gb() -> float:
    """Process RSS high-water mark in GB (getrusage: bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1e9 if platform.system() == "Darwin" else 1e6
    return rss / scale


class Stage:
    """Context manager timing a stage and snapshotting peak RSS afterward."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        self.log[self.name] = {"sec": dt, "peak_gb": peak_gb()}
        print(f"    [{self.name:14s}] {dt:8.2f}s   peak RSS {peak_gb():6.2f} GB",
              flush=True)
        return False


# --------------------------------------------------------------------------- #
# memory-safe sophisticated generator
# --------------------------------------------------------------------------- #
def make_sophisticated(n, n_features, n_bins, seed, block=50):
    """Generate (Xbin int8, spec, y, mo_masks, S_signal_floats, names).

    Never holds the full float X: structured columns are kept as a small float
    matrix S (n x 16) for the feature-engineering layer; every other column is
    generated -> binned -> discarded one block at a time.
    """
    rng = np.random.default_rng(seed)
    qs = np.arange(1, n_bins) / n_bins
    Xbin = np.empty((n, n_features), dtype=np.int8)
    edges: list = [None] * n_features

    def bin_col(j, col):
        e = np.quantile(col, qs)
        edges[j] = e.astype(np.float64)
        Xbin[:, j] = np.searchsorted(e, col, side="right").astype(np.int8)

    # ---- structured signal columns (kept as float for featgap) ----
    S = np.empty((n, NS), dtype=np.float32)
    for j in range(8):
        S[:, j] = rng.standard_normal(n)            # f0..f7 axis features
    S[:, 8] = rng.uniform(0, 40, n)                 # f8  geo lat
    S[:, 9] = rng.uniform(0, 40, n)                 # f9  geo lon
    S[:, 10] = rng.uniform(1, 100, n)               # f10 amount
    S[:, 11] = rng.uniform(10, 100, n)              # f11 income
    S[:, 12] = rng.uniform(0, 1000, n)              # f12 timestamp
    for j in range(13, NS):
        S[:, j] = rng.standard_normal(n)            # f13..f15 structured noise

    def q(col, p):
        return np.quantile(col, p)

    r = np.hypot(S[:, 8] - GEO_C[0], S[:, 9] - GEO_C[1])
    ratio = S[:, 10] / S[:, 11]
    tmod = np.mod(S[:, 12], 24)
    mos = [
        (S[:, 0] > q(S[:, 0], .60)) & (S[:, 1] > q(S[:, 1], .60))
        & (S[:, 2] < q(S[:, 2], .40)) & (S[:, 3] > q(S[:, 3], .55)),
        (S[:, 4] > q(S[:, 4], .45)) & (S[:, 4] < q(S[:, 4], .60)) & (S[:, 5] > q(S[:, 5], .80)),
        (S[:, 6] > q(S[:, 6], .30)) & (S[:, 6] < q(S[:, 6], .50))
        & (S[:, 7] > q(S[:, 7], .45)) & (S[:, 7] < q(S[:, 7], .65)),
        (r > 4) & (r < 6),
        ratio > q(ratio, .96),
        (tmod >= 2) & (tmod <= 3),
    ]
    # disjoint by priority so per-MO coverage is unambiguous
    mo_id = np.full(n, -1, dtype=np.int64)
    for k, m in enumerate(mos):
        mo_id[m & (mo_id < 0)] = k
    fire = 0.80
    y = np.zeros(n, dtype=np.int64)
    mo_masks = []
    for k in range(len(mos)):
        hit = (mo_id == k) & (rng.uniform(size=n) < fire)
        mo_masks.append(hit)
        y |= hit.astype(np.int64)
    # a little label noise (background fraud) so precision isn't trivially clean
    bg = (mo_id < 0) & (rng.uniform(size=n) < 0.0015)
    y |= bg.astype(np.int64)

    # ---- bin the structured columns ----
    for j in range(NS):
        bin_col(j, S[:, j])
    # ---- noise columns: generate -> bin -> discard, block by block ----
    for start in range(NS, n_features, block):
        end = min(start + block, n_features)
        Xf = rng.standard_normal((n, end - start), dtype=np.float32)
        for jj in range(end - start):
            bin_col(start + jj, Xf[:, jj])
        del Xf

    names = [f"f{i:04d}" for i in range(n_features)]
    spec = BinSpec(edges, qs, n_bins)
    return Xbin, spec, y, mo_masks, S, names


# --------------------------------------------------------------------------- #
# quality evaluation
# --------------------------------------------------------------------------- #
def covered_mask(rules, Xbin):
    cov = np.zeros(Xbin.shape[0], dtype=bool)
    for r in rules:
        cov |= rule_mask(r.preds, Xbin)
    return cov


def evaluate(rules, Xbin, y, mo_masks):
    cov = covered_mask(rules, Xbin)
    pos = y == 1
    overall_recall = float((cov & pos).sum() / max(1, pos.sum()))
    # policy precision = fraud share of everything the rule set flags (the union;
    # can dip below the per-rule floor when rules share the same frauds)
    flagged = int(cov.sum())
    policy_prec = float((cov & pos).sum() / max(1, flagged))
    per_mo = [float((cov & m).sum() / max(1, m.sum())) for m in mo_masks]
    # per-rule held-out precision/recall (confirms the >=0.2 floor is honored)
    vp = np.array([r.val_precision for r in rules if not np.isnan(r.val_precision)])
    depths = np.array([len(r.preds) for r in rules])
    return dict(overall_recall=overall_recall, policy_prec=policy_prec,
                flagged=flagged, per_mo=per_mo, cov=cov,
                rule_prec_min=float(vp.min()) if len(vp) else float("nan"),
                rule_prec_med=float(np.median(vp)) if len(vp) else float("nan"),
                depth_med=int(np.median(depths)) if len(depths) else 0,
                depth_max=int(depths.max()) if len(depths) else 0)


# --------------------------------------------------------------------------- #
# one scale end to end
# --------------------------------------------------------------------------- #
def run_scale(n, n_features, *, n_bins=16, seed=0, min_precision=0.20,
              min_recall=0.05, beam_width=16, max_depth=6, featgap_cap=500_000):
    log: dict = {}
    print(f"\n{'='*92}")
    print(f"SCALE  n={n:,}  features={n_features}   "
          f"targets: precision>={min_precision}  recall>={min_recall}")
    print('='*92)

    # ---------- generate + bin ----------
    with Stage(log, "generate+bin"):
        Xbin, spec, y, mo_masks, S, names = make_sophisticated(n, n_features, n_bins, seed)
    min_support = max(40, n // 4000)
    frauds = int(y.sum())
    print(f"    Xbin: {Xbin.shape} int8 = {Xbin.nbytes/1e9:.2f} GB        "
          f"frauds={frauds:,} ({100*y.mean():.1f}%)   min_support={min_support}")
    for k, mn in enumerate(MO_NAMES):
        sz = int(mo_masks[k].sum())
        print(f"      MO{k+1} {mn:10s} {sz:>8,}  ({100*sz/max(1,frauds):4.1f}% of fraud)")

    # ---------- train / val split ----------
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    Y = y.reshape(-1, 1)

    # ---------- mine ----------
    print(f"    mining (beam={beam_width}, max_depth={max_depth}) ...")
    with Stage(log, "mine"):
        rules, trace = targeted_beam_search(
            Xbin[tr], Y[tr], 0, spec,
            min_recall=min_recall, target_precision=min_precision,
            min_support=min_support, beam_width=beam_width, max_depth=max_depth,
            Xbin_val=Xbin[va], Y_val=Y[va], gap_tol=0.25, progress=True)

    # ---------- evaluate on validation ----------
    ev = evaluate(rules, Xbin[va], y[va], [m[va] for m in mo_masks])
    print(f"\n    RULES: {len(rules)}   "
          f"val overall-recall={ev['overall_recall']:.2f}   "
          f"policy(union)-precision={ev['policy_prec']:.2f}   "
          f"flagged={ev['flagged']:,}/{len(va):,}")
    print(f"    per-rule held-out precision: min={ev['rule_prec_min']:.2f} "
          f"median={ev['rule_prec_med']:.2f}  (floor was {min_precision})   "
          f"rule depth: median={ev['depth_med']} max={ev['depth_max']}")
    print("    per-MO recall (val):  " +
          "  ".join(f"{MO_NAMES[k]}={ev['per_mo'][k]:.2f}" for k in range(6)))
    axis = [ev['per_mo'][k] for k in (0, 1, 2)]
    nonaxis = [ev['per_mo'][k] for k in (3, 4, 5)]
    print(f"      axis MOs avg = {np.mean(axis):.2f}   "
          f"non-axis MOs avg = {np.mean(nonaxis):.2f}  (expected low -> featgap)")

    # ---------- featgap: diagnose + synthesize on the residual ----------
    # operate on the structured signal floats only (you engineer interpretable
    # raw features, not 1000 noise columns); subsample for tractability at 2M.
    with Stage(log, "featgap"):
        gap_full, _ = uncovered_positives(rules, Xbin, y)
        idx = np.arange(n)
        if n > featgap_cap:
            sub = rng.choice(n, featgap_cap, replace=False)
            idx = sub
        cands = propose_features(S[idx], gap_full[idx], names[:NS], max_features=4)
    print(f"    featgap residual = {int(gap_full.sum()):,} uncovered frauds; "
          f"top engineered features:")
    for c in cands:
        lo, hi = c["band"][1], c["band"][2]
        print(f"      +{c['name']:38s} [{c['kind']:8s}] band=({lo:.1f},{hi:.1f})  "
              f"lift={c['lift']:.1f}")

    # ---------- augmented re-mine (add the engineered features, mine once) ----------
    add = cands[:3]
    qs = np.arange(1, n_bins) / n_bins
    new_cols = np.empty((n, len(add)), dtype=np.int8)
    new_edges = []
    for c_i, c in enumerate(add):
        v = c["transform"](S)
        e = np.quantile(v, qs)
        new_edges.append(e.astype(np.float64))
        new_cols[:, c_i] = np.searchsorted(e, v, side="right").astype(np.int8)
    Xbin2 = np.concatenate([Xbin, new_cols], axis=1)
    spec2 = BinSpec(spec.edges + new_edges, qs, n_bins)
    with Stage(log, "remine+eng"):
        rules2, _ = targeted_beam_search(
            Xbin2[tr], Y[tr], 0, spec2,
            min_recall=min_recall, target_precision=min_precision,
            min_support=min_support, beam_width=beam_width, max_depth=max_depth,
            Xbin_val=Xbin2[va], Y_val=Y[va], gap_tol=0.25)
    ev2 = evaluate(rules2, Xbin2[va], y[va], [m[va] for m in mo_masks])
    print(f"\n    AFTER +{len(add)} engineered features: {len(rules2)} rules   "
          f"val overall-recall={ev2['overall_recall']:.2f}  "
          f"(was {ev['overall_recall']:.2f})   policy-precision={ev2['policy_prec']:.2f}")
    print("    per-MO recall (val):  " +
          "  ".join(f"{MO_NAMES[k]}={ev2['per_mo'][k]:.2f}" for k in range(6)))

    # ---------- summary row ----------
    total = sum(v["sec"] for v in log.values())
    log["_summary"] = dict(
        n=n, n_features=n_features, frauds=frauds, xbin_gb=Xbin.nbytes / 1e9,
        peak_gb=peak_gb(), total_sec=total,
        n_rules=len(rules), recall=ev["overall_recall"], prec=ev["policy_prec"],
        n_rules2=len(rules2), recall2=ev2["overall_recall"], prec2=ev2["policy_prec"])
    return log


def print_table(rows):
    print(f"\n\n{'='*92}\nSUMMARY  (time per stage in seconds; peak = process RSS high-water)\n{'='*92}")
    hdr = (f"{'scale':>16} | {'gen':>7} {'mine':>7} {'featgap':>8} {'remine':>7} "
           f"{'TOTAL':>7} | {'Xbin':>6} {'peakRSS':>8} | "
           f"{'rules':>6} {'recall':>6} {'prec':>5} | {'+eng recall':>11}")
    print(hdr)
    print("-" * len(hdr))
    for log in rows:
        s = log["_summary"]
        print(f"{s['n']//1000}K x {s['n_features']:>4} | "
              f"{log['generate+bin']['sec']:7.1f} {log['mine']['sec']:7.1f} "
              f"{log['featgap']['sec']:8.1f} {log['remine+eng']['sec']:7.1f} "
              f"{s['total_sec']:7.1f} | {s['xbin_gb']:5.2f}G {s['peak_gb']:7.2f}G | "
              f"{s['n_rules']:6d} {s['recall']:6.2f} {s['prec']:5.2f} | "
              f"{s['recall2']:6.2f} ({s['n_rules2']} rules)")


SCALES = {
    "small": (200_000, 500),
    "large": (2_000_000, 1000),
}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    prec = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    todo = ["small", "large"] if which == "both" else [which]
    rows = [run_scale(*SCALES[k], min_precision=prec) for k in todo]
    print_table(rows)
