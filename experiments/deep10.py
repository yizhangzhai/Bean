"""Can it capture VERY deep rules (depth up to 10)?  Light: 200K x 100.

Three planted patterns whose true conjunction length is 8-10:
  deep10_chain : 10 one-sided conditions on 10 features
  deep8_chain  : 8 one-sided conditions on 8 features
  deep10_bands : 5 two-sided bands (= 10 conditions) on 5 features

Thresholds are loose (each condition passes ~50-60%) so a depth-10 region is
still ~0.3-1% of rows -- rare but learnable at 200K. Because every planted
fraud satisfies every condition, each TRUE condition carries univariate lift
~1/passrate (well above noise lift ~1), so greedy beam can seed and climb;
recall stays ~1 while a true condition is added (precision climbs), and only a
WRONG condition collapses recall -- which is exactly what the recall floor uses.

Reports recovered depth, branch coverage, precision/recall, time and peak RSS.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np

from arp.fast import fit_bins
from arp.targeted import targeted_beam_search

DEEP = [
    dict(name="deep10_chain", fire=.92, bg=.00002, depth=10, dnf=[[
        (">", 0, .44), ("<", 1, .56), (">", 2, .44), ("<", 3, .56),
        (">", 4, .44), ("<", 5, .56), (">", 6, .44), ("<", 7, .56),
        (">", 8, .44), ("<", 9, .56)]]),
    dict(name="deep8_chain", fire=.92, bg=.00002, depth=8, dnf=[[
        (">", 10, .50), ("<", 11, .50), (">", 12, .50), ("<", 13, .50),
        (">", 14, .50), ("<", 15, .50), (">", 16, .50), ("<", 17, .50)]]),
    dict(name="deep10_bands", fire=.92, bg=.00001, depth=10, dnf=[[
        ("band", 18, .30, .70), ("band", 19, .30, .70), ("band", 20, .30, .70),
        ("band", 21, .30, .70), ("band", 22, .30, .70)]]),
]
SIG = 23


def peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if sys.platform == "darwin" else 1e6)


def _cond_mask(X, cond, q):
    if cond[0] == ">":
        return X[:, cond[1]] > q[cond[1]][cond[2]]
    if cond[0] == "<":
        return X[:, cond[1]] < q[cond[1]][cond[2]]
    f = cond[1]
    return (X[:, f] > q[f][cond[2]]) & (X[:, f] < q[f][cond[3]])


def make(n, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features), dtype=np.float32)
    q = {}
    for pat in DEEP:
        for conj in pat["dnf"]:
            for cond in conj:
                f = cond[1]
                q.setdefault(f, {})
                for p in cond[2:]:
                    q[f].setdefault(p, float(np.quantile(X[:, f], p)))
    Y = np.zeros((n, len(DEEP)), dtype=np.int64)
    for c, pat in enumerate(DEEP):
        member = np.zeros(n, dtype=bool)
        for conj in pat["dnf"]:
            m = np.ones(n, dtype=bool)
            for cond in conj:
                m &= _cond_mask(X, cond, q)
            member |= m
        p = np.where(member, pat["fire"], pat["bg"])
        Y[:, c] = (rng.uniform(size=n) < p).astype(np.int64)
    planted = [frozenset(cond[1] for cond in pat["dnf"][0]) for pat in DEEP]
    return X, Y, planted


def run(n=200_000, n_features=100, bins=16, max_depth=10, beam=24, seed=0,
        progress=False):
    t = time.perf_counter()
    X, Y, planted = make(n, n_features, seed)
    t_gen = time.perf_counter() - t

    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]
    t = time.perf_counter()
    Xtr, spec = fit_bins(X[tr], n_bins=bins)
    Xva, _ = fit_bins(X[va], n_bins=bins)
    t_bin = time.perf_counter() - t
    Ytr, Yva = Y[tr], Y[va]
    SIGSET = set().union(*planted)

    print(f"\n{'='*100}\nDEEP-RULE RECOVERY  n={n:,} F={n_features} bins={bins} "
          f"max_depth={max_depth} beam={beam}\n{'='*100}")
    print(f"gen={t_gen:.1f}s fit_bins(2x)={t_bin:.1f}s  "
          f"base rates={np.round(Y.mean(0)*100,3)}%  "
          f"(positives/type={Y.sum(0)})")
    print(f"\n{'pattern':14s} {'true_d':>6s} {'got_d':>5s} {'#cond_ok':>8s} "
          f"{'valP':>6s} {'valR':>6s} {'#rule':>6s}  status / best rule")
    print("-" * 132)

    t = time.perf_counter()
    summary = []
    for c, pat in enumerate(DEEP):
        from arp.progress import Progress
        prog = Progress(enabled=progress, label=f"deep10[{pat['name']}]")
        rules, trace = targeted_beam_search(
            Xtr, Ytr, c, spec, min_recall=0.20, target_precision=0.5,
            min_support=40, beam_width=beam, max_depth=max_depth,
            Xbin_val=Xva, Y_val=Yva, gap_tol=0.25, max_accept=1500, progress=prog)
        plant = planted[c]
        if not rules:
            summary.append((pat["name"], "MISS"))
            print(f"{pat['name']:14s} {pat['depth']:>6d} {'-':>5s} {'-':>8s} "
                  f"{'-':>6s} {'-':>6s} {'0':>6s}  MISS (no rule met targets)")
            continue
        # best rule by val P*R; clean = uses only this pattern's true features
        best = max(rules, key=lambda r: r.val_precision * r.val_recall)
        bf = {f for f, _, _ in best.preds}
        cond_ok = len(bf & plant)
        clean = not (bf - SIGSET)
        covered = (plant <= bf) and clean
        status = "OK" if covered else ("PARTIAL" if cond_ok >= len(plant) * .6 else "WEAK")
        summary.append((pat["name"], status))
        print(f"{pat['name']:14s} {pat['depth']:>6d} {best.depth:>5d} "
              f"{str(cond_ok)+'/'+str(len(plant)):>8s} {best.val_precision:>6.2f} "
              f"{best.val_recall:>6.2f} {len(rules):>6d}  {status}: "
              f"{best.label([f'f{i:03d}' for i in range(n_features)], spec)}")
    t_mine = time.perf_counter() - t

    print("-" * 132)
    okc = sum(s == "OK" for _, s in summary)
    print(f"recovered {okc}/{len(DEEP)} deep patterns   "
          f"mine={t_mine:.1f}s   peak RSS={peak_gb():.2f}GB   "
          f"total={t_gen+t_bin+t_mine:.1f}s")
    for name, s in summary:
        print(f"    {s:8s} {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--features", type=int, default=100)
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--beam", type=int, default=24)
    ap.add_argument("--progress", action="store_true", help="live per-depth progress")
    a = ap.parse_args()
    run(a.n, a.features, a.bins, a.max_depth, a.beam, progress=a.progress)
