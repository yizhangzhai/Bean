"""Recovery on MIXED numeric + categorical data.

Plants 3 patterns combining categorical membership/equality with numeric
thresholds, then checks the mixed miner recovers the right features, the right
category subsets, and the numeric cuts.
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np

from arp.fast import fit_bins
from arp.mixed import Meta, mixed_targeted_search

CARDS = [20, 6, 12, 8]                 # cardinalities of cat0..cat3
PLANT = {                              # (numeric feats, [(cat_feat, code_set)])
    "geo_risk":        ({0}, [(0, {0, 1, 2})]),
    "device_merchant": ({1}, [(1, {4}), (2, {7, 9})]),
    "giftcard_band":   ({2}, [(3, {3})]),
}
NAMES = list(PLANT)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1e9 if sys.platform == "darwin" else 1e6)


def make(n, seed):
    rng = np.random.default_rng(seed)
    n_num = 6
    Xnum = rng.standard_normal((n, n_num)).astype(np.float32)
    # skewed categorical distributions (some levels rare)
    Xcat = np.stack([
        rng.choice(K, size=n, p=np.linspace(2, 1, K) / np.linspace(2, 1, K).sum())
        for K in CARDS], axis=1).astype(np.int32)

    def q(f, p):
        return np.quantile(Xnum[:, f], p)

    member = {
        "geo_risk": np.isin(Xcat[:, 0], [0, 1, 2]) & (Xnum[:, 0] > q(0, .90)),
        "device_merchant": (Xcat[:, 1] == 4) & np.isin(Xcat[:, 2], [7, 9])
                           & (Xnum[:, 1] > q(1, .85)),
        "giftcard_band": (Xcat[:, 3] == 3) & (Xnum[:, 2] > q(2, .40))
                        & (Xnum[:, 2] < q(2, .60)),
    }
    Y = np.zeros((n, 3), dtype=np.int64)
    for c, name in enumerate(NAMES):
        p = np.where(member[name], 0.90, 0.0008)
        Y[:, c] = (rng.uniform(size=n) < p).astype(np.int64)
    return Xnum, Xcat, Y


def run(n=200_000, n_bins=12, seed=0):
    t = time.perf_counter()
    Xnum, Xcat, Y = make(n, seed)
    t_gen = time.perf_counter() - t

    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    cut = int(n * 0.67)
    tr, va = perm[:cut], perm[cut:]

    Xb_tr, spec = fit_bins(Xnum[tr], n_bins=n_bins)
    Xb_va, _ = fit_bins(Xnum[va], n_bins=n_bins)
    # unified int matrix: numeric bins then categorical codes
    M_tr = np.concatenate([Xb_tr.astype(np.int32), Xcat[tr]], axis=1)
    M_va = np.concatenate([Xb_va.astype(np.int32), Xcat[va]], axis=1)
    n_num = Xnum.shape[1]
    meta = Meta(
        names=[f"num{i}" for i in range(n_num)] + [f"cat{i}" for i in range(len(CARDS))],
        kind=["num"] * n_num + ["cat"] * len(CARDS),
        size=[n_bins] * n_num + CARDS,
    )

    print(f"\n{'='*104}\nMIXED numeric+categorical recovery  n={n:,}  "
          f"num={n_num} cat={len(CARDS)} (cards={CARDS})\n{'='*104}")
    print(f"gen={t_gen:.1f}s  base rates={np.round(Y.mean(0)*100,3)}%\n")
    print(f"{'pattern':16s} {'feat_ok':>7s} {'cat_jac':>7s} {'valP':>6s} "
          f"{'valR':>6s}  best rule")
    print("-" * 116)

    t = time.perf_counter()
    npass = 0
    for c, name in enumerate(NAMES):
        rules, pruned = mixed_targeted_search(
            M_tr, Y[tr, c].astype(float), meta, min_recall=0.25,
            target_precision=0.5, min_support=40, beam_width=12, max_depth=5,
            M_val=M_va, y_val=Y[va, c].astype(float), gap_tol=0.2)
        num_plant, cat_plant = PLANT[name]
        if not rules:
            print(f"{name:16s} {'MISS':>7s}")
            continue
        best = max(rules, key=lambda r: r.val_precision * r.val_recall)
        # numeric feature recovery
        num_found = {p.f for p in best.preds if type(p).__name__ == "NumPred"}
        feat_ok = num_plant <= num_found
        # categorical recovery: for each planted (feat, set), best overlap
        jacs = []
        for cf, cset in cat_plant:
            got = set()
            for p in best.preds:
                if getattr(p, "f", -1) == cf + 6:    # cat cols offset by n_num
                    got |= ({p.code} if type(p).__name__ == "EqPred" else set(p.codes))
            inter = len(got & cset)
            union = len(got | cset) or 1
            jacs.append(inter / union)
            feat_ok = feat_ok and (inter > 0)
        cat_jac = np.mean(jacs) if jacs else 0.0
        status = feat_ok and cat_jac >= 0.5
        npass += status
        print(f"{name:16s} {'YES' if feat_ok else 'no':>7s} {cat_jac:>7.2f} "
              f"{best.val_precision:>6.2f} {best.val_recall:>6.2f}  "
              f"{'OK' if status else 'PARTIAL'}: {best.label(meta)}")
    t_mine = time.perf_counter() - t
    print("-" * 116)
    print(f"recovered {npass}/3   mine={t_mine:.1f}s   peak RSS={peak_gb():.2f}GB   "
          f"total={t_gen+t_mine:.1f}s")


if __name__ == "__main__":
    run()
