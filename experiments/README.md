# Experiments

Two questions: **(1) does the miner recover known planted patterns?** and
**(2) how does it scale?**

## Setup

`arp.data.make_fraud_data_large` plants three fraud signatures among otherwise
pure-noise features, so we have ground truth:

| type | planted signature | features |
|---|---|---|
| account_takeover | `f1 > p95 AND f0 > p90` | {0, 1} |
| collusion | `p40 < f2 < p55 AND f3 < p10` | {2, 3} |
| friendly_fraud | `f4 > p80 AND f5 > p75` | {4, 5} |

`collusion` is the hard one: a **two-sided band** with a sharp, non-monotonic
signal that a single global threshold (or a coarse decile average) would dilute.

## Recovery — does it find the patterns?

```bash
PYTHONPATH=. python3 experiments/recovery.py
```

Result: **3/3 signatures recovered** at every size tested, including collusion's
band `f3 < p10 AND f2 > p40 AND f2 < p55`. The miner finds the right ~2 features
out of hundreds-to-thousands of noise features.

Caveat seen in output: for the rare types the depth-3 search sometimes appends a
**spurious 3rd noise condition** on tiny support — the overfitting-at-depth
caution. The planted features are still all recovered (`planted ⊆ found`), and
the spurious tail is what the val-lift column and `min_support` floor are there
to expose. Capping depth at 2, or gating growth on validation lift, removes it.

## Scalability — histogram/cumsum, no dense `M`

```bash
PYTHONPATH=. python3 experiments/scale.py --n 2000000 --features 1000
```

Why this works at all: the naive dense mask matrix `M` at 2M × 1000 would be
**36 GB** and cannot be built on an 18 GB machine. `arp.fast` instead pre-bins
features into an int8 `Xbin` (N × F bytes) and scores every threshold of every
feature with `np.bincount` cumsums — **no per-predicate mask is ever
materialized**, and conjunctions are grown by re-histogramming only a rule's
(small) support set. Memory beyond `Xbin` is O(N).

Measured on this 18 GB / 12-core machine (bins=10, depth=3, beam=8):

| N | F | cells | gen | bin | depth1 scan | mine (3×depth3) | peak RAM | recovered |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 100 K | 100 | 1e7 | 0.1s | 0.4s | 0.1s | 1.3s | 0.2 GB | 3/3 |
| 500 K | 200 | 1e8 | 0.8s | 4.9s | 2.4s | 20.2s | 0.6 GB | 3/3 |
| 1 M | 500 | 5e8 | 5.4s | 31.5s | 13.0s | 111.0s | 2.7 GB | 3/3 |
| 2 M | 1000 | 2e9 | 32.8s | **4708s** ⚠️ | 48.9s | 424.9s | 10.1 GB | 3/3 |
| 2 M | 1000 | 2e9 | — (fused) 98.1s — | 50.3s | 431.0s | **2.6 GB** | 3/3 |

Observations:

- **depth-1 scan throughput** holds at ~40 M cells/s across all sizes and is
  the cheap part — a full pass over every threshold of every feature.
- **`mine` dominates** because it runs one full beam search *per fraud type*
  (3×), each re-scanning all features at depth 1 then expanding. Obvious next
  optimization: score all types in a single shared depth-1 pass (the
  `_hist_scores` call already returns lift for all C at once) → ~3× on `mine`.
- **The 4708s binning was memory thrashing, not compute.** At 2M × 1000 the
  naive path (`scale.py`) holds the **8 GB float X** *and* the 2 GB `Xbin`
  simultaneously (~10 GB peak), which on an 18 GB machine tips into macOS
  memory compression / swap — note the "0M cells/s". The **fused
  generate→bin** path (`scale_fused.py`) never materializes the float matrix
  (peak **2.6 GB**) and bins the same data in **98s — a 48× speedup from a
  memory fix alone.** Same recovery (3/3).

This is the whole design thesis demonstrated end to end: **never materialize a
dense matrix you can stream.** Twice —
1. the bin–bin `MᵀM` / dense-`M` formulation would be **36 GB** (O(N·P²)) and is
   never built; scoring is histogram cumsums instead.
2. even the float **X** shouldn't be fully resident — fuse generation and
   binning. The histogram path is the same trick behind LightGBM, and it takes
   2M × 1000 (2 billion cells) from infeasible to ~10 min in 2.6 GB.

### Reproduce

```bash
PYTHONPATH=. python3 experiments/recovery.py                                  # recovery + timing
PYTHONPATH=. python3 experiments/scale.py        --n 2000000 --features 1000  # naive (will swap on <=18GB)
PYTHONPATH=. python3 experiments/scale_fused.py  --n 2000000 --features 1000  # fused, ~2.6GB
```
