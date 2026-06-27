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

## Coarse-to-fine + bitmask (`arp.bitset`, `experiments/bitset_bench.py`)

Two optimizations aimed at the `mine` bottleneck, composed:

- **Coarse-to-fine = feature pruning.** A cheap depth-1 pass (skip thresholds +
  row subsample) ranks features by univariate lift; only the **top-k** (e.g.
  1000→64) enter the deep beam search. In the histogram design, depth-1 cost is
  the `bincount` over N, *not* the threshold count — so the leverage is cutting
  **F**, not shrinking bins.
- **Bitmask conjunctions.** Surviving predicates are packed to bits
  (`np.packbits`); a rule mask = bitwise-`AND`, support/caught = `popcount`
  (`np.bitwise_count`). Removes the subset-gather of the histogram path.
  *Count-mode only* — `popcount` can't sum dollar weights, so dollar-lift stays
  on histograms (mine on counts, rank survivors by dollars).

Same data, same recovery (3/3), `mine` time only:

| N | F | histogram subset-rescan | coarse-to-fine + bitset | speedup |
|---:|---:|---:|---:|---:|
| 1 M | 500 | 105.3s | 22.0s (rank 3 / build 8 / search 11) | **4.8×** |
| 2 M | 1000 | 435.1s | 48.1s (rank 9 / build 16 / search 23) | **9.0×** |

Speedup grows with F (more features pruned away). Rules are identical to the
histogram path. `build_bits` is per-type here and is shareable across types —
more speedup left on the table.

## Targeted precision/recall growth (`arp.targeted`, `experiments/targeted.py`)

Practical mining is governed by operating targets, and each target maps to a
*monotonicity* fact:

- **Recall floor → admissible early-stop.** Recall only falls as a rule grows,
  so a shallow path below `min_recall` has no viable descendant — prune the
  subtree (exact, never drops a good rule). This is subgroup-discovery's
  optimistic-estimate pruning.
- **Precision target → stop-on-satisfied.** Precision isn't monotone, so it
  can't prune; but a path that reaches `target_precision` (with recall floor) is
  accepted and not grown — the shortest sufficient rule, which generalizes best.
- **Train/val gap → overfitting brake.** Each candidate is scored on held-out
  data in parallel; if `train_precision − val_precision > gap_tol`, drop it.

Result (200K × 80, target_precision=0.5, min_recall=0.30, gap_tol=0.08):

| type | outcome | depth | P / R (train) | P / R (val) | pruned by recall floor |
|---|---|---|---|---|---|
| account_takeover | **rejected** (no rule generalizes at decile resolution) | — | — | — | 16,619 (+249 val-gap) |
| collusion | accepted, recovers GT | 3 | 0.61 / 0.92 | 0.59 / 0.92 | 13,081 |
| friendly_fraud | accepted, recovers GT | 2 | 0.58 / 0.90 | 0.58 / 0.91 | 20,687 |

Takeaways: the recall floor prunes **13k–20k overfit deep paths** per type (both
an efficiency win and an overfit guard); the precision target stops growth at
the true signature depth; and the val-gap correctly **refuses** to emit
`account_takeover` — whose true `f1>p95` cut is inexpressible at decile bins, so
every path that hits P=0.5 overfits. The honest fix is finer bins (coarse-to-fine
refinement), not a looser guard.

## Hard patterns: deep / disjunctive / banded / adversarial (`arp.hard_data`, `experiments/hard_recovery.py`)

Six planted patterns of varied difficulty (depth 3–6, conjunctive / disjunctive
/ narrow-band / heavy-tailed, plus a gated-XOR adversary), with correlated decoy
features. Mined with the recall-aware targeted search; recovery scored by
**per-branch clean coverage** (is each DNF disjunct covered by a rule using only
that branch's true features?).

200K × 200, depth≤6, beam=16, ~10s / 0.57 GB:

| pattern | depth | branches recovered | best valP/valR | verdict |
|---|---|---|---|---|
| deep5_chain | 5 | 1/1 clean | 0.79 / 0.48 | ✅ |
| narrow_multiband | 3 bands | 1/1 clean | 0.62 / 0.57 | ✅ |
| deep6 | 6 | 1/1 clean | 0.58 / 0.50 | ✅ |
| heavy_tailed | 4 | 1/1 clean | 0.74 / 0.71 | ✅ |
| disjunctive | 3 (×2) | **1/2** | 0.49 / 0.41 | ⚠️ one branch |
| xor_gated | 3 | **0/2** | 0.44 / 0.28 | ❌ gate found, interaction missed |

**4 full, 1 partial, 1 miss.** Honest limitations exposed:
- **XOR / pure interactions** are missed — neither half of `f14⊕f15` has marginal
  signal, so greedy beam can't seed it (it finds the gate `f26` + noise instead).
  This is the known failure mode of greedy rule/tree methods; pairwise
  interaction seeding would be needed.
- **Disjunctions** recover only the strongest branch; the fix is sequential
  covering (remove covered positives, re-mine) — currently noted, not built.
- **Over-generation**: a strong pattern yields thousands of near-duplicate rules;
  controlled by a minimality filter + `max_accept` cap (heavy_tailed 11k→<2k).
- **No decoy ever entered a clean best rule** — false-discovery control held.

Same battery at **1M × 500** (depth≤6): 4 full + 1 partial (identical verdicts),
`gen+bin` 26s, `mine` 156s, peak **1.53 GB**. xor correctly emits **0** rules
(the gate+noise rules fail the val-gap at scale).

## Very deep rules — depth up to 10 (`experiments/deep10.py`)

Light but deep: 200K × 100, three patterns whose true conjunction length is 8–10
(two chains + one of five two-sided bands = 10 conditions), bins=16,
`max_depth=10`, beam=24. Loose per-condition thresholds keep a depth-10 region
~0.3–1% of rows (rare but learnable); because every planted fraud satisfies every
condition, each TRUE condition carries univariate lift ~1/passrate, so the beam
seeds and climbs while the recall floor keeps it on true conditions.

| pattern | true depth | recovered depth | conditions | valP / valR |
|---|---|---|---|---|
| deep10_chain | 10 | 10 | **10/10** | 0.53 / 0.24 |
| deep8_chain | 8 | 8 | **8/8** | 0.75 / 0.20 |
| deep10_bands | 10 (5 bands) | 10 | **5/5 bands** | 0.72 / 0.25 |

**3/3 recovered in 7.6s / 0.34 GB.** Depth is not the obstacle when each
condition carries marginal signal — the miner reconstructs all 10 conditions
(bands included). Recall is ~0.2–0.25 by construction: reaching precision 0.5 at
depth 10 restricts to a high-purity core, the inherent precision/recall tradeoff
the recall floor governs. The genuine obstacle is *interaction without marginal
signal* (the XOR case), not depth per se.

```bash
PYTHONPATH=. python3 experiments/hard_recovery.py              # 6 hard patterns, 200K
PYTHONPATH=. python3 experiments/hard_scale.py --n 1000000 --features 500
PYTHONPATH=. python3 experiments/deep10.py                     # depth-10 recovery, 200K
```
