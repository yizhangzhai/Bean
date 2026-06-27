# attention-rule-paths (`arp`)

**Mine interpretable conjunctive rules ("decision paths") from tabular data —
governed by precision/recall targets, scalable to 2M × 1000, and constrained to
meet interpretability/policy requirements.**

A rule looks like:

```
chargebacks > p90  AND  country in {NG, GH, KE}  AND  amount > p95   ⇒  fraud
```

The framework discovers such rules per outcome (e.g. per fraud type), in
dollar-weighted or count terms, with deployment-grade controls on rule shape.

---

## 1. Where this came from

The project began from a single question: *can an attention mechanism quantify
pairwise dependence among entities, the way a decision tree finds decision
paths?* Working through it produced a concrete, honest answer:

- The "attention" here is **label-as-query, parameter-free** — a co-occurrence /
  alignment kernel, **not** learned `W_Q/W_K/W_V`. It is one matrix product, and
  that is exactly what you want to read off as a rule's strength.
- The novel, useful parts are the **iterative rule growth** (beam search over
  conjunctions), the **precision/recall-targeted** branch-and-bound, and the
  **portfolio + constraint** layers — not the attention framing itself.

The README is deliberately honest about what is genuinely new vs. a re-derivation
of decision-tree / association-rule / subgroup-discovery machinery, and about
where the method fundamentally fails (greedy can't seed an interaction with no
marginal signal — the XOR case).

---

## 2. The core idea (how it works)

### 2.1 Threshold-bin encoding → "decision stumps"
Every numeric feature becomes a set of one-sided percentile predicates
(`f > p90`, `f < p10`) plus two-sided **bands** (`p40 < f < p55`). Categorical
features become **equality** (`== c`) and **subset** (`in {…}`) predicates. The
data is represented as an integer matrix of bin indices / category codes — the
dense one-hot membership matrix `M` is **never materialized**.

### 2.2 Label-as-query "attention" = one matmul
A rule's strength comes from a single product:

```
caught = mask @ Yw            # (C,)  per-outcome positives (or $) captured
lift   = (caught / support) / base_rate
```

The **query** is the label/loss column, the **key** is bin membership. With
identity projections, `QKᵀ` *is* this alignment — there is no training objective,
so there are no projection weights to learn. For multiple outcomes,
`Yᵀ @ M` scores every bin for every outcome at once = **multi-query attention**,
each outcome a query.

### 2.3 Beam search over conjunctions = "decision paths"
Rules grow greedily, keeping the top-K partial rules (like a tree picking the
best split at a node, but beam-limited). Conjunction membership is a boolean
`AND`; each depth step scores all candidate predicates in one pass.

### 2.4 Practical growth: precision/recall targets (branch-and-bound)
Real mining is governed by operating targets, each tied to a monotonicity fact:

| Target | Monotonicity | Mechanism |
|---|---|---|
| **recall floor** | recall only *falls* as a rule grows | **admissible early-stop** — prune a subtree once it's below the floor (never discards a viable rule) |
| **precision target** | precision is *not* monotone | **stop-on-satisfied** — accept the shortest rule that reaches it (generalizes best) |
| **train/val gap** | overfitting widens with depth | **overfit brake** — drop a rule whose held-out precision lags train by `gap_tol` |

### 2.5 Balancing & assembly
Different outcomes have different signatures, so the framework mines **per
outcome** and then assembles a **type-balanced portfolio** (greedy maximin —
maximize the worst-covered type) rather than forcing one rule to be a generalist.

---

## 3. Feature set

| Capability | Module | Notes |
|---|---|---|
| Parameter-free label-as-query scoring | `scoring.py` | `mask @ Yw`; count- or dollar-weighted |
| Multi-outcome (multi-type) mining | `scoring`, `fast`, `targeted` | each type is a query |
| **Dollar / cost weighting** | everywhere | swap counts for `$`; negative entries net out false-positive cost |
| Reference beam search + coarse-to-fine refine | `search.py` | clear, didactic implementation |
| **Scalable miner** (histogram / cumsum, no dense `M`) | `fast.py` | int8 bins, subset-rescan; 2M × 1000 feasible |
| **Coarse-to-fine feature pruning + bitmask** | `bitset.py` | `AND`/`popcount`; 4.8×–9× faster `mine` |
| **Precision/recall-targeted growth** | `targeted.py` | recall-floor / precision-target / val-gap |
| **Type-balanced portfolio** | `portfolio.py` | greedy maximin coverage |
| **Numeric + categorical** | `mixed.py` | `==`, `in`-set via the Fisher sort-by-rate trick + smoothing |
| **Interpretability constraints (in-search)** | `constraints.py` | monotonicity, 1-/2-way, ranges, forbidden/required/mutually-exclusive pairs, categorical eq-only / set caps |
| Decision-tree baseline | `baselines.py` | sklearn comparison |
| Synthetic data generators | `data.py`, `hard_data.py` | multi-type, heavy-tailed `$`, deep/disjunctive/banded/XOR patterns, decoys |

---

## 4. Architecture & data flow

```
            ┌─────────────┐   numeric → quantile bins (int8)
 raw table  │  encode      │   categorical → category codes
 ──────────▶│ fit_bins /   │──────────────┐
            │  Meta        │               ▼
            └─────────────┘        ┌──────────────────┐  candidate predicates:
                                   │  candidate gen    │  num: > / < / band
                          ┌───────▶│  (per feature)    │  cat: == / in {…}
                          │        └──────────────────┘  filtered by RulePolicy
                          │                 │
        RulePolicy ───────┘                 ▼
        (constraints)              ┌──────────────────┐  caught = mask @ Yw
                                   │ label-as-query    │  lift / precision / recall
                                   │ scoring           │  (count or $-weighted)
                                   └──────────────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐  recall-floor prune,
                                   │ beam search /     │  precision-target stop,
                                   │ targeted growth   │  val-gap brake, minimality
                                   └──────────────────┘
                                            │  per outcome
                                            ▼
                                   ┌──────────────────┐
                                   │ portfolio assembly│  maximin type balance
                                   └──────────────────┘
                                            │
                                            ▼  compliant, scored rule set
```

**Why it scales:** the dense membership matrix `M` at 2M × 1000 would be **36 GB**
(`O(N·P²)` for the bin–bin variant), impossible on commodity RAM. Instead,
features are pre-binned to int8 and scored with `np.bincount` cumsums — no
per-predicate mask is materialized, and conjunctions are grown by
re-histogramming only a rule's (small) support set. This is the same histogram
trick behind LightGBM. Memory beyond the bin matrix is `O(N)`.

---

## 5. Install & quick start

```bash
pip install -r requirements.txt        # numpy, scikit-learn (baseline only)

python -m arp.demo                      # multi-type fraud, count-based lift
python -m arp.demo --dollars            # dollar-loss-weighted lift
python -m pytest tests/ -q              # correctness tests
```

### 5.1 Minimal end-to-end (scalable numeric miner)

```python
import numpy as np
from arp import (make_binned_fraud_data_large, fit_bins,
                 fast_beam_search, base_rates, objective_single)

# binned synthetic data: Xbin (int8), spec, labels Y (N, C)
Xbin, spec, Y, loss, names, type_names, gt = make_binned_fraud_data_large(
    n=200_000, n_features=100, n_bins=10)

Yw   = Y.astype(float)                  # use `loss` instead for $-weighting
base = base_rates(Yw, len(Y))

target = 0                              # which outcome/fraud type to mine
objective = lambda lift: objective_single(lift, target)
rules = fast_beam_search(Xbin, Yw, base, objective, spec,
                         beam_width=8, max_depth=3, min_support=40)

for r in rules[:3]:
    print(r.label(names, spec), "| lift", round(float(r.lift[target]), 1))
```

### 5.2 Precision/recall-targeted growth (the practical path)

```python
from arp import targeted_beam_search

rules, trace = targeted_beam_search(
    Xbin_train, Y_train, target=0, spec=spec,
    min_recall=0.25, target_precision=0.5, min_support=40,
    beam_width=12, max_depth=6,
    Xbin_val=Xbin_val, Y_val=Y_val, gap_tol=0.20)   # held-out overfit brake

for r in rules:
    print(f"P={r.precision:.2f} R={r.recall:.2f} "
          f"valP={r.val_precision:.2f}  {r.label(names, spec)}")
```

### 5.3 Interpretability constraints

```python
from arp import RulePolicy, targeted_beam_search

policy = RulePolicy.build(
    monotone_up=[chargeback_idx],          # only "chargebacks > t" (risk ↑ with value)
    one_way=[score_idx],                   # no two-sided bands on this feature
    ranges={amount_idx: (0.50, 1.0)},      # only split amount in its upper half
    disable=[leaky_feature_idx],           # never use this feature
    forbidden_pairs=[(feat_a, feat_b)],    # may not co-appear in a rule
    mutually_exclusive=[(f1, f2, f3)],     # at most one of these per rule
    discouraged_pairs=[(g1, g2)],          # soft: avoided on ties
)
rules, _ = targeted_beam_search(..., policy=policy)   # every rule is compliant
```

### 5.4 Numeric + categorical

```python
import numpy as np
from arp import Meta, mixed_targeted_search, fit_bins, RulePolicy

Xb, spec = fit_bins(X_numeric, n_bins=12)
M = np.concatenate([Xb.astype(np.int32), X_categorical_codes], axis=1)
meta = Meta(names=feature_names,
            kind=["num"]*n_num + ["cat"]*n_cat,
            size=[12]*n_num + cardinalities)

policy = RulePolicy.build(
    cat_set_only=[country_idx],            # country only as a subset, never ==
    max_set_size={country_idx: 5},         # at most 5 countries in one rule
)
rules, pruned = mixed_targeted_search(
    M_train, y_train, meta, min_recall=0.25, target_precision=0.5,
    M_val=M_val, y_val=y_val, gap_tol=0.2, policy=policy)
```

### 5.5 Live progress

Pass `progress=True` (or a `Progress` instance) to any miner — `fast_beam_search`,
`targeted_beam_search`, `mixed_targeted_search` — for a per-depth view on stderr:
paths explored, prunes, the growing rule collection, best precision/recall, and
elapsed time. Watch a depth-10 search climb:

```
┌─ targeted[deep10_chain]: F=100  beam=24  max_depth=10  targets=P>=0.5 R>=0.2 gap<=0.25
│  depth 1  beam   1→ 24  explored    3,000  accept +0 (Σ0)     prune recall=595 gap=0     bestP=0.01 R=1.00  0.1s
│  depth 8  beam  24→ 24  explored   63,223  accept +0 (Σ0)     prune recall=57,371 gap=0  bestP=0.38 R=0.27  2.0s
│  depth 9  beam  24→ 24  explored   57,266  accept +147 (Σ147) prune recall=53,507 gap=0  bestP=0.63 R=0.27  2.2s
│  depth 10 beam  24→ 24  explored   53,069  accept +1353 (Σ1500) prune recall=49,169 gap=0 bestP=0.80 R=0.27  3.3s
└─ done  1426 rules (of 1484 pre-minimality)  Σprune recall=490,552 gap=0  10 depths  3.4s
```

The `fast` (lift) miner also emits feature-scan ticks during the depth-1 sweep —
useful on the long 2M × 1000 runs. Try it: `python3 experiments/deep10.py --progress`.

### 5.6 Dollar-weighting & portfolio

```python
from arp import build_portfolio
# mine per type into `all_rules`, then balance coverage across types on val:
port = build_portfolio(all_rules, X_val, W_val, max_rules=8)   # W_val = $ matrix
print(port.covered_frac, "worst:", port.covered_frac.min())
```

---

## 6. Results (validated, reproducible)

All numbers measured on an 18 GB / 12-core machine; see
[experiments/README.md](experiments/README.md) for full tables and commands.

**Pattern recovery** — planted signatures recovered from noise features:

| Setting | Result |
|---|---|
| 3 multi-type signatures, up to **2M × 1000** | **3/3** recovered |
| Hard battery (deep/disjunctive/banded/heavy-tailed/XOR), 200K | **4 full, 1 partial, 1 miss** |
| **Depth-10** chains + bands, 200K × 100 | **3/3** (all 10 conditions), 7.6 s, 0.34 GB |
| Numeric + categorical, exact subsets | **3/3**, `cat_jac = 1.0` |

**Scalability** (histogram path, depth-3, beam-8):

| N × F | cells | bin | depth-1 scan | mine | peak RAM | recovered |
|---:|---:|---:|---:|---:|---:|:--:|
| 100 K × 100 | 1e7 | 0.4 s | 0.1 s | 1.3 s | 0.2 GB | 3/3 |
| 1 M × 500 | 5e8 | 31 s | 13 s | 111 s | 2.7 GB | 3/3 |
| **2 M × 1000** | 2e9 | 98 s* | 49 s | 425 s | 2.6 GB* | 3/3 |

\* fused generate→bin path; the naive path's binning thrashed to swap (10 GB
peak) — a 48× slowdown fixed by never materializing the float matrix.

**Optimizations:** coarse-to-fine + bitmask gives **4.8×** (1M × 500) to **9.0×**
(2M × 1000) speedup on `mine` with identical rules.

---

## 7. Design principles & honest limitations

- **Never materialize a dense matrix you can stream.** Proved twice: the bin–bin
  attention matrix would be 36 GB (never built), and even the float feature
  matrix is fused away during binning.
- **Greedy beam misses interactions with no marginal signal.** The `xor_gated`
  pattern is found only as its gate; the `f14 ⊕ f15` interaction is invisible to
  any greedy seed. This is the fundamental boundary — *interaction without
  marginal signal*, **not depth** (depth-10 conjunctions recover fine). The fix
  is a cross-feature combination pre-miner that seeds such interactions (planned).
- **Disjunctions** currently recover the strongest branch; sequential covering
  (remove covered positives, re-mine) is the fix.
- **Target-encoding leakage** for categoricals (sorting levels by fraud rate) is
  controlled by smoothing + min-support + the train/val gap; do the ordering
  out-of-fold for production.
- **Multiple testing × C**: screening millions of rules finds chance lift —
  always validate **per outcome** on held-out data (every experiment does).
- **Constraints make the interpretability↔performance tradeoff explicit.** E.g.
  forcing a categorical to equality-only cost 32 points of recall on one
  pattern — the framework surfaces that cost rather than hiding it.

---

## 8. Module reference

```
arp/
  data.py        multi-type fraud generator (+ heavy-tailed $; memory-safe
                 fused generate→bin for large N)
  hard_data.py   hard patterns: deep / disjunctive / banded / heavy-tailed /
                 gated-XOR, with correlated decoys
  encoding.py    percentile-threshold predicates ("decision stumps")
  scoring.py     label-as-query scoring + objectives (single/weighted/maximin/fairness)
  search.py      reference beam search + coarse-to-fine refine + val eval
  fast.py        SCALABLE miner: int8 histogram bins, no dense M, subset rescan
  bitset.py      coarse-to-fine feature pruning + bitmask (AND/popcount)
  targeted.py    precision/recall-targeted growth; accepts a RulePolicy
  constraints.py RulePolicy: direction / 1-2-way / range / disable / forbidden /
                 mutually-exclusive / required-with / categorical eq-set caps /
                 soft discouraged pairs
  mixed.py       numeric + categorical predicates; honors RulePolicy
  gapfill.py     gap-driven feature engineering: find uncovered positives,
                 diagnose geometry (topology-lite ring/void), synthesize features
  progress.py    opt-in per-depth progress reporter (paths/prunes/collection/P-R)
  portfolio.py   greedy maximin type-balanced rule portfolio
  baselines.py   sklearn decision-tree comparison
  demo.py        end-to-end runnable demo
experiments/     reproducible studies (see experiments/README.md)
  recovery, scale, scale_fused, bitset_bench, targeted, hard_recovery,
  hard_scale, deep10, intervals, mixed_recovery, constrained, constrained_cat
tests/           correctness (label-as-query identity, GT recovery, portfolio, refine)
```

Run any study with `PYTHONPATH=. python3 experiments/<name>.py`.

---

## 8b. Gap-driven feature engineering (`gapfill.py`)

Axis-aligned rules can't cover a fraud signal that lives on a non-axis structure
(a ring, a ratio ridge). The gap-fill loop closes that:

1. **gap** = positives covered by no rule (the residual).
2. **diagnose geometry** — a topology-lite signal detects whether the residual
   forms a ring / void (an H₁-like hole), which means "use a radial coordinate."
   (A rigorous version would use persistent homology via ripser/gudhi.)
3. **synthesize & rank** features (radial / ratio / diff) by how well a single
   band on them isolates the residual; add the winner and re-mine.

Validated (`experiments/gapfill.py`): one label fired by an axis pattern **or** a
ring `2 < dist((f0,f1),(10,10)) < 4`. The base miner covers only the axis frauds
(recall **0.34**); the gap (71% of frauds, the ring) is diagnosed, the radial
feature is synthesized — recovering the center `(10.1, 9.9)` at **lift 10.3** (4×
the next candidate) — and after re-mining, coverage rises to **0.73**. This is
the documented antidote to "axis-aligned rules can't express the right geometry."

## 9. Roadmap

- **Cross-feature combination pre-miner** (out-of-fold) to seed the interactions
  greedy beam can't reach — the one documented capability gap.
- **Sequential covering** for full disjunction recovery.
- Constraints folded into the **portfolio** layer (joint recall under per-rule policy).
- Production swaps: bitset+popcount membership end-to-end, significance
  correction for the multiple-testing screen, and a real fraud table in place of
  the synthetic generators.

---

## 10. What this is / isn't

**Is:** a training-free, interpretable, scalable conjunctive-rule miner with
operating-target control and policy constraints — a modern synthesis of decision
stumps + beam search + subgroup discovery + histogram scaling, with an
attention-flavored scoring view.

**Isn't:** a learned model. There are no gradients and no fitted weights; the
"attention" is a deterministic alignment kernel. If you need cross-type
statistical strength sharing for very rare types, that's where a *learned*
two-tower embedding would earn its keep — at the cost of the interpretability
this framework is built to preserve.
