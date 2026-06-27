# attention-rule-paths (`arp`)

Mining conjunctive **decision paths** (e.g. `A > p90 AND B < p10`) with a
**parameter-free, label-as-query "attention"** heuristic — a training-free
cousin of a decision tree / association-rule miner, built for **multi-type,
dollar-weighted fraud** detection.

This is a prototype of the idea developed in conversation. The notes below map
each design choice back to the reasoning so it stays honest about what is novel
(the iterative rule growth + portfolio balancing) and what is not (the
"attention", which is really a co-occurrence/alignment kernel).

## The core idea

1. **Threshold-bin encoding.** Every feature becomes a set of one-sided
   percentile predicates ("decision stumps"): `f03 < p10`, `f00 > p90`, … Each
   instance is implicitly a binary vector over these bins (we never materialize
   the full matrix `M` — see *Scalability*).

2. **Label-as-query attention = one matmul.** A rule's strength is

   ```
   caught = mask.astype(float) @ Yw          # (C,) per fraud type
   lift   = (caught / support) / base_rate
   ```

   The **query** is the label/loss column; the **key** is bin membership.
   Dropping the learned `W_Q/W_K/W_V` is deliberate: with identity projections
   `QK^T` *is* this alignment, and it's the quantity we want to read off as a
   rule. There is no loss to train the projections against. (`scoring.py`)

3. **Beam search over conjunctions.** Grow rules greedily, keeping the top-K
   partial rules — like a tree picking the best split at a node, but
   beam-limited. Conjunction membership is a boolean `AND` of masks (cheap,
   exact). Each depth step scores **all** candidate predicates against a base
   rule in one matmul. (`search.py`)

4. **Coarse-to-fine.** Mine with coarse decile predicates, then `refine_rule`
   sharpens each winning threshold within its local band at fine resolution and
   re-validates — so full resolution is paid only on the survivors, and a sharp
   fraud band a decile average would dilute still gets found.

5. **Multi-type = multi-query.** One label vector `y` → label matrix `Y`
   (N × C). `Y.T @ M` gives a per-type score for every bin in one shot — this
   **is** multi-query attention, each fraud type a query, still parameter-free.

6. **Balance at the portfolio level, not inside each rule.** Fraud types have
   distinct signatures, so we don't force one rule to be a generalist. We mine
   strong per-type rules, then greedily assemble a set that maximizes the
   **worst-covered type** (`portfolio.py`).

## Why not learned Q/K/V?

No training objective ⇒ nothing to fit the projections with; identity
projections recover the exact alignment we want, and keep the rules
interpretable. Learned projections only pay off for **cross-type knowledge
sharing** (a two-tower / multi-task embedding) when rare types are data-starved
— at the cost of interpretability. `V` has essentially no role here: we
*select* rules, we don't aggregate representations.

## Dollar-weighting (fraud loss)

No weight *matrix* is needed — fraud loss is a property of the instance, so it's
a **vector/column** `w`. Swap `Y` (counts) for the dollar-loss matrix `W` and
the same matmul carries dollars. To net out false-positive cost, put negative
entries for legit rows; `w @ M` becomes net value. Run `--dollars` to see a few
expensive cases reshuffle the ranking vs. counts.

## Scalability (what scales, what doesn't)

| Operation | Cost | Scales? |
|---|---|---|
| `Yw.T @ M` (label-as-query scoring) | `O(N·P)` | ✅ linear |
| beam expansion (`AND` + matmul) | `O(beam·P·N)` per depth | ✅ |
| **full bin–bin `MᵀM` attention** | `O(N·P²)` | ❌ never build it |
| dense `M` | `N × P`, ~50% dense | ❌ don't materialize |

The binding constraint is **interaction depth × feature count**, controlled by
beam width + coarse-to-fine + feature pre-filtering — *not* raw `N`. The same
sorted-cumsum / histogram trick that lets LightGBM scale applies here; this
prototype uses on-demand boolean masks (swap in bitset+popcount for production).

## Honest cautions (built into the demo)

- **Overfitting at depth.** Train lift grows as rules deepen; the demo prints
  **val lift** beside it so dilution/overfit is visible. A relative `min_gain`
  makes a 3rd predicate earn its place.
- **Min-support floor** everywhere, including after refinement.
- **Heavy-tailed dollars** let a few cases dominate — keep counts *and* dollars
  in view; winsorize if a handful hijack the search.
- **Multiple testing ×C.** Screening millions of rules against C labels finds
  chance lift; validate **per type** on held-out data (the demo does).

## Layout

```
arp/
  data.py        synthetic multi-type fraud (hidden signatures + heavy-tailed $)
                 + memory-safe fused generate->bin for large N
  encoding.py    percentile-threshold predicates ("decision stumps")
  scoring.py     label-as-query scoring + objectives (single/weighted/maximin/fairness)
  search.py      reference beam search + coarse-to-fine refine + val eval
  fast.py        SCALABLE miner: int8 histogram bins, no dense M, subset rescan
  bitset.py      coarse-to-fine feature pruning + bitmask (AND/popcount) conjunctions
  targeted.py    precision/recall-targeted growth (admissible recall-floor prune,
                 precision-target stop, train/val-gap overfit brake)
  portfolio.py   greedy maximin type-balanced rule portfolio
  baselines.py   sklearn decision-tree comparison
  demo.py        end-to-end runnable demo
experiments/
  recovery.py    does it recover planted patterns? (3/3) + timing
  scale.py       scaling benchmark to 2M x 1000 (naive path)
  scale_fused.py same, memory-safe fused path (~2.6GB peak)
  bitset_bench.py histogram vs coarse-to-fine+bitset (4.8x-9x on mine)
  targeted.py    precision/recall/val-gap targeted growth demo
tests/           smoke + correctness (label-as-query identity, GT recovery, ...)
```

See [experiments/README.md](experiments/README.md) for full results: 3/3 pattern
recovery up to 2M x 1000, linear scaling, the dense-matrix memory lessons, the
9x bitmask speedup, and targeted precision/recall growth.

## Run

```bash
pip install -r requirements.txt
python -m arp.demo            # count-based lift
python -m arp.demo --dollars  # dollar-loss-weighted lift
python -m pytest tests/ -q    # or: python -c "import tests.test_basic as t; [getattr(t,f)() for f in dir(t) if f.startswith('test_')]"
```

## What the demo shows

On synthetic data with three hidden signatures of differing rarity/cost:

- The miner **recovers all three ground-truth signatures**, including
  collusion's two-sided band `p40 < f02 < p55 AND f03 < p10`.
- The **decision-tree baseline finds only the common type** (f04/f05) and misses
  the two rare/expensive ones — the failure mode the per-type + portfolio design
  is meant to fix.
- The portfolio spreads coverage across all three types (the worst-covered type
  is reported), instead of piling onto the easy one.

## Status / next steps

Prototype. Natural extensions: bitset+popcount membership, sorted-cumsum
single-feature scoring, learned cross-type embeddings for rare types,
significance correction for the multiple-testing screen, and a real fraud table
in place of the synthetic generator.
