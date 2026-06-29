# attention-rule-paths (`arp`)

**Mine interpretable conjunctive fraud rules from tabular data** — governed by
precision/recall targets, scalable, and able to recover *deep* rules that a plain
greedy beam can't find.

A rule looks like:

```
[P=0.91 R=0.18 n=2,310]  amount > p95  AND  country in {NG,GH,KE}  AND  velocity > p90
```

The pipeline encodes the data once, then mines rules with a sequential-covering
miner that uses a gradient-boosted detector to *seed* the search at patterns a
marginal beam is blind to.

---

## Install

```bash
pip install -r requirements.txt      # numpy, scikit-learn, lightgbm, joblib
```

## Quick start

**CLI** — on your own CSV, or on generated data:

```bash
python pipeline.py --data tx.csv --label is_fraud --categorical mcc,country
python pipeline.py --synthetic --n 200000 --features 120 --patterns 40 --jobs 6
```

**Python API:**

```python
from pipeline import mine_rules
rules, ev = mine_rules(X, y, categorical=[12, 13], n_jobs=6)
print(ev["recall"], ev["precision"])
for r in rules[:10]:
    print(r)            # "[P=.. R=.. n=..]  f037 < p06  AND  f046 in {2,4} ..."
```

`X` is `(n, F)` float (NaN allowed), `y` is `(n,)` in {0,1}, `categorical` lists
the categorical column indices. Output rules carry held-out precision/recall and a
human-readable string (percentile thresholds for numerics, value sets for
categoricals).

---

## How it works

1. **Encode** (`arp.encode`, `pipeline._encode`). Each numeric feature → 16
   equal-frequency **percentile bins** (cut-points estimated from a 100k-row
   sample); **missing → its own bin**; **categorical → fraud-rate rank**
   (Fisher trick: a threshold on the rank = an *optimal subset* `cat ∈ {…}`).

2. **Mine** (`featgap.recover_deep`). Sequential covering. Each round:
   - fit a **LightGBM detector** on the residual (uncovered frauds vs the rest);
   - **cluster the residual frauds by leaf co-membership** into up to `n_seeds`
     seeds — each seed ≈ one pattern (groups frauds on the *relevant* features,
     ignoring noise);
   - for each seed, pick its feature **block** by KL/KS concentration
     (`block_score`), and run an **F1-ranked beam** restricted to that block
     (the K searches run in parallel processes via `n_jobs`);
   - keep rules whose held-out precision clears `min_accept_precision`, subtract
     their coverage, repeat; **early-stop** when rounds stop paying off.

   The boosted detector is a *scout*: it points the search at the right features;
   the rules themselves come from the interpretable beam and are validated
   independently on held-out data.

3. **Feature engineering for non-axis structure** (`featgap.propose_features`).
   Patterns no threshold conjunction can express (ring / ratio / periodic) are
   diagnosed on the residual and turned into engineered features, after which the
   rule becomes a simple threshold. See `benchmark.py featgap`.

**Why the boosted seed is needed.** A deep AND of individually-weak conditions is
a *discovery wall*: every prefix looks like noise until almost all conditions are
present, so a greedy beam evicts the path. The detector sees the *joint* signal
and isolates the feature block; inside that small block the beam climbs the full
conjunction. (`benchmark.py deep`.)

---

## Layout

```
arp/            core miner
  encode.py       float -> int8 bins (sampled edges, column-major, threaded);
                  target_rank categoricals
  fast.py         BinSpec, fit_bins, fast_beam_search, rule_mask (histogram beam)
  targeted.py     targeted_beam_search (precision/recall-targeted, rank_by f1/precision)
  mixed.py        categorical-native miner (Eq/In predicates, Fisher-trick splits)
  constraints.py  RulePolicy (allowed features, monotone dirs, forbidden pairs, ...)
  ... search.py / scoring.py / encoding.py / portfolio.py  (reference miner + tests)
featgap/        gap-driven layer on top of arp (one-directional dependency)
  deep.py         recover_deep  -- the sequential-covering deep miner (the engine)
  gap.py          residual + axis/non-axis diagnostic
  screen.py       interaction-information / HSIC dependence screens
  synthesize.py   ring/ratio/diff/periodic feature synthesis
synth.py        one configurable synthetic fraud generator (make_fraud)
pipeline.py     end-to-end entry point: mine_rules(X, y, ...) + CLI
benchmark.py    one runner: capture | scale | deep | featgap | categorical
tests/          test_basic.py  (run: python -m pytest tests/, or import + call)
```

## Benchmarks

```bash
python benchmark.py capture     400000 200 100   # multi-pattern capture quality
python benchmark.py scale       500000 300       # encode + mine throughput
python benchmark.py deep        200000 120       # deep-conjunction recovery
python benchmark.py featgap     300000 120       # non-axis -> feature engineering
python benchmark.py categorical 200000 60        # target-rank subset rules
```

Representative result on the **500K × 200, 100-overlapping-pattern, 2% fraud**
gauntlet (missing values, categoricals, decoys, depths 2–11): **16 rules,
recall 0.51, precision 0.45, all 12 statistically-mineable patterns captured
(62% of all fraud), ~5 min on 6 cores.** Confirmed generic on a structurally
different generator without re-tuning.

## Honest limits

- **Rarity floor.** Patterns with < ~150 cases are below the statistical floor
  for *any* method — they're uncaptured by design, not by weakness.
- **Very deep + rare + overlapping** patterns (depth ≥ ~9 *and* few cases) remain
  the hard frontier.
- **Precision is an operating point**, set by `min_accept_precision` — it tracks
  the data's contamination level and is a business choice, not a constant.
- **Scale**: encoding (~14×) and mining were validated to 2M × 1000; the recent
  multi-pattern *capture* results were measured at 200K–500K.
