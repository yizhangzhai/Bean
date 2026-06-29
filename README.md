# Bean

**Mine interpretable conjunctive rules from labeled tabular data** — governed by
precision/recall targets, scalable to millions of rows, and able to recover *deep*
rules that a plain greedy beam search is fundamentally blind to.

A mined rule looks like this (held-out precision/recall and support in brackets):

```
[P=0.91 R=0.18 n=2,310]  f032 > p95  AND  f047 in {2,5,7}  AND  f018 > p90
```

The pipeline bins the data once, then mines rules with a **sequential-covering**
miner that uses a gradient-boosted detector to *seed* the search at patterns a
marginal beam can't find — recovering deep conjunctions and many overlapping
patterns while keeping every rule interpretable and independently validated.

The target is any **binary label** (the "positive class" / the events you want to
flag); the rules are conjunctions of conditions on numeric and categorical
features.

---

## Install

```bash
pip install -r requirements.txt      # numpy, scikit-learn, lightgbm, joblib
```

## Quick start

**CLI** — on generated data, or your own CSV:

```bash
python pipeline.py --synthetic --n 200000 --features 120 --patterns 40 --jobs 6
python pipeline.py --data data.csv --label target --categorical cat1,cat2 --jobs 6
```

prints:

```
data: 200,000 rows x 120 features, 4,012 positive (2.01%), 10 categorical
... 16 rules  ->  recall=0.51  precision=0.45  flagged=3,674
  [P=0.83 R=0.078 n=46]  f020 > p88  AND  f016 > p81  AND  f001 < p12
  [P=0.77 R=0.329 n=210]  f019 < p06  AND  f047 in {4}
  ...
```

**Python API** — the one entry point:

```python
from pipeline import mine_rules

rules, ev = mine_rules(X, y, categorical=[12, 13], n_jobs=6)
#   X : (n, F) float array (NaN allowed)      y : (n,) array of 0/1
#   categorical : list of column indices

print(ev["recall"], ev["precision"], ev["n_rules"])
for r in rules[:10]:
    print(r)                  # rendered string + r.preds / r.precision / r.recall / r.support
```

Numeric thresholds render as percentiles (`> p95`), categoricals as value sets
(`in {…}`), missing as `is MISSING`. Override any knob via kwargs, e.g.
`mine_rules(X, y, target_precision=0.7, n_seeds=8, block_score="ks")`.

### Key parameters (`mine_rules` / `recover_deep`)

| param | default | meaning |
|---|---|---|
| `target_precision` | 0.6 | precision a rule aims for (search stops growing once met) |
| `min_accept_precision` | 0.12 | held-out precision a rule must clear to be kept (the operating point) |
| `min_support` | 20 | minimum rows a rule must match |
| `min_recall` | 0.004 | recall floor (prunes paths that can't reach it) |
| `max_depth` | 18 | max conditions per rule |
| `n_seeds` | 8 | patterns sought per detector fit (searched in parallel) |
| `n_jobs` | 1 | worker processes for the per-round seed searches |
| `block_score` | `"hybrid"` | feature-block scorer: `kl` / `ks` / `hybrid` (KS numeric + KL categorical) |
| `max_misses` | 2 | consecutive low-yield rounds before stopping |
| `policy` | `None` | `RulePolicy` enforced during search: feature usage, 1-/2-way splits, forbidden / mutually-exclusive pairs, allowed directions & ranges, required-with |
| `val_gap_tol` | `None` | real-time held-out brake: stop growing a conjunction once train precision exceeds its validation precision by more than this (e.g. `0.1`) |
| `serial` | `False` | force the fully-serial peel-one-pattern-at-a-time miner (`n_seeds=1, n_jobs=1`): most accurate, slowest |
| `engineer` | `None` | run a feature-engineering second pass (`True` or a config dict) — see below |

**Rule-level constraints** are first-class: precision/recall floors are the
`min_accept_precision` / `min_recall` kwargs, and structural constraints go through
`policy=RulePolicy.build(monotone_up=…, disable=…, forbidden_pairs=…, ranges=…, …)`
— see `SKILL.md` §S6.

**Validation-checked growth.** By default rules grow on the training split and are
validated at acceptance (`min_accept_precision` on held-out). Pass `val_gap_tol=0.1`
to validate *during* growth: any conjunction whose train precision outruns its
held-out precision by more than the tolerance is stopped instead of extended — an
overfitting brake checked at every conjunction step.

**Serial mode.** `mine_rules(..., serial=True)` runs the original
peel-one-pattern-at-a-time miner (one seed per round, no parallelism). It is the
most accurate setting (no within-round blending) at the cost of runtime; the
default (`n_seeds=8, n_jobs>1`) trades a little accuracy for several patterns per
detector fit.

**Feature engineering.** `mine_rules(..., engineer=True)` adds a second pass: it
diagnoses the residual (uncovered positives), synthesizes candidate features, bins
and appends them, and re-mines — so patterns no axis-aligned threshold can express
(ratio / sum / weighted / ring / periodic) become simple rules. Configure with a
dict:

```python
rules, ev = mine_rules(
    X, y, categorical=cat,
    engineer=dict(
        cols=[0, 1, 2, 3, 4],                       # candidate columns (default: top-6 by residual separation)
        formats=("ratio", "diff", "sum", "linear", "radial", "periodic"),
        custom=[("f0*f1", lambda A: A[:, 0] * A[:, 1])],   # user-defined formats
        max_features=4))
```

Formats (`featgap.synthesize.FORMATS`): `ratio` (A/B), `diff` (A−B), `sum` (A+B),
`linear` (w1·A + w2·B, weights fit by logistic regression on the residual),
`radial` (dist to a detected ring center), `periodic` (A mod P), plus any `custom`
`(name, fn)` expressions. Engineered rules render with the formula as the feature
name, e.g. `f0 + f1 > p90`. The synthesizer is also callable directly:
`from featgap import propose_features`.

**Comparison / relational features.** To put *one feature expression against
another* — `A > B`, `A > w·B`, `A < (B+C+D)·w`, `(C−D) < (A−B)`, `(C−D)·A/B > w` —
pass `compare`, a list of `(label, fn)` where `fn(X)` returns the **margin**
`LHS − RHS` (the relation holds when the margin is `> 0`), evaluated on the full
matrix by original column index:

```python
rules, ev = mine_rules(
    X, y, engineer=dict(
        formats=(),                                   # comparisons only (skip auto-synthesis)
        compare=[
            ("f0 - f1",          lambda X: X[:, 0] - X[:, 1]),               # A > B
            ("f0 - 1.5*f1",      lambda X: X[:, 0] - 1.5 * X[:, 1]),         # A > 1.5·B
            ("(f1+f2+f3)*.3 - f0", lambda X: (X[:,1]+X[:,2]+X[:,3])*.3 - X[:,0]),  # A < (B+C+D)·w
            ("(f2-f3)-(f0-f1)",  lambda X: (X[:,2]-X[:,3]) - (X[:,0]-X[:,1])),  # (C−D) < (A−B)
            ("(f2-f3)*f0/f1",    lambda X: (X[:,2]-X[:,3]) * X[:,0] / (np.abs(X[:,1])+1e-6)),  # >w
        ]))
```

The margin is binned with **0 as an exact cut**, so the relation is representable
exactly, and the rule renders with the **real threshold**, e.g. `f0 - f1 > 0`
(= `A > B`) or `(f2-f3)*f0/f1 > 0.42` — for a `> w` relation the printed value *is*
the discovered `w`. Any expression of any number of columns works; fix the weight
inside `fn` (`A − 1.5·B`) or leave it free and let the threshold search find it.

---

## How it works

1. **Encode** (`arp.encode`, `pipeline._encode`). Each numeric feature → 16
   equal-frequency **percentile bins** (cut-points estimated from a 100k-row
   sample — a one-time, parallel, ~14× faster step); **missing → its own bin**;
   **categorical → positive-rate rank** (Fisher trick: a threshold on the rank is
   an *optimal subset* split `cat ∈ {…}`).

2. **Mine** (`featgap.recover_deep`) — sequential covering. Each round:
   - fit a **LightGBM detector** on the residual (uncovered positives vs the
     rest) — a *scout*, not the model;
   - **cluster the residual positives by leaf co-membership** into up to `n_seeds`
     seeds; each seed ≈ one pattern (it groups examples on the *relevant* features
     and ignores the noise ones);
   - for each seed, pick its feature **block** by KL/KS concentration vs the
     population (`block_score`), and run an **F1-ranked beam** restricted to that
     block — the K seed searches run **in parallel processes** (`n_jobs`);
   - keep rules whose **held-out** precision clears `min_accept_precision`,
     subtract their coverage, repeat; **early-stop** when rounds stop paying off.

3. **Feature engineering for non-axis structure** (`featgap.propose_features`).
   Patterns no threshold conjunction can express (ring / ratio / periodic) are
   diagnosed on the residual and synthesized into features, after which the rule
   is a simple threshold. (`benchmark.py featgap`.)

**Why the boosted seed is necessary.** A deep AND of individually-weak conditions
is a *discovery wall*: every prefix looks statistically like a random broad rule
until almost all conditions are present, so a greedy beam evicts the path before
it forms. No single-rule score (precision, recall, F1) can see it — the signal is
only **joint**. The detector sees that joint signal and names the feature block;
inside that small block the beam climbs the full conjunction. (`benchmark.py deep`.)

**Serial vs. parallel.** Sequential covering is inherently **serial across rounds**
(you remove the patterns you found so the next ones surface). The parallelism is
**within** each round: `n_seeds` blocks searched concurrently, capturing several
patterns per detector fit. `n_seeds=1, n_jobs=1` recovers the original
fully-serial behavior.

---

## Layout

```
arp/            core miner
  encode.py       float -> int8 bins (sampled edges, column-major, threaded); target_rank
  fast.py         BinSpec, fit_bins, fast_beam_search, rule_mask (histogram beam)
  targeted.py     targeted_beam_search (precision/recall-targeted; rank_by f1/precision)
  mixed.py        categorical-native miner (Eq/In predicates, Fisher-trick splits)
  constraints.py  RulePolicy (allowed features, monotone dirs, forbidden pairs, ...)
  ...             search/scoring/encoding/portfolio  (reference miner used by tests)
featgap/        gap-driven layer on top of arp (one-directional dependency)
  deep.py         recover_deep  -- the sequential-covering seed-parallel engine
  gap.py          residual + axis/non-axis diagnostic
  screen.py       interaction-information / HSIC dependence screens
  synthesize.py   ring/ratio/diff/periodic feature synthesis
synth.py        one configurable synthetic data generator (make_data)
pipeline.py     end-to-end entry point: mine_rules(X, y, ...) + CLI
benchmark.py    one runner: capture | scale | deep | featgap | categorical
tests/          test_basic.py
```

## Benchmarks

```bash
python benchmark.py capture     400000 200 100   # multi-pattern capture quality
python benchmark.py scale       500000 300       # encode + mine throughput
python benchmark.py deep        200000 120       # deep-conjunction recovery
python benchmark.py featgap     300000 120       # non-axis -> feature engineering
python benchmark.py categorical 200000 60        # target-rank subset rules
```

Representative result on the **500K × 200, 100-overlapping-pattern, 2% positive
rate** gauntlet (missing values, categoricals, correlated decoys, depths 2–11):
**16 rules, recall 0.51, precision 0.45, all 12 statistically-mineable patterns
captured (62% of all positives), ~5 min on 6 cores.** Confirmed generic on a
structurally different generator with no re-tuning.

## Honest limits

- **Rarity floor.** Patterns with fewer than ~150 cases are below the statistical
  floor for *any* method — uncaptured by design, not by weakness.
- **Very deep + rare + overlapping** patterns (depth ≥ ~9 *and* few cases) remain
  the hard frontier.
- **Precision is an operating point**, set by `min_accept_precision` — it tracks
  the data's contamination level and is a choice, not a constant.
- **Scale**: encoding (~14×) and mining were validated to 2M × 1000; the recent
  multi-pattern *capture* results were measured at 200K–500K.

## Tests

```bash
python -m pytest tests/ -q
# or, without pytest:
python -c "import tests.test_basic as t, inspect; \
[f() for n,f in inspect.getmembers(t,inspect.isfunction) if n.startswith('test')]; \
print('tests passed')"
```
