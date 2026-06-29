# SKILL: rule mining with **Bean**

> Read this top-to-bottom, then jump to the **Scenario playbook** that matches the
> user's goal. Every snippet is runnable as-is from the repo root. Prefer the one
> entry point `mine_rules`; drop to the lower-level building blocks only for the
> cases noted. Be honest about the limits at the end — don't promise capture the
> rarity floor forbids.

---

## 0. What this library does (mental model)

Given a table `X` (numeric + categorical, NaN allowed) and a **binary label** `y`
(0/1), it returns a small set of **interpretable conjunctive rules** — each an
`AND` of conditions — that flag the positive class, with held-out precision/recall
per rule. It can recover **deep** rules (many conditions) that ordinary greedy
rule search misses, by using a gradient-boosted detector to *seed* the search.

Pipeline: **encode once → sequential-covering miner (`recover_deep`) → rules.**

---

## 1. Setup

```bash
pip install -r requirements.txt        # numpy, scikit-learn, lightgbm, joblib
# work from the repo root; `import pipeline` / `import arp` / `import featgap`
```

## 2. The one call: `mine_rules`

```python
from pipeline import mine_rules
rules, ev = mine_rules(X, y, categorical=[12, 13], names=col_names, n_jobs=6)
```

**Inputs**
- `X`: `(n, F)` float array. **NaN is allowed** (handled as its own bin). Categorical
  columns must be **non-negative integer codes** (0,1,2,…), passed by index in
  `categorical`.
- `y`: `(n,)` array of 0/1.
- `categorical`: list of column indices to treat as categorical (default none).
- `names`: optional feature names for readable rules.
- `n_jobs`: worker processes for the parallel per-round searches (set to core count).
- `**deep_kwargs`: any miner knob (see table in §4) passes straight through.

**Outputs**
- `rules`: list of `Rule` objects, sorted by recall. Each has:
  `.text` (readable), `.precision`, `.recall`, `.support`, `.preds`
  (tuple of `(feature_idx, op, bin_k)`; `op` is `">"` or `"<="`).
  `print(rule)` → `[P=0.91 R=0.18 n=2,310]  f032 > p95  AND  f047 in {2,5,7}`
- `ev`: dict with `recall`, `precision`, `flagged`, `positives`, `n_rules`,
  and `val_idx` / `val_cov` (the validation row indices and the union coverage
  mask, for downstream per-segment analysis).

All metrics are on a held-out split (`val_frac=0.33` by default), so they are
honest, not training numbers.

## 3. Preparing real data (CSV → arrays)

```python
import pandas as pd, numpy as np
df = pd.read_csv("data.csv")
y = df["target"].to_numpy()
feat = [c for c in df.columns if c != "target"]
cats = ["mcc", "country"]                      # categorical column names
cat_idx = [feat.index(c) for c in cats]
X = df[feat].apply(lambda s: s.astype("category").cat.codes if s.name in cats else s
                   ).to_numpy(dtype=np.float32)
rules, ev = mine_rules(X, y, categorical=cat_idx, names=feat, n_jobs=6)
```

(Or just `python pipeline.py --data data.csv --label target --categorical mcc,country`.)

## 4. Tuning knobs (pass as kwargs to `mine_rules`)

| knob | default | raise it to… | lower it to… |
|---|---|---|---|
| `min_accept_precision` | 0.12 | **fewer false positives** (stricter rules) | **more coverage** (looser rules) |
| `target_precision` | 0.6 | force deeper/purer rules before accepting | accept shallower rules sooner |
| `min_support` | 20 | avoid tiny, overfit rules | catch rarer patterns |
| `min_recall` | 0.004 | drop low-yield paths faster | let rarer patterns survive |
| `max_depth` | 18 | allow longer conjunctions | cap rule length (simpler rules) |
| `n_seeds` | 8 | find more patterns per round (needs cores) | — |
| `n_jobs` | 1 | **parallel speed** (= core count) | — |
| `block_score` | `"hybrid"` | `"ks"` (numeric, order-aware) / `"kl"` | — |
| `max_rounds`,`max_misses` | 30, 2 | search longer / persist through dry rounds | stop sooner |
| `val_gap_tol` | `None` | — | set to `0.1` to stop conjunctions that overfit (train ≫ val precision) *during* growth |
| `serial` | `False` | `True` = most accurate (one pattern/round, no blending), slowest | — |
| `engineer` | `None` | `True` or a dict = synthesize features for non-axis structure and re-mine (S7) | — |

**`min_accept_precision` is the operating point** — it is the precision floor a
rule must clear on held-out data. It is a *business choice* (how many false
positives you tolerate), not a universal constant.

---

## 5. Scenario playbook

### S1 — "I want high-precision rules (few false alarms)"
```python
rules, ev = mine_rules(X, y, categorical=cat_idx, n_jobs=6,
                       min_accept_precision=0.5, target_precision=0.8, min_support=50)
rules = [r for r in rules if r.precision >= 0.5]      # belt-and-suspenders filter
```

### S2 — "I want maximum coverage / recall"
```python
rules, ev = mine_rules(X, y, categorical=cat_idx, n_jobs=6,
                       min_accept_precision=0.08, n_seeds=12, max_misses=4, max_rounds=60)
# expect lower precision; report ev["recall"] vs ev["precision"] trade-off honestly
```

### S3 — "Categorical features with risky *subsets* (`cat ∈ {…}`)"
Just pass `categorical`. Encoding ranks categories by positive-rate (Fisher trick),
so a single threshold becomes an optimal subset. `block_score="hybrid"` (default)
already does KS on numerics + KL on categoricals.
```python
rules, ev = mine_rules(X, y, categorical=cat_idx, names=feat, n_jobs=6)
# rules render as  "country in {NG,GH,KE}"
```

### S4 — "My data has missing values"
Do nothing special — pass `X` with `NaN`. Each missing value goes to its own bin;
rules can read `feature is MISSING`. No imputation needed.

### S5 — "Rules are too shallow; I need deep / complex logic"
Deep recovery is automatic (that's what `recover_deep` is for). Ensure room:
```python
rules, ev = mine_rules(X, y, categorical=cat_idx, n_jobs=6,
                       max_depth=20, target_precision=0.7)   # higher target forces depth
```
If still shallow, the deep patterns may be **below the rarity floor** (§7) — verify
with the diagnostic in S10 before promising more.

### S6 — "Apply rule-level / business constraints"
Build a `RulePolicy` and pass it to `mine_rules(policy=…)` — it is enforced
**during** the deep search (only compliant rules are accepted), and **min
precision / min recall are separate kwargs**:
```python
from arp.constraints import RulePolicy
from pipeline import mine_rules

policy = RulePolicy.build(
    monotone_up=[chargeback_idx],     # this feature may only split with ">" (1-way up)
    one_way=[score_idx],              # single cut only (no two-sided band)
    disable=[leaky_idx],              # never use this feature
    forbidden_pairs=[(a, b)],         # a and b may not co-occur in a rule
    mutually_exclusive=[(c, d)],      # at most one of c, d
    required_with={e: [f]},           # if e is used, f must be too
    ranges={amt_idx: (0.5, 1.0)},     # only split amt in the top-half percentile range
)
rules, ev = mine_rules(
    X, y, categorical=cat_idx, names=feat, n_jobs=6, policy=policy,
    min_accept_precision=0.30,        # min held-out precision per rule
    min_recall=0.01,                  # min recall per rule
    min_support=50, max_depth=12,     # support / max conditions
)
```
`RulePolicy.build` accepts: `monotone_up/down`, `one_way`, `ranges`, `disable`,
`forbidden_pairs`, `discouraged_pairs`, `mutually_exclusive`, `required_with`,
`cat_eq_only`, `cat_set_only`, `max_set_size`, `allowed_levels`. Every emitted
rule is guaranteed to satisfy the policy.

For a fully constrained **single-pass** miner (categorical-native, no detector
seeding) you can also call `arp.targeted.targeted_beam_search(..., policy=policy)`
or `arp.mixed.mixed_targeted_search(..., policy=policy)` on encoded bins directly.

### S7 — "Suspected non-axis structure (ratio / sum / weighted / ring / periodic)"
No raw threshold conjunction can express these. The built-in way is `engineer=`,
which diagnoses the residual, synthesizes features, appends them, and re-mines —
all in one call:
```python
rules, ev = mine_rules(
    X, y, categorical=cat_idx, names=feat, n_jobs=6,
    engineer=dict(
        cols=[0, 1, 2, 3, 4],                    # candidate columns (default: top-6 by residual separation)
        formats=("ratio", "diff", "sum", "linear", "radial", "periodic"),
        custom=[("f0*f1", lambda A: A[:, 0] * A[:, 1])],   # any user-defined format
        max_features=4))
# engineered rules render with the formula as the feature name, e.g.  "f0 + f1 > p90"
```
Formats (`featgap.synthesize.FORMATS`): `ratio` (A/B), `diff` (A−B), `sum` (A+B),
`linear` (w1·A + w2·B, weights fit by logistic regression on the residual), `radial`
(dist to a detected ring center), `periodic` (A mod P), + `custom` `(name, fn)`.
To control the loop yourself, call the synthesizer directly:
```python
from featgap import propose_features
# gap_mask = positives not covered by current rules (boolean over train rows)
cands = propose_features(X_train[:, candidate_cols], gap_mask,
                         [feat[c] for c in candidate_cols],
                         formats=("sum", "linear"), max_features=4)
# each cand has cand["transform"] (callable) and cand["lift"]; bin the top ones,
# append as columns, and call mine_rules again on the augmented matrix.
```

### S7c — "Compare one feature expression against another (A>B, A>w·B, …)"
For *relational* conditions (not a numeric band) use `engineer=dict(compare=…)`.
Each entry is `(label, fn)` where `fn(X)` returns the **margin LHS−RHS** on the
full matrix (the relation holds when margin > 0). The margin is binned with 0 as
an exact cut, and the rule renders with the real threshold (`label op value`):
```python
import numpy as np
rules, ev = mine_rules(X, y, categorical=cat_idx, engineer=dict(
    formats=(),                                       # comparisons only
    compare=[
        ("f0 - f1",         lambda X: X[:,0] - X[:,1]),                 # A > B
        ("f0 - 1.5*f1",     lambda X: X[:,0] - 1.5*X[:,1]),            # A > 1.5·B
        ("f0 - (f1+f2+f3)*.3", lambda X: X[:,0] - (X[:,1]+X[:,2]+X[:,3])*.3),  # A > (B+C+D)·w
        ("(f0-f1)-(f2-f3)", lambda X: (X[:,0]-X[:,1]) - (X[:,2]-X[:,3])),     # (A−B) > (C−D)
        ("(f2-f3)*f0/f1",   lambda X: (X[:,2]-X[:,3])*X[:,0]/(np.abs(X[:,1])+1e-6)),  # >w
    ]))
# rules render e.g. "f0 - f1 > 0" (= A>B) or "(f2-f3)*f0/f1 > 0.42" (the w is the cut)
```
Rule of thumb: write the relation as `LHS − RHS` in the lambda. Fix a weight inside
`fn` for a hard `A > 1.5·B`; leave it free (`A/B`, `A−B`) and the threshold search
discovers the best `w`. Works for any number of columns and any algebra.

### S7b — "Validation-checked growth / most-accurate (serial) mode"
Two reliability knobs:
- `val_gap_tol=0.1` — validate **during** conjunction: any path whose train
  precision exceeds its held-out precision by more than the tolerance is stopped
  rather than grown (an overfitting brake at every step, not just at acceptance).
- `serial=True` — the original peel-one-pattern-at-a-time miner (`n_seeds=1,
  n_jobs=1`): no within-round blending, most accurate, slowest.
```python
rules, ev = mine_rules(X, y, categorical=cat_idx, val_gap_tol=0.1, serial=True)
```

### S8 — "Large data / make it fast"
- Set `n_jobs` to your core count (parallel seed searches).
- Encoding is a one-time cost; mining is `O(F·N)` per depth.
- Detector fits on a fixed 80k-row subsample regardless of `n` — so at small `n`
  it's detector-bound: a relative early-stop (`min_round_gain`, already set by
  `mine_rules`) keeps round count down.

### S9 — "Rare positive class (e.g. <2%)"
Works, but patterns with **< ~150 positive cases are below the statistical floor**
and won't be captured by any method. Lower `min_support`/`min_recall` only helps
down to that floor. Set expectations accordingly; don't chase the tail.

### S10 — "Diagnose *why* a pattern / segment is missed"
Lower-level: operate on the binned matrix. `uncovered_positives` takes **masks**
(or rule objects with `.preds`), not raw pred tuples.
```python
import numpy as np
from arp.fast import rule_mask
from pipeline import _encode
from featgap import uncovered_positives, remine_residual, interaction_screen

n = len(y); tr = np.random.default_rng(0).permutation(n)[: int(n * 0.67)]
Xbin, _, spec, _ = _encode(X[tr], X[tr], y[tr], set(cat_idx), 100_000, 7)   # train bins
masks = [rule_mask(r.preds, Xbin) for r in rules]              # rules from mine_rules
gap, covered = uncovered_positives(masks, Xbin, y[tr])
print(remine_residual(Xbin, y[tr], covered, spec)["verdict"])  # axis vs non-axis gap
pairs, _ = interaction_screen(X[tr], gap.astype(int), top_k=5) # joint-but-not-marginal pairs
```

### S11 — "Iterate / evaluate / segment"
`rules`/`ev` are already held-out. For per-segment coverage use `ev["val_idx"]`
(validation rows) and `ev["val_cov"]` (which were flagged):
```python
va, cov = ev["val_idx"], ev["val_cov"]
seg = (X[va, region_idx] > thr)                  # any segment mask on the val rows
print("segment recall:", (cov & seg & (y[va]==1)).sum() / max(1,(seg & (y[va]==1)).sum()))
```

### S12 — "Just give me a quick result / demo"
```bash
python pipeline.py --synthetic --n 200000 --features 120 --patterns 40 --jobs 6
```

---

## 6. Lower-level building blocks (when `mine_rules` isn't enough)

- `arp.encode` — `quantile_edges`, `assign_bins`, `target_rank`; `arp.fast` —
  `BinSpec`, `fit_bins`, `rule_mask`. Easiest: `pipeline._encode(Xtr, Xva, ytr,
  cat_set, sample, seed)` → `(Xtr_bins, Xva_bins, spec, render)` ready for any miner.
- `arp.targeted.targeted_beam_search(Xbin, Y, target, spec, *, policy=…, rank_by=…)`
  — single-pass precision/recall-targeted beam (no detector seeding); the place to
  enforce a `RulePolicy`.
- `arp.mixed.mixed_targeted_search(M, y, meta, *, policy=…)` — categorical-native
  miner with `Eq`/`In` predicates (needs an `arp.mixed.Meta` describing column
  kinds/cardinalities).
- `featgap.recover_deep(...)` — the deep engine `mine_rules` wraps; call directly
  if you need to pass an external train/val split, `covered_tr`, or `Xraw_tr`.
- `synth.make_data(n, n_features, n_patterns, bad_rate, *, nonaxis=, missing=)` →
  `DataSet(X, y, categorical, patterns, names)` for testing any of the above.

## 7. Invariants & gotchas (read before promising results)

1. **Categoricals must be integer-coded** (`cat.codes`), non-negative; pass their
   indices in `categorical`. Strings/floats will be mis-binned.
2. **Metrics are held-out.** A rule's `.precision`/`.recall` are on validation rows.
3. **Precision is an operating point**, set by `min_accept_precision` — it tracks
   the data's contamination level; pick it from the user's false-positive budget.
4. **Rarity floor (~150 positive cases/pattern).** Below it, capture is impossible
   for any method — state this rather than over-tuning.
5. **Constraints are enforced during the deep search** via `mine_rules(policy=…)`
   — every emitted rule satisfies the `RulePolicy` (S6). Min precision / min recall
   are separate kwargs (`min_accept_precision`, `min_recall`).
6. **`n_jobs>1` spawns processes** (the beam is GIL-bound); needs `joblib`.
7. **Determinism**: same `seed` + data ⇒ same rules. Vary `seed` to assess stability.

## 8. Validate your setup

```bash
# fast smoke (should print rules with recall/precision):
python pipeline.py --synthetic --n 30000 --features 40 --patterns 10 --jobs 4 --top 3
# capability benchmarks:
python benchmark.py capture 400000 200 100      # multi-pattern capture
python benchmark.py deep 200000 120             # deep-conjunction recovery
python benchmark.py featgap 300000 120          # non-axis -> feature engineering
# unit tests:
python -m pytest tests/ -q
```

## 9. Quick reference (goal → action)

| user goal | do this |
|---|---|
| fewer false positives | `min_accept_precision↑`, `target_precision↑`, post-filter on `.precision` |
| more coverage | `min_accept_precision↓`, `n_seeds↑`, `max_misses↑`, `max_rounds↑` |
| categorical subsets | pass `categorical=...` (target-rank is automatic) |
| missing values | pass NaN as-is (handled) |
| deeper rules | `max_depth↑`, `target_precision↑` |
| business / rule constraints | `mine_rules(policy=RulePolicy.build(...))` — enforced during search (S6) |
| ratio/sum/weighted/ring/periodic | `mine_rules(engineer=dict(formats=...))` (S7) |
| compare features (A>B, A>w·B, …) | `mine_rules(engineer=dict(compare=[(label, fn)]))` (S7c) |
| stop overfit growth | `mine_rules(val_gap_tol=0.1)` — validate during conjunction (S7b) |
| most accurate (slow) | `mine_rules(serial=True)` (S7b) |
| faster | `n_jobs = cores` |
| why missed? | `featgap.remine_residual` + `interaction_screen` (S10) |
| quick demo | `python pipeline.py --synthetic` |
