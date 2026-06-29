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

### S6 — "Apply business constraints (monotonic, forbidden pairs, allowed-only)"
`mine_rules`/`recover_deep` do **not** take a policy. Two correct options:

**(a) Post-filter** the rules from `mine_rules` (simplest; keeps deep mining).
Use the policy's own predicates — `pred_ok` (disabled feature / direction / range),
`extend_ok` (forbidden & mutually-exclusive pairs), `rule_ok` (required-with):
```python
from arp.constraints import RulePolicy
policy = RulePolicy.build(monotone_up=[chargeback_idx], forbidden_pairs=[(a, b)],
                          disable=[leaky_idx])
NB = 17                                       # the encoder's bin count (16 + missing bin)
def compliant(r):
    fs = {f for f, _, _ in r.preds}
    if not all(policy.pred_ok(f, op, k, NB) for f, op, k in r.preds):       # per-feature
        return False
    if not all(policy.extend_ok(fs - {f}, f, op) for f, op, _ in r.preds):  # pairwise
        return False
    return policy.rule_ok(fs)                                               # required-with
rules = [r for r in rules if compliant(r)]
```

**(b) Constrained single-pass mining** (enforced *during* search, but no deep
detector seeding — shallower):
```python
import numpy as np
from pipeline import _encode
from arp.targeted import targeted_beam_search
n = len(y); perm = np.random.default_rng(0).permutation(n); cut = int(n*0.67)
tr, va = perm[:cut], perm[cut:]
Xtr_b, Xva_b, spec, render = _encode(X[tr], X[va], y[tr], set(cat_idx), 100_000, 7)
rules_t, _ = targeted_beam_search(Xtr_b, y[tr][:,None], 0, spec,
                                  Xbin_val=Xva_b, Y_val=y[va][:,None],
                                  target_precision=0.6, min_support=30, policy=policy)
```
`RulePolicy.build` accepts: `monotone_up/down`, `one_way`, `ranges`, `disable`,
`forbidden_pairs`, `discouraged_pairs`, `mutually_exclusive`, `required_with`,
`cat_eq_only`, `cat_set_only`, `max_set_size`, `allowed_levels`.

### S7 — "Suspected non-axis structure (ratio / ring / periodic)"
No threshold conjunction can express these. Diagnose the residual and synthesize
features, then re-mine (see `benchmark.py featgap` for the full recipe):
```python
from featgap import propose_features
# gap_mask = positives not covered by current rules (boolean over train rows)
cands = propose_features(X_train[:, candidate_cols], gap_mask,
                         [feat[c] for c in candidate_cols], max_features=4)
# each cand has cand["transform"] (callable) and cand["lift"]; bin the top ones,
# append as new columns, and call mine_rules again on the augmented matrix.
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
5. **Constraints aren't enforced in the deep path** — use post-filter (S6a) or the
   constrained single-pass miner (S6b).
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
| business constraints | `RulePolicy` + post-filter (S6a) or `targeted_beam_search(policy=)` (S6b) |
| ratio/ring/periodic | `featgap.propose_features` → append feature → re-mine (S7) |
| faster | `n_jobs = cores` |
| why missed? | `featgap.remine_residual` + `interaction_screen` (S10) |
| quick demo | `python pipeline.py --synthetic` |
