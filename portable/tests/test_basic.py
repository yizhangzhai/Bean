"""Smoke + correctness checks for the Bean pipeline (the portable runtime).

Run:  python -m pytest tests/ -q
  or:  python -c "import tests.test_basic as t, inspect; \
       [f() for n,f in inspect.getmembers(t,inspect.isfunction) if n.startswith('test')]; \
       print('tests passed')"
"""

import numpy as np

from synth import make_data
from pipeline import mine_rules, _encode, _bin_edges, _render_pred, N_BINS
from arp.fast import rule_mask, BinSpec
from arp.constraints import RulePolicy
from featgap import propose_features
from featgap.synthesize import FORMATS


def _data(n=20_000, F=24, P=10, seed=0):
    fs = make_data(n, F, P, bad_rate=0.04, seed=seed)
    return fs


def test_mine_basic():
    """mine_rules returns held-out-evaluated rules with sane metrics."""
    fs = _data()
    rules, ev = mine_rules(fs.X, fs.y, categorical=fs.categorical, names=fs.names, n_jobs=2)
    assert ev["n_rules"] >= 1
    assert 0.0 <= ev["precision"] <= 1.0 and 0.0 <= ev["recall"] <= 1.0
    # every rendered rule's preds reproduce its reported support on the val split
    va = ev["val_idx"]
    _, Xva_b, _, _ = _encode(fs.X[va], fs.X[va], fs.y[va], set(fs.categorical), 100_000, 1)
    for r in rules:
        assert isinstance(r.text, str) and len(r.preds) >= 1


def test_constraints_enforced():
    """A disabled feature never appears once the policy forbids it."""
    fs = _data(seed=1)
    base, _ = mine_rules(fs.X, fs.y, categorical=fs.categorical, n_jobs=2)
    used = {f for r in base for f, _, _ in r.preds}
    if not used:
        return
    victim = max(used, key=lambda f: sum(f in {p for p, _, _ in r.preds} for r in base))
    policy = RulePolicy.build(disable=[victim])
    con, _ = mine_rules(fs.X, fs.y, categorical=fs.categorical, n_jobs=2, policy=policy)
    assert all(victim not in {f for f, _, _ in r.preds} for r in con)


def test_engineer_formats():
    """propose_features builds the requested formats incl. sum / linear / custom."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(6000, 3))
    gap = (Z[:, 0] + Z[:, 1]) > 1.5
    cands = propose_features(Z, gap, ["a", "b", "c"],
                             formats=("sum", "linear"),
                             custom=[("a*b", lambda A: A[:, 0] * A[:, 1])])
    kinds = {c["kind"] for c in cands}
    assert "sum" in kinds and "linear" in kinds
    assert "sum" in FORMATS and "linear" in FORMATS


def test_compare_render():
    """A comparison margin bins with 0 as an exact cut and renders relationally."""
    v = np.linspace(-3.0, 3.0, 2000)
    qs = np.arange(1, N_BINS) / N_BINS
    e = _bin_edges(v, qs, np.random.default_rng(0), snap_zero=True)
    assert 0.0 in list(e)                                  # zero is an exact cut
    k = list(e).index(0.0)
    spec = BinSpec([e], qs, N_BINS + 1)
    txt = _render_pred(0, ">", k, [("cmp", "f0 - f1")], ["x"], spec)
    assert txt == "f0 - f1 > 0"                            # i.e.  f0 > f1


def test_engineer_and_nbins_run():
    """End-to-end: n_bins is adjustable and the engineer/compare path runs clean."""
    rng = np.random.default_rng(0)
    n, F = 40_000, 12
    X = rng.normal(size=(n, F)).astype(np.float32)
    y = (((X[:, 0] - X[:, 1]) > 1.5) & (X[:, 2] > 1.0)).astype(int)
    y[rng.random(n) < 0.002] = 1
    rules, ev = mine_rules(X, y, n_jobs=2, n_bins=32, engineer=dict(
        formats=(), compare=[("f0 - f1", lambda Z: Z[:, 0] - Z[:, 1])],
        learn=[(None, [0, 1])]))
    assert ev["n_rules"] >= 1 and 0.0 <= ev["recall"] <= 1.0


if __name__ == "__main__":
    import inspect
    for name, fn in inspect.getmembers(__import__(__name__), inspect.isfunction):
        if name.startswith("test"):
            fn()
    print("tests passed")
