"""Smoke + correctness checks for the core mechanics."""

import numpy as np

from arp import (
    base_rates, beam_search, build_portfolio, evaluate_rule,
    make_fraud_data, make_predicates, objective_single, refine_rule,
    stack_masks, train_val_split,
)
from arp.scoring import score_one


def test_label_as_query_equals_dot_product():
    """caught == mask @ Yw, the parameter-free 'attention' identity."""
    data = make_fraud_data(n=2000, seed=3)
    W = data.Y.astype(float)
    mask = data.X[:, 4] > np.quantile(data.X[:, 4], 0.8)
    support, caught, _ = score_one(mask, W, base_rates(W, data.n))
    assert support == int(mask.sum())
    np.testing.assert_allclose(caught, mask.astype(float) @ W)


def test_recovers_ground_truth_feature():
    """Type 2's signature uses f04 & f05; the miner should surface them."""
    data = make_fraud_data(n=8000, seed=1)
    tr, _ = train_val_split(data, seed=2)
    W = data.Y.astype(float)
    preds = make_predicates(data.X[tr], n_bins=10)
    pm = stack_masks(preds, data.X[tr])
    obj = lambda lift: objective_single(lift, 2)
    rules = beam_search(preds, pm, W[tr], base_rates(W[tr], len(tr)), obj,
                        beam_width=8, max_depth=2, min_support=30)
    top_feats = rules[0].features
    assert 4 in top_feats and 5 in top_feats


def test_refine_does_not_break_support_floor():
    data = make_fraud_data(n=6000, seed=5)
    tr, _ = train_val_split(data)
    W = data.Y.astype(float)
    preds = make_predicates(data.X[tr], n_bins=10)
    pm = stack_masks(preds, data.X[tr])
    obj = lambda lift: objective_single(lift, 0)
    rules = beam_search(preds, pm, W[tr], base_rates(W[tr], len(tr)), obj,
                        beam_width=6, max_depth=2, min_support=40)
    r = refine_rule(rules[0], data.X[tr], W[tr], base_rates(W[tr], len(tr)),
                    obj, min_support=40)
    assert r.support >= 40


def test_portfolio_improves_worst_type():
    data = make_fraud_data(n=10000, seed=7)
    tr, va = train_val_split(data)
    W = data.Y.astype(float)
    preds = make_predicates(data.X[tr], n_bins=10)
    pm = stack_masks(preds, data.X[tr])
    pool = []
    for c in range(len(data.type_names)):
        obj = lambda lift, c=c: objective_single(lift, c)
        pool += beam_search(preds, pm, W[tr], base_rates(W[tr], len(tr)), obj,
                            beam_width=6, max_depth=2, min_support=40)[:3]
    port = build_portfolio(pool, data.X[va], W[va], max_rules=6)
    # every type should get some coverage
    assert port.covered_frac.min() > 0.0
