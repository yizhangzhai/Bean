"""Decision-tree baseline for sanity comparison.

Extracts high-fraud-rate leaf rules from a sklearn tree so we can compare the
mined conjunctive paths against ordinary recursive partitioning on the same
data (lift, support, recovered ground-truth features).
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree


def tree_leaf_rules(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 30,
    seed: int = 0,
) -> list[dict]:
    """Fit a tree on a binary label; return leaf rules sorted by fraud rate."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed
    )
    clf.fit(X_train, y_train)
    t = clf.tree_
    base = y_train.mean()
    rules: list[dict] = []

    def recurse(node: int, conds: list[str]):
        if t.feature[node] == _tree.TREE_UNDEFINED:  # leaf
            val = t.value[node][0]            # class proportions in sklearn>=1.x
            n = int(t.n_node_samples[node])   # actual sample count
            rate = float(val[1] / val.sum()) if val.sum() > 0 else 0.0
            rules.append({
                "conds": list(conds),
                "support": n,
                "rate": rate,
                "lift": float(rate / base) if base > 0 else 0.0,
            })
            return
        f = feature_names[t.feature[node]]
        thr = t.threshold[node]
        recurse(t.children_left[node], conds + [f"{f} <= {thr:.2f}"])
        recurse(t.children_right[node], conds + [f"{f} > {thr:.2f}"])

    recurse(0, [])
    return sorted(rules, key=lambda r: r["lift"], reverse=True)
