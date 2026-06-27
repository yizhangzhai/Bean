"""Interpretability / policy constraints enforced DURING rule discovery.

Constraints are applied at the only two places rules are built -- candidate
generation and beam expansion -- so every emitted rule is compliant by
construction (no post-hoc filtering, no wasted search).

Supported (per the deployment requirements):

  feature usage
    * allowed            : feature may be used at all
    * directions         : allowed ops, e.g. (">",) for a monotone-increasing
                           risk feature (chargebacks: higher => riskier, so only
                           "feature > t" is meaningful, never "< t")
    * two_way            : may a feature appear as a 2-sided band (both > and <)?
                           1-way features set two_way=False
    * pct_range          : restrict thresholds to a percentile window, e.g.
                           (0.05, 0.50) -- "only split this feature in its lower
                           half" / avoid extreme low-support cuts

  feature interactions
    * forbidden_pairs    : two features may NOT co-appear in a rule (hard)
    * mutually_exclusive : at most one feature from each listed group
    * discouraged_pairs  : soft -- allowed but penalized in ranking
    * required_with      : if A is used, B must also appear (checked at emit)

Monotonicity here is a *constraint on rule shape* (direction of the split), not
a smoothing of scores -- it guarantees the discovered rules read the way a
reviewer expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureConstraint:
    allowed: bool = True
    # --- numeric ---
    directions: tuple = (">", "<")
    two_way: bool = True
    pct_range: tuple = (0.0, 1.0)
    # --- categorical ---
    cat_kinds: tuple = ("eq", "in")     # allowed predicate kinds
    max_set_size: int = 10 ** 9         # cap on `in {...}` size (interpretability)
    allowed_levels: frozenset = None    # None = any level usable


@dataclass
class RulePolicy:
    per_feature: dict = field(default_factory=dict)          # f -> FeatureConstraint
    forbidden_pairs: set = field(default_factory=set)        # {frozenset({fi,fj})}
    discouraged_pairs: set = field(default_factory=set)      # soft
    mutually_exclusive: list = field(default_factory=list)   # [set(...), ...]
    required_with: dict = field(default_factory=dict)        # f -> set(features)
    default: FeatureConstraint = field(default_factory=FeatureConstraint)

    def fc(self, f) -> FeatureConstraint:
        return self.per_feature.get(f, self.default)

    # ---- candidate generation: is this single predicate allowed? ----
    def pred_ok(self, f, op, k, n_bins) -> bool:
        c = self.fc(f)
        if not c.allowed or op not in c.directions:
            return False
        pct = (k + 1) / n_bins
        return c.pct_range[0] <= pct <= c.pct_range[1] + 1e-9

    # ---- candidate generation: is this categorical predicate allowed? ----
    def cat_pred_ok(self, f, kind, codes) -> bool:
        c = self.fc(f)
        if not c.allowed or kind not in c.cat_kinds:
            return False
        if kind == "in" and len(codes) > c.max_set_size:
            return False
        if c.allowed_levels is not None and not set(codes) <= c.allowed_levels:
            return False
        return True

    # ---- soft ranking penalty for discouraged co-occurrence within a rule ----
    def rule_penalty(self, feats) -> float:
        fl = list(feats)
        pen = 0.0
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                if frozenset((fl[i], fl[j])) in self.discouraged_pairs:
                    pen += 0.05
        return pen

    # ---- expansion: may we add feature f (op) to a rule using rule_feats? ----
    def extend_ok(self, rule_feats, f, op) -> bool:
        c = self.fc(f)
        if f in rule_feats and not c.two_way:           # 2nd cut on f => band
            return False
        for g in rule_feats:
            if g != f and frozenset((g, f)) in self.forbidden_pairs:
                return False
        for grp in self.mutually_exclusive:
            if f in grp and any(g in grp and g != f for g in rule_feats):
                return False
        return True

    # ---- soft ranking penalty for discouraged co-occurrence ----
    def soft_penalty(self, rule_feats, f) -> float:
        return sum(0.05 for g in rule_feats
                   if frozenset((g, f)) in self.discouraged_pairs)

    # ---- final check: required-with partners present ----
    def rule_ok(self, rule_feats) -> bool:
        for f in rule_feats:
            need = self.required_with.get(f)
            if need and not need <= set(rule_feats):
                return False
        return True

    # ---------- convenience builders ----------
    @staticmethod
    def build(*, monotone_up=(), monotone_down=(), one_way=(),
              ranges=None, disable=(), forbidden_pairs=(),
              discouraged_pairs=(), mutually_exclusive=(), required_with=None,
              cat_eq_only=(), cat_set_only=(), max_set_size=None,
              allowed_levels=None):
        pf: dict = {}

        def upd(f, **kw):
            cur = pf.get(f, FeatureConstraint())
            pf[f] = FeatureConstraint(
                allowed=kw.get("allowed", cur.allowed),
                directions=kw.get("directions", cur.directions),
                two_way=kw.get("two_way", cur.two_way),
                pct_range=kw.get("pct_range", cur.pct_range),
                cat_kinds=kw.get("cat_kinds", cur.cat_kinds),
                max_set_size=kw.get("max_set_size", cur.max_set_size),
                allowed_levels=kw.get("allowed_levels", cur.allowed_levels))

        for f in monotone_up:
            upd(f, directions=(">",), two_way=False)
        for f in monotone_down:
            upd(f, directions=("<",), two_way=False)
        for f in one_way:
            upd(f, two_way=False)
        for f in disable:
            upd(f, allowed=False)
        for f, rng in (ranges or {}).items():
            upd(f, pct_range=rng)
        for f in cat_eq_only:
            upd(f, cat_kinds=("eq",))
        for f in cat_set_only:
            upd(f, cat_kinds=("in",))
        for f, sz in (max_set_size or {}).items():
            upd(f, max_set_size=sz)
        for f, lv in (allowed_levels or {}).items():
            upd(f, allowed_levels=frozenset(lv))
        return RulePolicy(
            per_feature=pf,
            forbidden_pairs={frozenset(p) for p in forbidden_pairs},
            discouraged_pairs={frozenset(p) for p in discouraged_pairs},
            mutually_exclusive=[set(g) for g in mutually_exclusive],
            required_with={f: set(v) for f, v in (required_with or {}).items()},
        )
