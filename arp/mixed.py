"""Mixed numeric + categorical rule mining.

Numeric features use ordered threshold predicates (as before). Nominal
categoricals use two predicate families:

  ==  equality on one category            (interpretable, K candidates)
  in  subset membership                   (Fisher-optimal: sort categories by
                                           SMOOTHED fraud rate, take prefixes ->
                                           K-1 candidates, reuses the cumsum idea)

Everything lives in one int matrix M (n, F): numeric columns hold bin indices,
categorical columns hold category codes. A per-column `kind` array dispatches
predicate generation and masking. The conjunction/recall-floor/precision-target/
val-gap machinery is identical to arp.targeted -- only candidate generation and
masking differ by kind.

Leakage note: ordering categories by fraud rate is target encoding. We smooth
  rate_c = (pos_c + alpha*base) / (n_c + alpha)
and require min_support per category/subset; the train/val gap (gap_tol) is the
backstop that drops categorical rules whose rate ordering overfit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------- predicates (polymorphic, hashable) ----------------
@dataclass(frozen=True)
class NumPred:
    f: int
    op: str          # '>' or '<'
    k: int
    def mask(self, M):
        c = M[:, self.f]
        return c > self.k if self.op == ">" else c <= self.k
    def fkey(self):  # forbid same (feature, direction)
        return (self.f, self.op)
    def label(self, meta):
        pct = int(round((self.k + 1) / meta.size[self.f] * 100))
        return f"{meta.names[self.f]} {self.op} p{pct:02d}"


@dataclass(frozen=True)
class EqPred:
    f: int
    code: int
    def mask(self, M):
        return M[:, self.f] == self.code
    def fkey(self):  # forbid reusing a categorical feature
        return (self.f, "cat")
    def label(self, meta):
        return f"{meta.names[self.f]} == {meta.cat_name(self.f, self.code)}"


@dataclass(frozen=True)
class InPred:
    f: int
    codes: tuple
    def mask(self, M):
        return np.isin(M[:, self.f], self.codes)
    def fkey(self):
        return (self.f, "cat")
    def label(self, meta):
        vals = ", ".join(meta.cat_name(self.f, c) for c in self.codes)
        return f"{meta.names[self.f]} in {{{vals}}}"


@dataclass
class Meta:
    names: list
    kind: list          # 'num' | 'cat' per column
    size: list          # n_bins (num) or cardinality (cat)
    cat_values: dict = field(default_factory=dict)  # f -> list of display names
    def cat_name(self, f, c):
        vals = self.cat_values.get(f)
        return vals[c] if vals and c < len(vals) else f"c{c}"


# ---------------- candidate generation ----------------
def num_cands(f, col, y, n_bins, min_support):
    B = n_bins
    counts = np.bincount(col, minlength=B).astype(np.float64)
    cc = np.cumsum(counts)
    wc = np.cumsum(np.bincount(col, weights=y, minlength=B))
    tot = wc[-1]
    N = col.shape[0]
    out = []
    for k in range(B - 1):
        for op, s, tp in (("<", cc[k], wc[k]), (">", N - cc[k], tot - wc[k])):
            if s >= min_support:
                out.append((NumPred(f, op, k), float(s), float(tp)))
    return out


def cat_cands(f, col, y, card, min_support, base, alpha=20.0,
              singletons=True, max_sets=16):
    K = card
    cnt = np.bincount(col, minlength=K).astype(np.float64)
    pos = np.bincount(col, weights=y, minlength=K)
    rate = (pos + alpha * base) / (cnt + alpha)        # smoothed (anti-leakage)
    order = np.argsort(rate)[::-1]                      # high fraud-rate first
    out = []
    if singletons:
        for c in range(K):
            if cnt[c] >= min_support:
                out.append((EqPred(f, int(c)), float(cnt[c]), float(pos[c])))
    # Fisher prefixes along the rate ordering = best subset splits for binary y
    cum_n = cum_p = 0.0
    chosen = []
    for i, c in enumerate(order):
        cum_n += cnt[c]
        cum_p += pos[c]
        chosen.append(int(c))
        if cum_n >= min_support and i < K - 1 and len(chosen) <= max_sets:
            out.append((InPred(f, tuple(sorted(chosen))), cum_n, cum_p))
    return out


def feature_cands(f, col, y, meta, min_support, base, policy=None):
    if meta.kind[f] == "num":
        cands = num_cands(f, col, y, meta.size[f], min_support)
        if policy is None:
            return cands
        return [(p, s, tp) for p, s, tp in cands
                if policy.pred_ok(p.f, p.op, p.k, meta.size[f])]
    cands = cat_cands(f, col, y, meta.size[f], min_support, base)
    if policy is None:
        return cands
    out = []
    for p, s, tp in cands:
        kind = "eq" if isinstance(p, EqPred) else "in"
        codes = (p.code,) if kind == "eq" else p.codes
        if policy.cat_pred_ok(p.f, kind, codes):
            out.append((p, s, tp))
    return out


# ---------------- rule + mixed targeted search ----------------
@dataclass
class MixedRule:
    preds: tuple
    support: int
    tp: float
    precision: float
    recall: float
    val_precision: float = float("nan")
    val_recall: float = float("nan")
    gap: float = float("nan")
    depth: int = 0
    mask: np.ndarray = field(repr=False, default=None)
    def fkeys(self):
        return {p.fkey() for p in self.preds}
    def key(self):
        return frozenset(self.preds)
    def label(self, meta):
        return "  AND  ".join(p.label(meta) for p in self.preds)


def _mask(preds, M):
    m = np.ones(M.shape[0], dtype=bool)
    for p in preds:
        m &= p.mask(M)
    return m


def mixed_targeted_search(M, y, meta, *, min_recall=0.25, target_precision=0.5,
                          min_support=40, beam_width=12, max_depth=6,
                          M_val=None, y_val=None, gap_tol=0.20, max_accept=2000,
                          policy=None, progress=None):
    from .progress import make_progress
    N, F = M.shape
    total = float(y.sum())
    have_val = M_val is not None and y_val is not None
    total_val = float(y_val.sum()) if have_val else 0.0
    base = total / N
    accepted, pruned = [], {"recall": 0, "gap": 0}

    def make(preds, s, tp, depth):
        prec = tp / s if s else 0.0
        return MixedRule(tuple(preds), int(s), tp, prec, tp / total if total else 0.0,
                         depth=depth)

    def add_val(r):
        if have_val and np.isnan(r.val_precision):
            m = _mask(r.preds, M_val)
            sv = int(m.sum())
            tpv = float(y_val[m].sum())
            r.val_precision = tpv / sv if sv else 0.0
            r.val_recall = tpv / total_val if total_val else 0.0
            r.gap = r.precision - r.val_precision

    def settle(acc_c, grow_c):
        for r in acc_c:
            if len(accepted) >= max_accept:
                break
            add_val(r)
            if gap_tol is not None and have_val and r.gap > gap_tol:
                pruned["gap"] += 1
            else:
                accepted.append(r)
        # soft penalty: deprioritize discouraged feature co-occurrence on ties
        def rkey(r):
            pen = policy.rule_penalty({p.f for p in r.preds}) if policy else 0.0
            return (r.precision - pen, r.recall)
        grow_c.sort(key=rkey, reverse=True)
        beam = []
        for r in grow_c:
            if len(beam) >= beam_width:
                break
            add_val(r)
            if gap_tol is not None and have_val and r.gap > gap_tol:
                pruned["gap"] += 1
                continue
            r.mask = _mask(r.preds, M)
            beam.append(r)
        return beam

    def triage(r):
        if r.recall < min_recall:
            return "prune"
        return "accept" if r.precision >= target_precision else "grow"

    prog = make_progress(progress, "mixed")
    prog.start(F=F, beam=beam_width, max_depth=max_depth,
               targets=f"P>={target_precision} R>={min_recall} gap<={gap_tol}")

    def _report(d, beam_in, beam_out, explored, pr0):
        pool = beam_out + acc_c
        prog.depth(d, beam_in=beam_in, beam_out=len(beam_out), explored=explored,
                   accepted_new=len(accepted) - pr0[2], accepted_total=len(accepted),
                   prune_recall=pruned["recall"] - pr0[0],
                   prune_gap=pruned["gap"] - pr0[1],
                   best_p=max((r.precision for r in pool), default=0.0),
                   best_r=max((r.recall for r in pool), default=0.0))

    # depth 1
    pr0 = (pruned["recall"], pruned["gap"], len(accepted))
    acc_c, grow_c, explored = [], [], 0
    for f in range(F):
        for pred, s, tp in feature_cands(f, M[:, f], y, meta, min_support, base, policy):
            explored += 1
            r = make([pred], s, tp, 1)
            a = triage(r)
            if a == "prune":
                pruned["recall"] += 1
            elif a == "accept":
                acc_c.append(r)
            else:
                grow_c.append(r)
    beam = settle(acc_c, grow_c)
    _report(1, 1, beam, explored, pr0)

    # grow
    for depth in range(2, max_depth + 1):
        pr0 = (pruned["recall"], pruned["gap"], len(accepted))
        beam_in, explored = len(beam), 0
        acc_c, grow_d = [], {}
        for r in beam:
            idx = np.flatnonzero(r.mask)
            sub_y = y[idx]
            used = r.fkeys()
            rfeats = {fk[0] for fk in used}
            for f in range(F):
                if any(fk[0] == f for fk in used) and meta.kind[f] == "cat":
                    continue                       # one predicate per cat feature
                if policy is not None and not policy.extend_ok(rfeats, f, None):
                    continue                       # forbidden pair / mutual-excl
                col = M[idx, f]
                for pred, s, tp in feature_cands(f, col, sub_y, meta, min_support, base, policy):
                    if pred.fkey() in used:
                        continue
                    explored += 1
                    nr = make(r.preds + (pred,), s, tp, depth)
                    a = triage(nr)
                    if a == "prune":
                        pruned["recall"] += 1
                    elif a == "accept":
                        acc_c.append(nr)
                    else:
                        kk = nr.key()
                        if kk not in grow_d or nr.precision > grow_d[kk].precision:
                            grow_d[kk] = nr
        if not acc_c and not grow_d:
            break
        beam = settle(acc_c, list(grow_d.values()))
        _report(depth, beam_in, beam, explored, pr0)
        if (not beam and not acc_c) or len(accepted) >= max_accept:
            break

    # minimality: drop rules that strictly contain a simpler accepted rule
    uniq = {}
    for r in accepted:
        if r.key() not in uniq or r.precision > uniq[r.key()].precision:
            uniq[r.key()] = r
    kept, ksets = [], []
    for r in sorted(uniq.values(), key=lambda r: (len(r.preds), -r.precision)):
        if policy is not None and not policy.rule_ok({p.f for p in r.preds}):
            continue                              # required-with not satisfied
        fs = frozenset(r.preds)
        if any(ks <= fs for ks in ksets):
            continue
        kept.append(r)
        ksets.append(fs)
    out = sorted(kept, key=lambda r: (r.recall, r.precision), reverse=True)
    prog.done(rules=len(out), prune_recall=pruned["recall"], prune_gap=pruned["gap"],
              note=f"(of {len(uniq)} pre-minimality)")
    return out, pruned
