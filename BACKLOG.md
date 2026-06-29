# Backlog

Deferred work — not blocking; pick up when needed.

## Feature engineering
- **Nonlinear weight fitting for comparison features.** Today, weights are learnable
  only for *linear* combinations: `formats=("linear",)` (per-pair `w1·A + w2·B`) and
  `engineer=dict(learn=[(label, cols)])` (`Σ wᵢ·xᵢ` over any columns), both fit by
  logistic regression on the residual. A weight buried *inside* a nonlinear term —
  the `w` in a product/ratio such as `(C−D)·w·A/B > t` where `w` is not the cut —
  is **not** auto-fit, because the margin is nonlinear in `w` and logistic
  regression only fits linear combinations.
  - Current workaround: fix the weight in `fn`, or rewrite so the weight becomes the
    threshold (`A > w·B` ⇔ `A/B > w`, and the learned cut is `w`).
  - Future: a small per-expression optimizer (e.g. fit `w` by maximizing residual
    separation / lift over a 1-D or low-D grid or gradient step) for parametric
    nonlinear families.
  - Status: deferred — only linear is needed for now (user, 2026-06-29).
