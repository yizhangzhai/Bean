"""featgap -- gap-driven feature engineering, a SEPARATE layer on top of `arp`.

The core `arp` package mines interpretable axis-aligned rules. `featgap` sits on
top of it (one-directional dependency: featgap imports arp, never the reverse)
and handles what the rules CANNOT express on the given features:

  gap.py        identify the gap (uncovered positives) + the cheap step-1
                diagnostic: re-mine the residual to tell "greedy myopia" from a
                "genuine non-axis gap".
  screen.py     step-3 interaction screens: interaction-information (fast) and
                HSIC (kernel dependence) over feature pairs vs the residual --
                finds interactions with no marginal signal.
  synthesize.py diagnose geometry (topology-lite ring/void) and synthesize +
                rank candidate features (radial / ratio / diff) for re-mining.

Recommended triage (cheap -> fancy): re-mine residual -> interaction screen ->
geometry/synthesize -> re-mine. Stop as soon as the gap is explained.
"""

from .gap import uncovered_positives, remine_residual
from .screen import mutual_information, interaction_screen, hsic
from .synthesize import best_band, ring_score, propose_features

__all__ = [
    "uncovered_positives", "remine_residual",
    "mutual_information", "interaction_screen", "hsic",
    "best_band", "ring_score", "propose_features",
]
