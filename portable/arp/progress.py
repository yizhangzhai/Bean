"""Lightweight progress reporting for the miners.

Opt-in and side-effect-free when disabled. Pass `progress=True` to a miner to
get a per-depth view of the search:

  ┌─ targeted[account_takeover]: F=60  beam=12  max_depth=4  P>=0.50 R>=0.25 gap<=0.25
  │  depth 1  beam   1→ 12  explored    118  accept +0 (Σ0)    prune recall=982 gap=0    bestP=0.33 R=0.81   0.0s
  │  depth 2  beam  12→ 12  explored  1,416  accept +3 (Σ3)    prune recall=11,003 gap=12  bestP=0.50 R=0.31   0.4s
  └─ done  5 rules (after minimality)  Σprune recall=23,901 gap=44   2 depths   0.6s

Writes to stderr by default so it never pollutes stdout data.
"""

from __future__ import annotations

import sys
import time


class Progress:
    def __init__(self, enabled=True, stream=None, label="mine"):
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.label = label
        self.t0 = time.perf_counter()
        self.depths = 0

    def _w(self, s):
        if self.enabled:
            print(s, file=self.stream, flush=True)

    def elapsed(self):
        return time.perf_counter() - self.t0

    def start(self, **info):
        self.t0 = time.perf_counter()
        kv = "  ".join(f"{k}={v}" for k, v in info.items())
        self._w(f"┌─ {self.label}: {kv}")

    def depth(self, d, *, beam_in=0, beam_out=0, explored=0, accepted_new=0,
              accepted_total=0, prune_recall=0, prune_gap=0,
              best_p=0.0, best_r=None):
        self.depths = max(self.depths, d)
        # precision/recall miners show "bestP=.. R=.."; the lift-based fast miner
        # passes best_r=None and shows a single best score instead
        quality = (f"best={best_p:.2f}" if best_r is None
                   else f"bestP={best_p:.2f} R={best_r:.2f}")
        prune = (f"  prune recall={prune_recall:,} gap={prune_gap:,}"
                 if (prune_recall or prune_gap) else "")
        grown = (f"  collection Σ{accepted_total}" if accepted_new == 0
                 and prune_recall == 0 and prune_gap == 0
                 else f"  accept +{accepted_new} (Σ{accepted_total})")
        self._w(
            f"│  depth {d}  beam {beam_in:>3}→{beam_out:>3}  "
            f"explored {explored:>8,}{grown}{prune}  "
            f"{quality}  {self.elapsed():.1f}s")

    def tick(self, msg):
        self._w(f"│   {msg}")

    def done(self, *, rules=0, prune_recall=0, prune_gap=0, note=""):
        extra = f"  {note}" if note else ""
        self._w(
            f"└─ done  {rules} rules{extra}  "
            f"Σprune recall={prune_recall:,} gap={prune_gap:,}  "
            f"{self.depths} depths  {self.elapsed():.1f}s")


def make_progress(progress, label) -> Progress:
    """Normalize a progress arg (None | bool | Progress) into a Progress."""
    if isinstance(progress, Progress):
        return progress
    return Progress(enabled=bool(progress), label=label)
