"""Matched-budget threshold selection for the baseline methods.

Picks each baseline's threshold on tuning instances only, subject to a cap on
the clean-control flag rate, so methods are compared at equal nuisance-alarm
cost rather than at their own natural operating points."""

from __future__ import annotations

from pccv_bench.core.harness import MAX_X_OFFSET_MM


# NOTE: for these two baselines, *raising* `threshold` makes them flag MORE,
# not less -- symmetric_flag's checks require geometry to clear `threshold`
# to count as "explained", so a stricter (higher) bar means more scenes fail
# it. An exploratory sweep confirmed this empirically: at the baselines'
# natural threshold (0.7) clean false-error is 42.0%/40.1%; at 0.005 it is
# already down near single digits. So the informative range for matching a
# low false-alarm budget is near zero, not near 1 -- this grid is fine-
# grained there and coarser above it only to bound the sweep.
THRESHOLD_GRID = [
    0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02,
    0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70,
]


def _baseline_kwargs(baseline: str) -> dict:
    """Extra ArmConfig fields for *baseline*, matching ``default_arms()``'s own
    registration exactly (``harness.py``) -- ``baseline_coherent`` is the one
    baseline that also searches the offset/temporal grid, and forgetting
    those two fields here would silently degenerate it into plain
    ``symmetric`` for the whole sweep."""
    if baseline == "coherent":
        return {"correction_offset_mm": MAX_X_OFFSET_MM, "use_temporal": True}
    return {}


def pick_matched_threshold(rows: list[dict], budget: float) -> dict | None:
    """Same rule as ``matched_false_alarm_budgets``: among rows affordable
    within *budget*, the one with the highest pooled detection rate --
    missing_part_recall is not consulted, so the pick cannot be tuned
    toward a preferred answer on the metric actually in dispute."""
    affordable = [r for r in rows if r["clean_false_error"] is not None and r["clean_false_error"] <= budget]
    if not affordable:
        return None
    return max(affordable, key=lambda r: r["detection_rate_pooled"] or 0.0)
