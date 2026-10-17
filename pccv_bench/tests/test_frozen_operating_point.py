"""E7 (PCCV-Bench agreement study, PREREGISTRATION.md §1): operating-point
alignment. ``build_arms(frozen=...)`` must score every arm -- ablation floors
and baselines included -- at the real study's frozen metric/threshold/
twoway_threshold, exactly as ``eval/eval_injection_benchmark.py:505`` does it
for the real study (``dataclasses.replace`` over ``default_arms``). Without
``frozen``, the ladder must be exactly what it always was.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.ablate import build_arms, default_arms

FROZEN = {"metric": "iou", "threshold": 0.7, "twoway_threshold": 0.4}


def test_every_arm_matches_the_frozen_point_when_aligned():
    arms = build_arms(twoway_threshold=0.3, frozen=FROZEN)
    assert len(arms) == len(default_arms())
    for arm in arms:
        assert arm.metric == FROZEN["metric"], arm.name
        assert arm.threshold == FROZEN["threshold"], arm.name
        # twoway_threshold only governs arms that actually use the two-way
        # (record-side) channel; ArmConfig's own default (0.3) is carried
        # through unused on every other arm, exactly as default_arms() itself
        # already behaves -- asserting equality there would test a field the
        # arm never consults, not what E7 changed.
        if arm.twoway:
            assert arm.twoway_threshold == FROZEN["twoway_threshold"], arm.name


def test_baselines_are_included_in_the_alignment():
    arms = {a.name: a for a in build_arms(frozen=FROZEN)}
    baseline_names = [n for n in arms if n.startswith("baseline_")]
    assert baseline_names, "expected at least one baseline arm in the ladder"
    for name in baseline_names:
        assert arms[name].metric == FROZEN["metric"], name
        assert arms[name].threshold == FROZEN["threshold"], name


def test_every_other_field_is_preserved_by_the_replace():
    """dataclasses.replace must carry every field but metric/threshold/
    twoway_threshold forward unchanged -- baseline, correction_offset_mm,
    use_temporal, twoway, twoway_offset_aware, force, ... -- so aligning the
    operating point changes only what E7 says it changes."""
    plain = {a.name: a for a in default_arms(twoway_threshold=FROZEN["twoway_threshold"])}
    aligned = {a.name: a for a in build_arms(twoway_threshold=0.3, frozen=FROZEN)}
    assert set(plain) == set(aligned)
    for name in plain:
        for field in ("baseline", "correction_offset_mm", "use_temporal", "twoway", "twoway_offset_aware", "force"):
            assert getattr(plain[name], field) == getattr(aligned[name], field), (name, field)


def test_without_frozen_the_ladder_is_unchanged():
    assert build_arms(twoway_threshold=0.55) == default_arms(twoway_threshold=0.55)


def test_frozen_twoway_threshold_overrides_the_configs_own():
    """The two-way arms' record-side threshold must come from `frozen`, not
    from the generating config's own twoway_threshold -- default_arms() is
    built with the frozen twoway_threshold from the start (not patched
    in afterwards by the replace), exactly mirroring how the real study
    builds its arms (eval_injection_benchmark.py:505-506)."""
    arms = {a.name: a for a in build_arms(twoway_threshold=0.3, frozen=FROZEN)}
    assert arms["+all+twoway+offset_aware"].twoway_threshold == FROZEN["twoway_threshold"]
    assert arms["+all+twoway+offset_aware"].twoway_threshold != 0.3
