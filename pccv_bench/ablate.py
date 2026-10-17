"""Ablation runner.

Thin by design: the arm ladder and the matcher-invocation loop already live in
``pccv_bench.core.harness`` (built for the real-data study, whose "shared
harness" requirement it meets), so this module only re-exports it and — the one
PCCV-Bench-specific addition — needs a ``mappers`` dict at call time, since
the harness's offset-correction sweep re-projects world-space parts and needs
each instance's camera's fitted :class:`CameraModelMapper` to do so.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.core.harness import ArmConfig, Instance, Verdict, default_arms, run_arm

__all__ = ["ArmConfig", "Instance", "Verdict", "build_arms", "default_arms", "run_arm", "run_all_arms"]


def build_arms(twoway_threshold: float = 0.3, frozen: dict | None = None) -> list[ArmConfig]:
    """The default ladder, optionally aligned to the real study's frozen point.

    ``frozen`` (PCCV-Bench agreement study E7): when given, every arm --
    baselines included -- is scored at the real study's frozen operating
    point instead of its own native one. ``frozen`` is a mapping with
    ``metric``, ``threshold``, and ``twoway_threshold`` keys (the shape of
    ``frozen_operating_point.json``'s ``"frozen"`` object). Applied with
    :func:`dataclasses.replace` exactly as the real-data study does it: build
    the ladder with ``twoway_threshold`` already at the frozen value (so the
    two-way arms' record-side threshold is correct), then replace ``metric``
    and ``threshold`` on every arm, ablation floors and baselines included.
    """
    tw = frozen["twoway_threshold"] if frozen is not None else twoway_threshold
    arms = default_arms(twoway_threshold=tw)
    if frozen is not None:
        arms = [
            dataclasses.replace(a, metric=frozen["metric"], threshold=frozen["threshold"])
            for a in arms
        ]
    return arms


def run_all_arms(
    instances: list[Instance],
    mappers: dict,
    twoway_threshold: float = 0.3,
    frozen: dict | None = None,
) -> dict[str, list[Verdict]]:
    """Run every arm in the default ladder over the same instance set.

    See :func:`build_arms` for what ``frozen`` does.
    """
    arms = build_arms(twoway_threshold=twoway_threshold, frozen=frozen)
    return {arm.name: run_arm(instances, arm, mappers=mappers) for arm in arms}
