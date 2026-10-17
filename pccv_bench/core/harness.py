"""Shared instance/verdict schema and matcher invocation for the injection study.

Design note: this module is deliberately the
*only* place that calls the production matcher, and it takes its instances from
an abstract :class:`InstanceProvider`.  The real-data study
(:mod:`pccv_bench.core.projection`) and the future synthetic PCCV-Bench both
plug in here, so both compute metrics through identical code and the
"same code, two data sources" agreement claim holds automatically.

Nothing in this module reimplements geometry.  Box-side matching is the
production :func:`~pccv_bench.core.matching.auto_match_boxes_to_parts`;
record-side matching is :mod:`pccv_bench.core.twoway`, which itself reuses
:func:`~pccv_bench.core.matching.combo_metrics`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol
from pccv_bench.core.camera_model import CameraModelMapper
from pccv_bench.core.matching import auto_match_boxes_to_parts
from pccv_bench.core.inject import expected_action_for, fault_type_for
from pccv_bench.core.twoway import record_side_unmatched


@dataclass
class Instance:
    """One (observation, record) pair with a known consistency label.

    The *observed* side is ``detections`` (polygons in pixel space); the
    *expected* side is ``record_parts`` in the flat format produced by
    :func:`~pccv_bench.core.matching.build_gql_flat` (keys ``id``,
    ``kind``, ``kind_index``, ``polygon``, ``world``).

    Ground truth is two separate facts, not one boolean:
    ``fault_type`` is what is wrong with the pair and ``expected_action`` is what
    the system should do about it.  Both default to the taxonomy in
    :mod:`pccv_bench.core.inject` for the declared ``injection_class``; a study
    over *real* episodes rather than injected ones passes them explicitly, since there the truth comes from human adjudication
    and not from a perturbation we applied.
    """

    instance_id: str
    camera: str
    run_id: str
    detections: list[dict]
    record_parts: list[dict]
    image_wh: tuple[int, int]
    injection_class: str = "clean"
    fault_type: str | None = None
    expected_action: str | None = None
    severity: float = 0.0
    next_record_parts: list[dict] | None = None
    occlusion_stratum: str = "none"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fault_type is None:
            self.fault_type = fault_type_for(self.injection_class)
        if self.expected_action is None:
            self.expected_action = expected_action_for(self.injection_class)

    @property
    def should_alarm(self) -> bool:
        return self.expected_action == "alarm"


class InstanceProvider(Protocol):
    """Anything iterable of :class:`Instance` can be evaluated by :func:`run_arm`."""

    def __iter__(self) -> Iterator[Instance]: ...


@dataclass(frozen=True)
class ArmConfig:
    """One point in the ablation ladder.

    ``twoway`` enables the record-side coverage channel.  It is the only
    flag that can *add* alarms rather than remove them, which is why its cost is
    reported on the ``clean`` arm alongside its benefit on ``ghost_part``.
    """

    name: str
    metric: str = "containment"
    threshold: float = 0.3
    correction_offset_mm: float = 0.0
    use_temporal: bool = False
    twoway: bool = False
    twoway_threshold: float = 0.3
    #: Give the record-side channel the same bounded x-offset tolerance the
    #: box side has (reusing ``correction_offset_mm`` as the shared bound).
    #: Default False reproduces every previously-reported ``+all+twoway``
    #: number exactly; True is the fix for the measured 0.583 false-flag rate
    #: on in-bound shifts (see ``twoway.record_side_unmatched``).
    twoway_offset_aware: bool = False
    # Reference baselines that ignore the matcher entirely (a benchmark needs a
    # floor and a ceiling): "none" flags nothing, "all" flags everything.
    force: str | None = None
    #: One of :data:`pccv_bench.core.baselines.BASELINES`.  A competing method
    #: rather than an ablation: it replaces the production matcher outright, so
    #: the proposed method is measured against something that can express the
    #: same errors instead of only against weakened copies of itself.
    baseline: str | None = None
    #: Also search negative x offsets in the spatial correction (box side,
    #: offset-aware record side, and the coherent baseline).  Default False is
    #: the production one-sided search.
    two_sided_offset: bool = False
    #: Separate record-side threshold for the symmetric/coherent baselines;
    #: None applies ``threshold`` to both directions, as originally.
    baseline_record_threshold: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metric": self.metric,
            "threshold": self.threshold,
            "correction_offset_mm": self.correction_offset_mm,
            "use_temporal": self.use_temporal,
            "twoway": self.twoway,
            "twoway_threshold": self.twoway_threshold,
            "twoway_offset_aware": self.twoway_offset_aware,
            "force": self.force,
            "baseline": self.baseline,
            "two_sided_offset": self.two_sided_offset,
            "baseline_record_threshold": self.baseline_record_threshold,
        }


@dataclass
class Verdict:
    """The system's answer for one instance under one arm, plus diagnostics.

    Four things are kept apart here that a single ``is_inconsistent`` boolean
    and a 4-state ``temporal_state`` used to conflate:

    ``fault_type`` / ``expected_action``
        Ground truth, carried over from the instance.  Independent of any method.
    ``correction_outcome``
        What the **box side** did — one of :data:`CORRECTION_OUTCOMES`.  It is a
        statement about the production matcher only; it deliberately says nothing
        about the record-side channel, because the old field described the box
        side even when the record side had already raised the alarm.
    ``flagged``
        The **final alarm**, box side OR record side.  ``flagged_box_side`` and
        ``flagged_record_side`` stay separate so the ablation can attribute each
        direction's contribution, as the design requires.
    ``was_recovered`` / ``was_fault_masked``
        The two ways a correction can end an instance without an alarm: it
        absorbed something that was not a fault (success), or it explained away
        something that was (the dangerous case).  Both are only meaningful
        against the *final* alarm, so both are False whenever the system alarmed.

    ``run_id`` is carried over unchanged from the instance.  It is the
    independent capture event, and it is on the verdict so that
    :mod:`pccv_bench.core.statistics` can cluster by it directly: six cameras and every
    injection class share one event, so the verdicts derived from it are not
    independent observations.  It is a field rather than
    something recovered from ``instance_id`` because that identifier encodes
    the camera and frame stem, not the run.

    ``mixed_corrections`` records that different detections in the same scene
    were explained by different mechanisms.  Whether one *coherent* correction
    explains the whole capture is a separate question; this flag at
    least makes the incoherent cases visible instead of silently labelling them
    with whichever mechanism sorted first.
    """

    instance_id: str
    arm: str
    camera: str
    injection_class: str
    severity: float
    flagged: bool
    flagged_box_side: bool
    flagged_record_side: bool
    n_detections: int
    n_record_parts: int
    n_unmatched_boxes: int
    n_unmatched_parts: int
    run_id: str = ""
    correction_outcome: str = "unresolved"
    corrected_offset_mm: float = 0.0
    mixed_corrections: bool = False
    fault_type: str | None = None
    expected_action: str | None = None
    occlusion_stratum: str = "none"

    def __post_init__(self) -> None:
        if self.fault_type is None:
            self.fault_type = fault_type_for(self.injection_class)
        if self.expected_action is None:
            self.expected_action = expected_action_for(self.injection_class)

    @property
    def should_alarm(self) -> bool:
        return self.expected_action == "alarm"

    @property
    def _corrected(self) -> bool:
        return self.correction_outcome in ("recovered_by_offset", "recovered_temporally")

    @property
    def masking_mechanism(self) -> str | None:
        """Which correction explained a real fault away, if one did."""
        if not self.was_fault_masked:
            return None
        return "offset" if self.correction_outcome == "recovered_by_offset" else "temporal"

    @property
    def was_recovered(self) -> bool:
        """A correction absorbed a non-fault and no alarm was raised.

        This is the success case for both the ``recover`` classes (a snapshot
        desynchronization re-explained at t+1) and the ``pass`` classes (an
        in-bound shift absorbed by the offset sweep).
        """
        return (
            self.expected_action in ("pass", "recover")
            and self._corrected
            and not self.flagged
        )

    @property
    def was_fault_masked(self) -> bool:
        """A correction explained away a genuine physical fault."""
        return self.should_alarm and self._corrected and not self.flagged

    @property
    def diagnosis(self) -> str:
        """The box-side outcome crossed with ground truth, as one label."""
        if self.was_fault_masked:
            return (
                "fault_masked_by_offset"
                if self.correction_outcome == "recovered_by_offset"
                else "fault_masked_temporally"
            )
        return self.correction_outcome

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "arm": self.arm,
            "camera": self.camera,
            "run_id": self.run_id,
            "injection_class": self.injection_class,
            "fault_type": self.fault_type,
            "expected_action": self.expected_action,
            "should_alarm": self.should_alarm,
            "severity": self.severity,
            "flagged": self.flagged,
            "flagged_box_side": self.flagged_box_side,
            "flagged_record_side": self.flagged_record_side,
            "n_detections": self.n_detections,
            "n_record_parts": self.n_record_parts,
            "n_unmatched_boxes": self.n_unmatched_boxes,
            "n_unmatched_parts": self.n_unmatched_parts,
            "correction_outcome": self.correction_outcome,
            "corrected_offset_mm": self.corrected_offset_mm,
            "mixed_corrections": self.mixed_corrections,
            "was_recovered": self.was_recovered,
            "was_fault_masked": self.was_fault_masked,
            "masking_mechanism": self.masking_mechanism,
            "diagnosis": self.diagnosis,
            "occlusion_stratum": self.occlusion_stratum,
        }


def _baseline_flag(
    inst: Instance, arm: ArmConfig, mapper: CameraModelMapper | None
) -> bool:
    """Dispatch to one of the simple record-aware alternatives."""
    from pccv_bench.core import baselines

    if arm.baseline == "envelope":
        # Deliberately not passed arm.metric: this baseline is a whole-scene
        # IoU test by design (see the module docstring), not a metric-generic
        # one.
        return baselines.envelope_flag(
            inst.record_parts, inst.detections, arm.threshold, inst.image_wh
        )
    if arm.baseline == "symmetric":
        return baselines.symmetric_flag(
            inst.record_parts, inst.detections, arm.threshold, inst.image_wh,
            metric=arm.metric, record_threshold=arm.baseline_record_threshold,
        )
    if arm.baseline == "coherent":
        return baselines.coherent_flag(
            inst.record_parts,
            inst.detections,
            arm.threshold,
            inst.image_wh,
            metric=arm.metric,
            correction_offset_mm=arm.correction_offset_mm,
            mapper=mapper,
            next_record_parts=inst.next_record_parts if arm.use_temporal else None,
            record_threshold=arm.baseline_record_threshold,
            two_sided_offset=arm.two_sided_offset,
        )
    if arm.baseline == "assignment":
        return baselines.assignment_flag(
            inst.record_parts,
            inst.detections,
            arm.threshold,
            inst.image_wh,
            metric=arm.metric,
            correction_offset_mm=arm.correction_offset_mm,
            mapper=mapper,
            next_record_parts=inst.next_record_parts if arm.use_temporal else None,
            record_threshold=arm.baseline_record_threshold,
            two_sided_offset=arm.two_sided_offset,
        )
    if arm.baseline == "pairwise":
        return baselines.pairwise_flag(
            inst.record_parts, inst.detections, arm.threshold, inst.image_wh,
            metric=arm.metric,
        )
    raise ValueError(f"unknown baseline {arm.baseline!r}")


def evaluate_instance(
    inst: Instance,
    arm: ArmConfig,
    mapper: CameraModelMapper | None = None,
) -> Verdict:
    """Run one instance through one arm and return its :class:`Verdict`.

    The box-side decision is the production matcher, called exactly as the
    the production pipeline calls it.  The record-side decision (when ``arm.twoway``) is
    an additional channel; an instance is flagged when *either* channel raises.
    """
    iw, ih = inst.image_wh

    # Reference baselines short-circuit the matcher entirely.
    if arm.force == "none":
        return Verdict(
            instance_id=inst.instance_id, arm=arm.name, camera=inst.camera,
            run_id=inst.run_id,
            injection_class=inst.injection_class, fault_type=inst.fault_type,
            expected_action=inst.expected_action,
            severity=inst.severity, flagged=False, flagged_box_side=False,
            flagged_record_side=False, n_detections=len(inst.detections),
            n_record_parts=len(inst.record_parts), n_unmatched_boxes=0,
            n_unmatched_parts=0, correction_outcome="matched_at_t",
            occlusion_stratum=inst.occlusion_stratum,
        )
    if arm.force == "all":
        return Verdict(
            instance_id=inst.instance_id, arm=arm.name, camera=inst.camera,
            run_id=inst.run_id,
            injection_class=inst.injection_class, fault_type=inst.fault_type,
            expected_action=inst.expected_action,
            severity=inst.severity, flagged=True, flagged_box_side=True,
            flagged_record_side=False, n_detections=len(inst.detections),
            n_record_parts=len(inst.record_parts),
            n_unmatched_boxes=len(inst.detections), n_unmatched_parts=0,
            correction_outcome="unresolved",
            occlusion_stratum=inst.occlusion_stratum,
        )

    if arm.baseline:
        # A competing method, not an ablation: it replaces the matcher rather
        # than weakening it, so nothing below runs.  Baselines return one scene
        # decision and do not attribute it to a direction, so the box/record
        # split is reported as the single decision it actually was rather than
        # invented.
        flagged = _baseline_flag(inst, arm, mapper)
        return Verdict(
            instance_id=inst.instance_id, arm=arm.name, camera=inst.camera,
            run_id=inst.run_id,
            injection_class=inst.injection_class, fault_type=inst.fault_type,
            expected_action=inst.expected_action,
            severity=inst.severity, flagged=flagged, flagged_box_side=flagged,
            flagged_record_side=False, n_detections=len(inst.detections),
            n_record_parts=len(inst.record_parts),
            n_unmatched_boxes=0, n_unmatched_parts=0,
            correction_outcome="unresolved" if flagged else "matched_at_t",
            occlusion_stratum=inst.occlusion_stratum,
        )

    next_parts = inst.next_record_parts if arm.use_temporal else None

    result = auto_match_boxes_to_parts(
        gql_parts_with_ids=inst.record_parts,
        predictions=inst.detections,
        metric=arm.metric,
        threshold=arm.threshold,
        iw=iw,
        ih=ih,
        correction_offset_mm=arm.correction_offset_mm,
        mapper=mapper,
        next_run_gql_parts=next_parts,
        two_sided_offset=arm.two_sided_offset,
    )

    unmatched_boxes = result["unmatched_boxes"]
    flagged_box_side = bool(unmatched_boxes)

    # Which correction (if any) explained this instance.
    groups = result["groups"]
    methods = [g.get("correction_method") for g in groups]
    corrected_offsets = [
        float(g.get("corrected_offset_mm") or 0.0)
        for g in groups
        if g.get("correction_method") == "x_offset"
    ]
    used = {m for m in methods if m}

    # An offset rescue is a box-side *success*.  Recording it as neither
    # "matched at t" nor "matched at t+1" — as the two booleans this replaces
    # did — filed 174 such successes under "no match".
    if flagged_box_side:
        correction_outcome = "unresolved"
    elif "temporal" in used:
        correction_outcome = "recovered_temporally"
    elif "x_offset" in used:
        correction_outcome = "recovered_by_offset"
    else:
        correction_outcome = "matched_at_t"

    # ── Record-side channel (two-way matching) ──────────────────────────
    unmatched_parts: list[int] = []
    if arm.twoway and inst.record_parts:
        unmatched_parts = record_side_unmatched(
            record_parts=inst.record_parts,
            detections=inst.detections,
            threshold=arm.twoway_threshold,
            image_wh=inst.image_wh,
            correction_offset_mm=(arm.correction_offset_mm if arm.twoway_offset_aware else 0.0),
            mapper=mapper if arm.twoway_offset_aware else None,
            two_sided_offset=arm.two_sided_offset,
        )
    flagged_record_side = bool(unmatched_parts)

    return Verdict(
        instance_id=inst.instance_id,
        arm=arm.name,
        camera=inst.camera,
        run_id=inst.run_id,
        injection_class=inst.injection_class,
        fault_type=inst.fault_type,
        expected_action=inst.expected_action,
        severity=inst.severity,
        flagged=flagged_box_side or flagged_record_side,
        flagged_box_side=flagged_box_side,
        flagged_record_side=flagged_record_side,
        n_detections=len(inst.detections),
        n_record_parts=len(inst.record_parts),
        n_unmatched_boxes=len(unmatched_boxes),
        n_unmatched_parts=len(unmatched_parts),
        correction_outcome=correction_outcome,
        corrected_offset_mm=max(corrected_offsets) if corrected_offsets else 0.0,
        mixed_corrections=len(used) > 1,
        occlusion_stratum=inst.occlusion_stratum,
    )


def run_arm(
    provider: InstanceProvider,
    arm: ArmConfig,
    mappers: dict[str, CameraModelMapper | None] | None = None,
    progress_every: int = 0,
) -> list[Verdict]:
    """Evaluate every instance from *provider* under *arm*.

    *mappers* maps camera name → fitted :class:`CameraModelMapper`, required
    only when ``arm.correction_offset_mm > 0`` (the x-offset sweep re-projects
    world-space parts).
    """
    mappers = mappers if mappers is not None else {}
    verdicts: list[Verdict] = []
    for i, inst in enumerate(provider, start=1):
        verdicts.append(
            evaluate_instance(inst, arm, mapper=mappers.get(inst.camera))
        )
        if progress_every and i % progress_every == 0:
            print(f"    [{arm.name}] {i} instances…", flush=True)
    return verdicts


MAX_X_OFFSET_MM: float = 600.0


def default_arms(twoway_threshold: float = 0.3) -> list[ArmConfig]:
    """The ablation ladder: geometry → +offset → +temporal → +both → +twoway.

    ``+all+occlusion`` is not a separate arm here because robot filtering is a
    caller-side pre-filter, not a matcher flag; the provider applies it and records the stratum, so occlusion
    is reported as a *stratification* of every arm instead.

    ``+all+twoway+offset_aware`` is the fix for the record-side channel's
    measured 0.583 false-flag rate on in-bound shifts (it had no tolerance for
    the same camera-wobble the box side is designed to excuse). It is listed
    *after* ``+all+twoway`` on purpose so the before/after is directly visible
    in the summary table, and ``+all+twoway`` itself is left unchanged so every
    previously-reported number stays reproducible.
    """
    base = dict(metric="containment", threshold=0.3)
    return [
        ArmConfig(name="flag_none", force="none", **base),
        ArmConfig(name="flag_all", force="all", **base),
        # Competing methods, not ablations — see pccv_bench/core/baselines.py.
        ArmConfig(name="baseline_envelope", baseline="envelope", **base),
        ArmConfig(name="baseline_symmetric", baseline="symmetric", **base),
        ArmConfig(
            name="baseline_coherent",
            baseline="coherent",
            correction_offset_mm=MAX_X_OFFSET_MM,
            use_temporal=True,
            **base,
        ),
        ArmConfig(name="baseline_pairwise", baseline="pairwise", **base),
        ArmConfig(name="geometry", correction_offset_mm=0.0, use_temporal=False, **base),
        ArmConfig(name="+offset", correction_offset_mm=MAX_X_OFFSET_MM, use_temporal=False, **base),
        ArmConfig(name="+temporal", correction_offset_mm=0.0, use_temporal=True, **base),
        ArmConfig(
            name="+offset+temporal",
            correction_offset_mm=MAX_X_OFFSET_MM,
            use_temporal=True,
            **base,
        ),
        # Each correction arm's two-way counterpart.  Without these the study
        # cannot separate "the corrections help" from "the record-side channel
        # helps": the only two-way arm used to be the one with everything on.
        ArmConfig(
            name="geometry+twoway",
            correction_offset_mm=0.0,
            use_temporal=False,
            twoway=True,
            twoway_threshold=twoway_threshold,
            **base,
        ),
        ArmConfig(
            name="+offset+twoway",
            correction_offset_mm=MAX_X_OFFSET_MM,
            use_temporal=False,
            twoway=True,
            twoway_threshold=twoway_threshold,
            **base,
        ),
        ArmConfig(
            name="+temporal+twoway",
            correction_offset_mm=0.0,
            use_temporal=True,
            twoway=True,
            twoway_threshold=twoway_threshold,
            **base,
        ),
        ArmConfig(
            name="+all+twoway",
            correction_offset_mm=MAX_X_OFFSET_MM,
            use_temporal=True,
            twoway=True,
            twoway_threshold=twoway_threshold,
            **base,
        ),
        ArmConfig(
            name="+all+twoway+offset_aware",
            correction_offset_mm=MAX_X_OFFSET_MM,
            use_temporal=True,
            twoway=True,
            twoway_threshold=twoway_threshold,
            twoway_offset_aware=True,
            **base,
        ),
    ]
