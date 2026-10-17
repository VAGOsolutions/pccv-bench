"""Record-side perturbations with known ground truth.

Every injector takes the *world-space* record parts of a verified-consistent
instance and returns a perturbed copy plus metadata.  Perturbing the record
rather than the image is what makes the ground-truth label exact: we control the
GQL side, so we know precisely which inconsistency we created and how severe it
is, without needing a single new human label.

Ground-truth convention
-----------------------
Two orthogonal facts are recorded per class, and conflating them was the
measurement defect an independent review of an earlier version found:

``FAULT_TYPE`` says what is actually wrong with the pair — ``physical`` (the
record and the world genuinely disagree), ``timing`` (the record is a correct
description of a *different* moment), ``alignment`` (nothing is wrong; the
projection is offset by less than the calibration tolerance), or ``none``.

``EXPECTED_ACTION`` says what the system is supposed to do about it — ``alarm``,
``pass``, or ``recover``.  It is not "was the record modified":

- ``position_shift_x_inbound`` shifts parts by **less** than the offset-search
  bound.  The bounded x-offset correction exists precisely to absorb residual
  calibration misalignment of that magnitude, so the expected action is ``pass``.
  Flagging one of these is a false error, and the rate at which the correction
  absorbs them is its recovery rate.
- ``position_shift_out``, ``missing_part``, ``ghost_part`` and
  ``composition_swap`` are physical faults, so their expected action is
  ``alarm`` and a non-flag there is a *masked* fault.  Sweeping the offset bound
  trades those two quantities against each other — the benefit-vs-masking
  curve.
- ``snapshot_desync`` is a **timing** condition whose expected action is
  ``recover``: the record does not describe this image, and the temporal retry
  exists to notice that and re-explain it.  Recovering it is success, not a
  masked fault, and raising an alarm instead is a defensible escalation rather
  than a false error — which is why it belongs to neither the detection-rate nor
  the false-error denominator.  Scoring it as an alarm condition is what
  previously recorded 22 successful recoveries as ``masked_by_t_plus_1``.

``severity`` is always defined geometrically (area or envelope delta), never as
a part count, so the difficulty gradient is comparable across classes.

Directions, stated once
-----------------------------------
The two presence classes are named after what was done to the *record*, which
has repeatedly been read as a claim about the *world*.  In terms of what the
camera sees:

- ``missing_part`` deletes a part from the record, so the image contains metal
  the record does not account for: **observed-but-unexpected**.
- ``ghost_part`` adds a part to the record, so the record claims metal that is
  not in the image: **expected-but-unobserved**.  A part physically removed
  *while the record still shows it present* would produce this direction, and
  the box-centric matcher is structurally blind to it.

Whether a perturbation in either direction is actually *observable* is decided
separately, and never by the matcher under evaluation — see
:mod:`pccv_bench.core.oracle`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from pccv_bench.core.geometry import rect_corners_world


INJECTION_CLASSES = (
    "clean",
    "missing_part",
    "ghost_part",
    "position_shift_x_inbound",
    "position_shift_out",
    "snapshot_desync",
    "composition_swap",
)


#: What is actually wrong with the pair, independent of any method.
FAULT_TYPE: dict[str, str] = {
    "clean": "none",
    "missing_part": "physical",
    "ghost_part": "physical",
    "position_shift_x_inbound": "alignment",
    "position_shift_out": "physical",
    "snapshot_desync": "timing",
    "composition_swap": "physical",
}


#: What the system is supposed to do about it.  See the module docstring — only
#: a physical fault is an alarm condition.
EXPECTED_ACTION: dict[str, str] = {
    "clean": "pass",
    "missing_part": "alarm",
    "ghost_part": "alarm",
    "position_shift_x_inbound": "pass",
    "position_shift_out": "alarm",
    "snapshot_desync": "recover",
    "composition_swap": "alarm",
}


def fault_type_for(injection_class: str) -> str:
    return FAULT_TYPE.get(injection_class, "physical")


def expected_action_for(injection_class: str) -> str:
    return EXPECTED_ACTION.get(injection_class, "alarm")


# Minimum envelope-geometry delta for a composition swap to be emitted at all
# (a scope correction): swaps that leave the group
# envelope effectively unchanged are NOT inconsistencies under this system's
# single-class semantics and must be rejected rather than mislabelled.
MIN_ENVELOPE_DELTA: float = 0.15


@dataclass
class InjectionContext:
    """Everything an injector may need beyond the parts themselves.

    ``project`` maps a list of world-part dicts to pixel polygons; it is supplied
    by the provider so injectors can verify that what they created actually
    lands in the image (and, for ghost parts, that nothing was detected there).
    ``donor_parts`` are real world-space parts observed on the same
    camera+table elsewhere in the dataset, used to place ghost parts with
    genuine production geometry rather than invented coordinates.
    """

    camera: str
    image_wh: tuple[int, int]
    project: Callable[[list[dict]], list[list[list[float]]]]
    detections: list[dict]
    donor_parts: list[dict]
    max_offset_mm: float = 600.0


@dataclass
class InjectionResult:
    """A perturbed record plus its ground-truth label and diagnostics.

    The label is carried by ``injection_class`` alone: :data:`FAULT_TYPE` and
    :data:`EXPECTED_ACTION` derive from it, so an injector cannot disagree with
    the taxonomy.  A per-result ``should_flag`` boolean used to duplicate that
    mapping, which is exactly how ``snapshot_desync`` came to be labelled an
    alarm condition in one place and a recovery condition in another.

    ``ops`` is a *replayable* description of what was done, so the identical
    perturbation can be applied to the t+1 record via :func:`apply_ops`.  That
    replay is not cosmetic — see :func:`apply_ops` for why omitting it silently
    turns the temporal retry into an oracle and drives every detection rate to
    zero.
    """

    parts_world: list[dict]
    injection_class: str
    severity: float
    ok: bool = True
    reason: str = ""
    meta: dict[str, Any] = None  # type: ignore[assignment]
    ops: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}
        if self.ops is None:
            self.ops = {"kind": "none"}


def apply_ops(parts_world: list[dict], ops: dict[str, Any]) -> list[dict]:
    """Replay an injection's perturbation on another record (the t+1 snapshot).

    Why this exists
    ---------------
    The injected classes model *a wrong record*.  If the perturbation is applied
    only at time t and the temporal retry is then handed a pristine t+1 record,
    that retry becomes a perfect oracle: consecutive runs usually describe an
    almost identical table state, so the untouched t+1 record reconstructs the
    truth we just removed and explains away the injected error.  Measured on a
    smoke run, that pushed the box-side detection rate to **exactly 0.000 for
    every metric and threshold** — an artefact of the harness, not a property of
    the matcher.

    Replaying the same operation on both snapshots models the realistic failure
    ("the record is wrong, and stays wrong") and leaves the temporal mechanism a
    legitimate chance to recover only when the table state genuinely changed
    between runs.  That residual is what ``fault_masked_temporally`` counts.

    ``snapshot_desync`` deliberately does **not** use this: putting the true
    record at t+1 is precisely the error that class injects.
    """
    kind = ops.get("kind", "none")
    if kind == "none":
        return [dict(p) for p in parts_world]

    if kind == "drop_ids":
        drop = set(ops.get("ids", []))
        return [dict(p) for p in parts_world if str(p.get("id", "")) not in drop]

    if kind == "add_parts":
        return [dict(p) for p in parts_world] + [dict(p) for p in ops.get("parts", [])]

    if kind == "translate":
        axis = ops.get("axis", "x")
        mm = float(ops.get("mm", 0.0))
        out = []
        for p in parts_world:
            q = dict(p)
            q[axis] = float(q.get(axis, 0.0)) + mm
            out.append(q)
        return out

    if kind == "swap_ids":
        a_id, b_id = str(ops.get("a", "")), str(ops.get("b", ""))
        out = [dict(p) for p in parts_world]
        ia = next((i for i, p in enumerate(out) if str(p.get("id", "")) == a_id), None)
        ib = next((i for i, p in enumerate(out) if str(p.get("id", "")) == b_id), None)
        if ia is None or ib is None:
            return out  # the corresponding parts are absent at t+1; nothing to swap
        out[ia]["x"], out[ib]["x"] = out[ib].get("x", 0.0), out[ia].get("x", 0.0)
        out[ia]["y"], out[ib]["y"] = out[ib].get("y", 0.0), out[ia].get("y", 0.0)
        return out

    raise ValueError(f"unknown op kind {kind!r}")


def _part_area(part: dict) -> float:
    return abs(float(part.get("sizeX", 0.0)) * float(part.get("sizeY", 0.0)))


def _world_envelope(parts: list[dict]) -> tuple[float, float, float, float] | None:
    """Axis-aligned world-space envelope of *parts* as ``(x1, y1, x2, y2)``."""
    pts: list[np.ndarray] = []
    for p in parts:
        sx, sy = float(p.get("sizeX", 0.0)), float(p.get("sizeY", 0.0))
        if sx <= 0 or sy <= 0:
            continue
        pts.append(
            rect_corners_world(
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                sx,
                sy,
                float(p.get("rotation", p.get("r", 0.0)) or 0.0),
            )
        )
    if not pts:
        return None
    allp = np.concatenate(pts, axis=0)
    return (
        float(allp[:, 0].min()),
        float(allp[:, 1].min()),
        float(allp[:, 0].max()),
        float(allp[:, 1].max()),
    )


def _envelope_delta(before: list[dict], after: list[dict]) -> float:
    """1 - IoU of the two world-space envelopes (0 = identical, 1 = disjoint).

    This is the severity definition for
    ``composition_swap``: a *geometric* delta, not a count of swapped parts.
    """
    eb, ea = _world_envelope(before), _world_envelope(after)
    if eb is None or ea is None:
        return 1.0
    ix1, iy1 = max(eb[0], ea[0]), max(eb[1], ea[1])
    ix2, iy2 = min(eb[2], ea[2]), min(eb[3], ea[3])
    iw_, ih_ = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw_ * ih_
    area_b = max(0.0, eb[2] - eb[0]) * max(0.0, eb[3] - eb[1])
    area_a = max(0.0, ea[2] - ea[0]) * max(0.0, ea[3] - ea[1])
    union = area_b + area_a - inter
    if union <= 0:
        return 1.0
    return float(1.0 - inter / union)


def _polygon_in_image(poly: list[list[float]], iw: int, ih: int, min_frac: float = 0.6) -> bool:
    """True when at least *min_frac* of the polygon's bbox lies inside the image.

    Projected coordinates are clipped to the image by the projection helpers, so
    a part that is mostly off-frame collapses to a sliver at the border; this
    check rejects such degenerate ghosts.
    """
    arr = np.asarray(poly, dtype=np.float64)
    if len(arr) < 3:
        return False
    x1, y1 = arr[:, 0].min(), arr[:, 1].min()
    x2, y2 = arr[:, 0].max(), arr[:, 1].max()
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return False
    # Distance from each border, as a fraction of the box's own size.
    inside_w = min(x2, iw) - max(x1, 0)
    inside_h = min(y2, ih) - max(y1, 0)
    if inside_w <= 0 or inside_h <= 0:
        return False
    return (inside_w * inside_h) / (w * h) >= min_frac


def _overlaps_any_detection(
    poly: list[list[float]],
    detections: list[dict],
    max_overlap: float = 0.05,
) -> bool:
    """True when *poly* overlaps a detection more than *max_overlap* of its area.

    Used to guarantee a ghost part really is a ghost: the record claims a part
    somewhere nothing was observed.
    """
    from pccv_bench.core.twoway import record_part_coverage

    if not detections:
        return False
    return record_part_coverage(poly, detections) > max_overlap


def inject_clean(parts_world: list[dict], rng: np.random.Generator, ctx: InjectionContext) -> InjectionResult:
    """The control arm: the record is returned untouched."""
    return InjectionResult(
        parts_world=[dict(p) for p in parts_world],
        injection_class="clean",
        severity=0.0,
    )


def inject_missing_part(
    parts_world: list[dict], rng: np.random.Generator, ctx: InjectionContext
) -> InjectionResult:
    """Drop k parts from the record while keeping every detection.

    The observable consequence is **observed-but-unexpected** metal: the image
    still shows the parts, the record no longer accounts for them.

    Severity is the fraction of total recorded part *area* removed — a
    geometric proxy for how much observed extent is left unexplained.  Dropping
    one of four parts can legitimately still pass at ``containment@0.3``; that
    is the intended difficulty gradient, not a defect.
    """
    if not parts_world:
        return InjectionResult([], "missing_part", 0.0, ok=False, reason="no_parts")
    n = len(parts_world)
    k = 1 if n == 1 else int(rng.integers(1, n))  # keep at least one part when n>1
    idx = list(rng.choice(n, size=k, replace=False))
    total_area = sum(_part_area(p) for p in parts_world) or 1.0
    dropped_area = sum(_part_area(parts_world[i]) for i in idx)
    dropped_ids = [str(parts_world[i].get("id", "")) for i in idx]
    kept = [dict(p) for i, p in enumerate(parts_world) if i not in idx]
    return InjectionResult(
        parts_world=kept,
        injection_class="missing_part",
        severity=round(dropped_area / total_area, 4),
        meta={"n_dropped": k, "n_before": n, "dropped_ids": dropped_ids},
        ops={"kind": "drop_ids", "ids": dropped_ids},
    )


def inject_ghost_part(
    parts_world: list[dict], rng: np.random.Generator, ctx: InjectionContext
) -> InjectionResult:
    """Add a recorded part where nothing was detected.

    The observable consequence is an **expected-but-unobserved** part, which is
    also what a physical removal with an unchanged record looks like.

    Placement uses **real part geometry observed elsewhere on the same camera
    and table** (``ctx.donor_parts``) rather than invented world coordinates, so
    the ghost is production-plausible by construction.  A candidate is accepted
    only when its projection lands substantially inside the image and overlaps
    no detection — otherwise it would not be a ghost at all.

    Empty tables are the canonical case for this class, and they are the
    dataset's largest stratum.
    """
    if not ctx.donor_parts:
        return InjectionResult(
            [dict(p) for p in parts_world], "ghost_part", 0.0,
            ok=False, reason="no_donor_parts",
        )

    iw, ih = ctx.image_wh
    n_donors = len(ctx.donor_parts)
    for _ in range(24):  # bounded retries; rejection is expected and fine
        donor = ctx.donor_parts[int(rng.integers(0, n_donors))]
        cand = dict(donor)
        # Jitter the donor position so we are not merely replaying another
        # frame's exact layout.
        cand["x"] = float(cand.get("x", 0.0)) + float(rng.normal(0.0, 150.0))
        cand["y"] = float(cand.get("y", 0.0)) + float(rng.normal(0.0, 150.0))
        cand["id"] = f"ghost_{rng.integers(1_000_000)}"
        polys = ctx.project([cand])
        if not polys:
            continue
        poly = polys[0]
        if not _polygon_in_image(poly, iw, ih):
            continue
        if _overlaps_any_detection(poly, ctx.detections):
            continue
        new_parts = [dict(p) for p in parts_world] + [cand]
        total_area = sum(_part_area(p) for p in new_parts) or 1.0
        return InjectionResult(
            parts_world=new_parts,
            injection_class="ghost_part",
            severity=round(_part_area(cand) / total_area, 4),
            meta={"n_before": len(parts_world), "ghost_id": cand["id"]},
            ops={"kind": "add_parts", "parts": [cand]},
        )

    return InjectionResult(
        [dict(p) for p in parts_world], "ghost_part", 0.0,
        ok=False, reason="no_valid_ghost_placement",
    )


def inject_position_shift_x_inbound(
    parts_world: list[dict], rng: np.random.Generator, ctx: InjectionContext
) -> InjectionResult:
    """Translate every part along world x by **less** than the offset bound.

    Expected behaviour: absorbed by the bounded x-offset correction, i.e. *not*
    flagged.  The rate at which it is absorbed is the correction's recovery
    rate; anything flagged here is a false error.  This is the "over-flexibility"
    probe the literature review explicitly asks for.
    """
    if not parts_world:
        return InjectionResult([], "position_shift_x_inbound", 0.0, ok=False, reason="no_parts")
    # Stay clearly inside the bound so the class tests the mechanism rather than
    # its boundary: 20%..80% of the search range.
    lo, hi = 0.2 * ctx.max_offset_mm, 0.8 * ctx.max_offset_mm
    shift = float(rng.uniform(lo, hi)) * (1.0 if rng.random() < 0.5 else -1.0)
    out = []
    for p in parts_world:
        q = dict(p)
        q["x"] = float(q.get("x", 0.0)) + shift
        out.append(q)
    return InjectionResult(
        parts_world=out,
        injection_class="position_shift_x_inbound",
        severity=round(abs(shift) / max(ctx.max_offset_mm, 1.0), 4),
        meta={"shift_mm": round(shift, 1), "axis": "x"},
        ops={"kind": "translate", "axis": "x", "mm": shift},
    )


def inject_position_shift_out(
    parts_world: list[dict], rng: np.random.Generator, ctx: InjectionContext
) -> InjectionResult:
    """Translate parts beyond the offset bound, or along **y**.

    The x-offset sweep only searches x, so a y-shift must be caught regardless
    of the bound — which makes this the sharpest discriminator between "the
    correction is doing bounded alignment" and "the correction explains
    everything".
    """
    if not parts_world:
        return InjectionResult([], "position_shift_out", 0.0, ok=False, reason="no_parts")
    along_y = rng.random() < 0.5
    if along_y:
        # A y shift of the same order as the x bound: unreachable by the sweep.
        shift = float(rng.uniform(0.3, 1.0)) * ctx.max_offset_mm
        axis = "y"
    else:
        # Strictly outside the search range.
        shift = float(rng.uniform(1.5, 3.0)) * ctx.max_offset_mm
        axis = "x"
    sign = 1.0 if rng.random() < 0.5 else -1.0
    out = []
    for p in parts_world:
        q = dict(p)
        q[axis] = float(q.get(axis, 0.0)) + shift * sign
        out.append(q)
    return InjectionResult(
        parts_world=out,
        injection_class="position_shift_out",
        severity=round(min(1.0, abs(shift) / max(ctx.max_offset_mm, 1.0)), 4),
        meta={"shift_mm": round(shift * sign, 1), "axis": axis},
        ops={"kind": "translate", "axis": axis, "mm": shift * sign},
    )


def inject_composition_swap(
    parts_world: list[dict], rng: np.random.Generator, ctx: InjectionContext
) -> InjectionResult:
    """Exchange the positions of two parts whose extents differ.

    Implemented **geometrically only**, per a scope
    correction: this system has no type channel (detection is single-class), so
    swapping equal-extent parts leaves the projected envelope unchanged and is
    *not* an inconsistency the matcher could or should flag.  Emitting such an
    instance as an alarm condition would put a mislabelled class into the
    study, which is worse than an honest zero elsewhere.

    A swap is therefore emitted only when it moves the world-space envelope by
    at least :data:`MIN_ENVELOPE_DELTA`; severity *is* that delta.  Partial
    recall is expected by construction, since the class degenerates into
    ``missing_part`` + ``ghost_part`` per group and the ghost half lands in the
    box-centric blind spot.
    """
    if len(parts_world) < 2:
        return InjectionResult(
            [dict(p) for p in parts_world], "composition_swap", 0.0,
            ok=False, reason="need_two_parts",
        )

    n = len(parts_world)
    order = list(rng.permutation(n))
    # Prefer the pair with the largest extent difference among a few samples.
    best: tuple[float, int, int] | None = None
    for a_i in range(min(n, 6)):
        for b_i in range(a_i + 1, min(n, 6)):
            a, b = order[a_i], order[b_i]
            ea = max(float(parts_world[a].get("sizeX", 0.0)), float(parts_world[a].get("sizeY", 0.0)))
            eb = max(float(parts_world[b].get("sizeX", 0.0)), float(parts_world[b].get("sizeY", 0.0)))
            diff = abs(ea - eb)
            if best is None or diff > best[0]:
                best = (diff, a, b)
    if best is None:
        return InjectionResult(
            [dict(p) for p in parts_world], "composition_swap", 0.0,
            ok=False, reason="no_pair",
        )

    _, ia, ib = best
    out = [dict(p) for p in parts_world]
    out[ia]["x"], out[ib]["x"] = out[ib].get("x", 0.0), out[ia].get("x", 0.0)
    out[ia]["y"], out[ib]["y"] = out[ib].get("y", 0.0), out[ia].get("y", 0.0)

    delta = _envelope_delta(parts_world, out)
    if delta < MIN_ENVELOPE_DELTA:
        return InjectionResult(
            [dict(p) for p in parts_world], "composition_swap", round(delta, 4),
            ok=False, reason="envelope_delta_below_min",
        )

    return InjectionResult(
        parts_world=out,
        injection_class="composition_swap",
        severity=round(delta, 4),
        meta={"swapped": [ia, ib], "envelope_delta": round(delta, 4)},
        ops={
            "kind": "swap_ids",
            "a": str(parts_world[ia].get("id", "")),
            "b": str(parts_world[ib].get("id", "")),
        },
    )


#: ``snapshot_desync`` is not a pure function of the current record — it needs a
#: *different run's* record — so the provider builds it directly (it owns the
#: run sequence).  See :meth:`pccv_bench.core.projection.RealProvider`.
INJECTORS: dict[str, Callable[[list[dict], np.random.Generator, InjectionContext], InjectionResult]] = {
    "clean": inject_clean,
    "missing_part": inject_missing_part,
    "ghost_part": inject_ghost_part,
    "position_shift_x_inbound": inject_position_shift_x_inbound,
    "position_shift_out": inject_position_shift_out,
    "composition_swap": inject_composition_swap,
}
