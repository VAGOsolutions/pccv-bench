"""Two-way matching: the record-side coverage channel.

Why this exists
---------------
:func:`~pccv_bench.core.matching.auto_match_boxes_to_parts` is
*box-centric*: it scores prediction boxes only, and unmatched record parts are
explicitly not errors (``matching.py`` L471-473).  Under the system's
single-class scope the expressible error space is exhaustively
presence / count / position / extent, so "the record says a part is there and
nothing was detected" is a structural *half* of that space the matcher is blind
to — and that blind direction sits on the dataset's largest stratum (empty
tables, where a non-empty record is a ghost-part error by
definition).

What this module adds
---------------------
For each projected expected part, require that some detection explain it.  The
verdict is reported as a **separate channel** from the box-side verdict so the
ablation can attribute each direction independently.

Geometry reuse
--------------
The design requires reusing the existing metric helpers rather than forking
the geometry.  We do, via one observation about
:func:`~pccv_bench.core.matching.combo_metrics`, whose signature is
``combo_metrics(gql_polys_subset, pred_poly)`` and which returns

    coverage    = intersection / area(union of gql_polys_subset)
    containment = intersection / area(pred_poly)

Calling it with the roles **swapped** — the detection polygons as the "subset"
and the record part as the "pred" — makes ``containment`` exactly the quantity
we want: the fraction of the expected part covered by the union of detections.
Using the union (rather than a max over individual detections) matters because a
long part legitimately spans two adjacent detection boxes; scoring per-detection
would under-report its coverage and inflate the false-error rate, which is the
one column to watch."""

from __future__ import annotations

import numpy as np
from pccv_bench.core.camera_model import CameraModelMapper
from pccv_bench.core.matching import SAW_X_OFFSET, combo_metrics, correction_offsets_mm, pred_to_polygon, project_parts_polys_with_offset


def record_part_coverage(
    part_polygon: list[list[float]] | np.ndarray,
    detections: list[dict],
    metric: str = "containment",
) -> float:
    """Return *part_polygon*'s score against the union of *detections*.

    0.0 when there are no detections (an expected part with nothing observed
    anywhere is entirely unexplained) or when the polygon is degenerate.
    ``metric`` selects which of ``combo_metrics``'s keys to read; the default
    ``containment`` is the fraction of *part_polygon* covered by the union.
    """
    part_poly = np.asarray(part_polygon, dtype=np.float64)
    if len(part_poly) < 3:
        return 0.0
    if not detections:
        return 0.0
    det_polys = [pred_to_polygon(d) for d in detections]
    det_polys = [p for p in det_polys if len(p) >= 3]
    if not det_polys:
        return 0.0
    # Roles swapped on purpose — see the module docstring.  ``containment`` is
    # intersection / area(part_poly).
    return float(combo_metrics(det_polys, part_poly)[metric])


#: A projected part must be at least this wide and tall (pixels) to be treated
#: as visible.  The projection helpers *clip* coordinates to the image, so a
#: part outside the field of view collapses into a degenerate sliver along a
#: border rather than disappearing.
MIN_VISIBLE_PX: float = 3.0


#: …and at least this fraction of its own bounding box must fall inside the
#: image, so a part straddling the edge is judged on what is actually in frame.
MIN_VISIBLE_FRACTION: float = 0.5


def is_part_visible(
    part_polygon: list[list[float]] | np.ndarray,
    image_wh: tuple[int, int] | None,
) -> bool:
    """Whether an expected part is actually observable in this frame.

    Two-way matching must not demand a detection for a part that lies outside
    the camera's field of view — the record legitimately describes the whole
    table, of which each camera sees a portion.  Because
    :func:`~pccv_bench.core.matching.project_parts_to_pixels_with_ids`
    clips projected coordinates to the image bounds, an out-of-view part does
    not vanish; it degenerates into a one-pixel sliver pinned to a border, whose
    coverage is structurally zero and can never be explained by any detection.

    Skipping the guard is not a small effect: measured on 890 record parts from
    120 clean frames, **33.6%** of projected parts were such slivers while only
    12.5% were genuinely uncovered — so the unguarded channel would have
    reported a false-error rate driven mostly by geometry it was never entitled
    to check.
    """
    arr = np.asarray(part_polygon, dtype=np.float64)
    if len(arr) < 3:
        return False
    x1, y1 = float(arr[:, 0].min()), float(arr[:, 1].min())
    x2, y2 = float(arr[:, 0].max()), float(arr[:, 1].max())
    w, h = x2 - x1, y2 - y1
    if w < MIN_VISIBLE_PX or h < MIN_VISIBLE_PX:
        return False
    if image_wh is None:
        return True
    iw, ih = image_wh
    inside_w = min(x2, iw) - max(x1, 0.0)
    inside_h = min(y2, ih) - max(y1, 0.0)
    if inside_w <= 0 or inside_h <= 0:
        return False
    return (inside_w * inside_h) / (w * h) >= MIN_VISIBLE_FRACTION


def record_side_unmatched(
    record_parts: list[dict],
    detections: list[dict],
    threshold: float = 0.3,
    image_wh: tuple[int, int] | None = None,
    correction_offset_mm: float = 0.0,
    mapper: CameraModelMapper | None = None,
    saw_x_offset: float = SAW_X_OFFSET,
    two_sided_offset: bool = False,
) -> list[int]:
    """Indices of *visible* expected parts that no detection explains.

    Parts judged not visible by :func:`is_part_visible` are skipped entirely —
    neither matched nor unmatched — because requiring a detection for them would
    penalise the record for geometry outside this camera's view.

    Parameters
    ----------
    record_parts:
        Flat record parts as produced by
        :func:`~pccv_bench.core.matching.build_gql_flat` (each with a
        ``polygon`` in pixel space, and — required only when
        ``correction_offset_mm > 0`` — a ``world`` entry).
    detections:
        Observation-side dicts with a ``bbox`` or ``polygon`` key.
    threshold:
        Minimum covered fraction for a visible expected part to count as
        observed.  Swept in the paper to give the benefit-vs-false-error
        trade-off curve.
    image_wh:
        Frame size, used for the visibility guard.  When ``None`` only the
        degenerate-sliver test applies.
    correction_offset_mm, mapper:
        Give this channel the same bounded x-offset tolerance the box-side
        matcher already has (:func:`~pccv_bench.core.matching.auto_match_boxes_to_parts`).
        Its absence was a measured limitation: on ``position_shift_x_inbound`` —
        a class the box side is *supposed* to absorb as ordinary camera-alignment
        wobble — the unaware record-side channel independently re-flagged 58.3%
        of those same instances, because it compared raw projected coverage with
        no allowance for exactly the drift the offset mechanism exists to excuse.
        When ``correction_offset_mm > 0`` and *mapper* is given, a part that
        fails the base check is retried at increasing world-space x offsets (the
        same ``CORRECTION_STEP_MM`` grid the box side sweeps, up to
        *correction_offset_mm*) before being reported as unmatched. 0 (the
        default) reproduces the original offset-blind behaviour exactly.
    two_sided_offset:
        Also try negative offsets (:func:`~pccv_bench.core.matching.correction_offsets_mm`).
    """
    candidates: list[int] = []
    for i, part in enumerate(record_parts):
        poly = part.get("polygon")
        if poly is None:
            continue
        if not is_part_visible(poly, image_wh):
            continue
        if record_part_coverage(poly, detections) < threshold:
            candidates.append(i)

    if not candidates or correction_offset_mm <= 0.0 or mapper is None or image_wh is None:
        return candidates

    iw, ih = image_wh
    remaining = set(candidates)
    rescued: set[int] = set()
    for trial_mm in correction_offsets_mm(correction_offset_mm, two_sided_offset):
        if not remaining:
            break
        orig_idx, off_polys = project_parts_polys_with_offset(
            record_parts, trial_mm, mapper, iw, ih, saw_x_offset,
        )
        poly_by_orig = dict(zip(orig_idx, off_polys))
        for i in list(remaining):
            poly = poly_by_orig.get(i)
            if poly is None:
                continue
            if record_part_coverage(poly.tolist(), detections) >= threshold:
                remaining.discard(i)
                rescued.add(i)

    return [i for i in candidates if i not in rescued]
