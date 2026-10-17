"""Simple record-aware alternatives the proposed method must beat.

The saved ablation ladder went geometry → +offset → +temporal → both → both plus
two-way.  It never established whether plain **geometry + two-way** already
suffices, and it never compared against anything simpler than the greedy subset
matcher.  Near-perfect ghost recall over a baseline that is *structurally unable*
to flag ghosts demonstrates added functionality, not method superiority — so the
comparison needs alternatives that can express the same errors.

Four are implemented here, each deliberately cruder than the production matcher:

``envelope``
    One test on the whole scene: does the union of expected parts and the union
    of observations explain each other?  It matters because with
    *containment* alone, enlarging the expected envelope can never reduce
    containment — so a method that scores a subset envelope has a built-in
    advantage that only a whole-envelope baseline exposes.

``symmetric``
    Per element, both directions, no *chosen-subset* search and no corrections:
    every visible expected part must be covered by the union of all
    observations, and every observation must be covered by the union of all
    expected parts.  This is the obvious thing a practitioner would write
    first — but unioning the whole side is still a combination, just not a
    searched one; see ``pairwise`` for the baseline that has none at all.

``coherent``
    A coherence objection made into a method.  The production matcher
    corrects each detection independently, so different detections may be
    explained by different offsets or different snapshots — "each object has
    *some* explanation" is weaker than "*one* corrected scene explains the
    capture".  This variant searches the same offset grid and the same two
    snapshots but must pick a **single** (offset, snapshot) hypothesis for the
    whole frame.

``pairwise``
    The true no-combination floor: every check is one part against one
    detection, never a union of either side.  ``symmetric``'s det-side check
    unions *every* visible part in the scene — the most permissive possible
    combination, not an absence of one — so it cannot isolate "no subset
    search" from "unrestricted combination".  ``pairwise`` can.

All four take the same visibility guard as the record-side channel: a method is
never asked to account for geometry outside the camera's field of view."""

from __future__ import annotations

import numpy as np
from pccv_bench.core.camera_model import CameraModelMapper
from pccv_bench.core.matching import SAW_X_OFFSET, combo_metrics, correction_offsets_mm, pred_to_polygon, project_parts_polys_with_offset
from pccv_bench.core.twoway import is_part_visible, record_part_coverage


def _visible_part_polys(
    record_parts: list[dict], image_wh: tuple[int, int] | None
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in record_parts:
        poly = p.get("polygon")
        if poly is None or not is_part_visible(poly, image_wh):
            continue
        arr = np.asarray(poly, dtype=np.float64)
        if len(arr) >= 3:
            out.append(arr)
    return out


def _detection_polys(detections: list[dict]) -> list[np.ndarray]:
    polys = [pred_to_polygon(d) for d in detections]
    return [p for p in polys if len(p) >= 3]


def envelope_flag(
    record_parts: list[dict],
    detections: list[dict],
    threshold: float = 0.3,
    image_wh: tuple[int, int] | None = None,
) -> bool:
    """Flag unless the two whole-scene envelopes explain each other.

    One IoU test over everything: no per-part correspondence, no subset search.
    """
    parts = _visible_part_polys(record_parts, image_wh)
    dets = _detection_polys(detections)
    if not parts and not dets:
        return False
    if not parts or not dets:
        return True
    m = combo_metrics(parts, _union_hull(dets))
    return m["iou"] < threshold


def _union_hull(polys: list[np.ndarray]) -> np.ndarray:
    """Convex hull of every point, standing in for the observed envelope.

    ``combo_metrics`` scores a set against one polygon, so the observation side
    is collapsed to a single shape.  A hull, not a bounding box: an axis-aligned
    box around two diagonally separated groups would claim most of the table.
    """
    import cv2

    pts = np.concatenate(polys, axis=0).astype(np.float32)
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2).astype(np.float64)


def symmetric_flag(
    record_parts: list[dict],
    detections: list[dict],
    threshold: float = 0.3,
    image_wh: tuple[int, int] | None = None,
    metric: str = "containment",
    record_threshold: float | None = None,
) -> bool:
    """Flag when either direction leaves an element unexplained.

    Both directions union the whole opposite side — no *chosen*-subset search,
    which is exactly the capability being isolated — scored under ``metric``.
    *record_threshold* gives the record-side (part-coverage) test its own
    threshold; None uses *threshold* for both directions, as originally.
    """
    part_threshold = threshold if record_threshold is None else record_threshold
    parts = _visible_part_polys(record_parts, image_wh)
    dets = _detection_polys(detections)
    if not parts and not dets:
        return False
    if bool(parts) != bool(dets):
        return True
    for poly in parts:
        if record_part_coverage(poly, detections, metric=metric) < part_threshold:
            return True
    for det in dets:
        if combo_metrics(parts, det)[metric] < threshold:
            return True
    return False


def coherent_flag(
    record_parts: list[dict],
    detections: list[dict],
    threshold: float = 0.3,
    image_wh: tuple[int, int] | None = None,
    metric: str = "containment",
    correction_offset_mm: float = 0.0,
    mapper: CameraModelMapper | None = None,
    next_record_parts: list[dict] | None = None,
    saw_x_offset: float = SAW_X_OFFSET,
    record_threshold: float | None = None,
    two_sided_offset: bool = False,
) -> bool:
    """Flag unless one single corrected scene explains the whole capture.

    Searches the same offset grid and the same two snapshots the production
    matcher does, but a candidate only counts when it explains **every**
    observation and **every** visible expected part at once, scored under
    ``metric``.  Per-detection correction — the production behaviour — can
    assemble an explanation out of mutually incompatible pieces; this cannot.
    """
    candidates: list[list[dict]] = [record_parts]
    if next_record_parts is not None:
        candidates.append(next_record_parts)

    for parts in candidates:
        if not symmetric_flag(
            parts, detections, threshold, image_wh, metric=metric, record_threshold=record_threshold
        ):
            return False
        if correction_offset_mm <= 0.0 or mapper is None or image_wh is None:
            continue
        iw, ih = image_wh
        for trial_mm in correction_offsets_mm(correction_offset_mm, two_sided_offset):
            orig_idx, off_polys = project_parts_polys_with_offset(
                parts, trial_mm, mapper, iw, ih, saw_x_offset
            )
            shifted = [
                {**parts[i], "polygon": poly.tolist()}
                for i, poly in zip(orig_idx, off_polys)
            ]
            if not symmetric_flag(
                shifted, detections, threshold, image_wh, metric=metric, record_threshold=record_threshold
            ):
                return False
    return True


def coherent_passes(scores: list[tuple[float, float]], obs_threshold: float, record_threshold: float) -> bool:
    """Decision of :func:`coherent_flag` (negated) from precomputed scores."""
    return any(rec >= record_threshold and obs >= obs_threshold for rec, obs in scores)


def pairwise_flag(
    record_parts: list[dict],
    detections: list[dict],
    threshold: float = 0.3,
    image_wh: tuple[int, int] | None = None,
    metric: str = "containment",
) -> bool:
    """Flag when no single detection/part pair explains an element.

    The true no-combination floor: unlike ``symmetric_flag``, neither side is
    ever unioned — each part is checked against one detection at a time, and
    each detection against one part at a time, so a scene where only a
    *union* explains an element is correctly reported as unexplained.
    """
    parts = _visible_part_polys(record_parts, image_wh)
    dets = _detection_polys(detections)
    if not parts and not dets:
        return False
    if bool(parts) != bool(dets):
        return True
    for part in parts:
        if not any(combo_metrics([det], part)[metric] >= threshold for det in dets):
            return True
    for det in dets:
        if not any(combo_metrics([part], det)[metric] >= threshold for part in parts):
            return True
    return False


def _assignment_scores(
    record_parts: list[dict],
    detections: list[dict],
    image_wh: tuple[int, int] | None,
    metric: str,
) -> tuple[float, float]:
    """(weakest assigned-part coverage, weakest assigned-observation score).

    Same semantics as :func:`_symmetric_scores`: an unassigned element scores
    0, an empty side +inf, a one-sided empty scene -inf.  The assignment itself
    maximises total observation score and does not depend on the thresholds.
    """
    from scipy.optimize import linear_sum_assignment

    parts = _visible_part_polys(record_parts, image_wh)
    det_dicts = [d for d in detections if len(pred_to_polygon(d)) >= 3]
    dets = [pred_to_polygon(d) for d in det_dicts]
    if not parts and not dets:
        return float("inf"), float("inf")
    if bool(parts) != bool(dets):
        return float("-inf"), float("-inf")
    scores = np.array([[combo_metrics([part], det)[metric] for part in parts] for det in dets])
    det_idx, part_idx = linear_sum_assignment(scores, maximize=True)
    obs_scores = np.zeros(len(dets))
    part_scores = np.zeros(len(parts))
    for d, p in zip(det_idx, part_idx):
        obs_scores[d] = scores[d, p]
        part_scores[p] = record_part_coverage(parts[p], [det_dicts[d]])
    return float(part_scores.min()), float(obs_scores.min())


def assignment_hypothesis_scores(
    record_parts: list[dict],
    detections: list[dict],
    image_wh: tuple[int, int] | None = None,
    metric: str = "iou",
    correction_offset_mm: float = 0.0,
    mapper: CameraModelMapper | None = None,
    next_record_parts: list[dict] | None = None,
    saw_x_offset: float = SAW_X_OFFSET,
    two_sided_offset: bool = False,
) -> list[tuple[float, float]]:
    """Scores of every (snapshot, offset) hypothesis, as
    :func:`coherent_hypothesis_scores` but with one-to-one assignment."""
    candidates: list[list[dict]] = [record_parts]
    if next_record_parts is not None:
        candidates.append(next_record_parts)
    scores: list[tuple[float, float]] = []
    for parts in candidates:
        scores.append(_assignment_scores(parts, detections, image_wh, metric))
        if correction_offset_mm <= 0.0 or mapper is None or image_wh is None:
            continue
        iw, ih = image_wh
        for trial_mm in correction_offsets_mm(correction_offset_mm, two_sided_offset):
            orig_idx, off_polys = project_parts_polys_with_offset(
                parts, trial_mm, mapper, iw, ih, saw_x_offset
            )
            shifted = [
                {**parts[i], "polygon": poly.tolist()}
                for i, poly in zip(orig_idx, off_polys)
            ]
            scores.append(_assignment_scores(shifted, detections, image_wh, metric))
    return scores


def assignment_flag(
    record_parts: list[dict],
    detections: list[dict],
    threshold: float = 0.3,
    image_wh: tuple[int, int] | None = None,
    metric: str = "iou",
    correction_offset_mm: float = 0.0,
    mapper: CameraModelMapper | None = None,
    next_record_parts: list[dict] | None = None,
    record_threshold: float | None = None,
    two_sided_offset: bool = False,
) -> bool:
    """Flag unless one hypothesis's one-to-one assignment explains every
    observation (score >= *threshold*) and covers every visible part
    (>= *record_threshold*, default *threshold*)."""
    scores = assignment_hypothesis_scores(
        record_parts, detections, image_wh, metric=metric,
        correction_offset_mm=correction_offset_mm, mapper=mapper,
        next_record_parts=next_record_parts, two_sided_offset=two_sided_offset,
    )
    part_threshold = threshold if record_threshold is None else record_threshold
    return not coherent_passes(scores, threshold, part_threshold)
