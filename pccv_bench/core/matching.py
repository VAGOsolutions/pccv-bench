"""Box-side matching, projection, and correction logic shared by the benchmark.

A record part is a world-space rectangle; this module projects it into the image
and scores how well a set of detections explains it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from pccv_bench.core.camera_model import CameraModelMapper
from pccv_bench.core.geometry import rect_corners_world


# ── table-A x offset ───────────────────────────────────────────────────────────
#: Fixed world-space x shift (mm) applied to table-A parts before projection: a
#: calibration constant of the modelled installation, not a tunable.
SAW_X_OFFSET: float = -450.5


# ── GQL projection constants ───────────────────────────────────────────────────
EDGE_SUBDIVISIONS: int = 8


CORRECTION_STEP_MM: float = 5.0


def correction_offsets_mm(bound_mm: float, two_sided: bool = False) -> list[float]:
    """World-space x offsets the correction search tries, in search order.

    One-sided (the production default): +5, +10, ..., up to *bound_mm*.
    Two-sided: +5, -5, +10, -10, ..., so the smallest displacement in either
    direction that explains an observation is found first.
    """
    n_steps = int(bound_mm // CORRECTION_STEP_MM) if bound_mm > 0 else 0
    offsets: list[float] = []
    for i in range(1, n_steps + 1):
        offsets.append(i * CORRECTION_STEP_MM)
        if two_sided:
            offsets.append(-i * CORRECTION_STEP_MM)
    return offsets


# ── Orange-pixel robot-arm detection (HSV, OpenCV ranges H 0-179) ───────────────
# Mirrors the production pipeline: images whose orange-pixel ratio exceeds the
# threshold are assumed to contain the robot arm and are excluded from GT-vs-GQL
# classification (the robot occludes/obscures the parts).
ORANGE_LOWER = np.array([5, 100, 100], dtype=np.uint8)


ORANGE_UPPER = np.array([25, 255, 255], dtype=np.uint8)


ORANGE_THRESHOLD: float = 0.002


def orange_ratio(img: np.ndarray) -> float:
    """Return the fraction of saturated-orange pixels in *img* (0.0–1.0)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    if mask.size == 0:
        return 0.0
    return float(cv2.countNonZero(mask)) / float(mask.size)


def subdivide_polygon(corners: np.ndarray, n: int) -> np.ndarray:
    """Subdivide each edge of a polygon into *n* segments."""
    if n <= 1:
        return corners
    pts: list[np.ndarray] = []
    for i in range(len(corners)):
        a, b = corners[i], corners[(i + 1) % len(corners)]
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            pts.append(a + t * (b - a))
    return np.array(pts, dtype=np.float64)


def project_parts_to_pixels_with_ids(
    parts: list[dict],
    mapper: CameraModelMapper,
    iw: int,
    ih: int,
    saw_x_offset: float = SAW_X_OFFSET,
) -> dict:
    """Like :func:`project_parts_to_pixels` but each entry retains the piece id.

    Returns::

        {
            "saw": [{"id": str, "polygon": [[x,y], ...]}, ...],
            "pla": [{"id": str, "polygon": [[x,y], ...]}, ...],
        }
    """
    result: dict[str, Any] = {"saw": [], "pla": []}
    for kind in ("saw", "pla"):
        for part in parts:
            if part.get("table_kind") != kind:
                continue
            x = float(part.get("x", 0.0))
            y = float(part.get("y", 0.0))
            sx = float(part.get("sizeX", 0.0))
            sy = float(part.get("sizeY", 0.0))
            r = float(part.get("rotation", part.get("r", 0.0)) or 0.0)
            if sx <= 0 or sy <= 0:
                continue
            if kind == "saw":
                x += saw_x_offset
            corners = rect_corners_world(x, y, sx, sy, r)
            pts = subdivide_polygon(corners, EDGE_SUBDIVISIONS)
            try:
                proj = mapper.project(pts)
            except Exception:
                continue
            proj[:, 0] = np.clip(proj[:, 0], 0, iw)
            proj[:, 1] = np.clip(proj[:, 1], 0, ih)
            result[kind].append({"id": str(part.get("id", "")), "polygon": proj.tolist()})
    return result


def rasterize_shapes(
    gql_polygons: list[np.ndarray],
    pred_polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Rasterize GQL and prediction polygons into aligned binary masks.

    Returns ``(mask_gql, mask_pred, origin_x, origin_y)``.
    """
    all_pts = np.concatenate(gql_polygons + [pred_polygon], axis=0)
    all_x = all_pts[:, 0]
    all_y = all_pts[:, 1]
    ox = max(0, int(all_x.min()) - 4)
    oy = max(0, int(all_y.min()) - 4)
    W = int(all_x.max()) - ox + 4
    H = int(all_y.max()) - oy + 4
    if W <= 0 or H <= 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return empty, empty, ox, oy

    mask_gql = np.zeros((H, W), dtype=np.uint8)
    for pts in gql_polygons:
        shifted = (pts - np.array([ox, oy])).astype(np.int32)
        cv2.fillPoly(mask_gql, [shifted], 255)

    mask_pred = np.zeros((H, W), dtype=np.uint8)
    shifted_pred = (pred_polygon - np.array([ox, oy])).astype(np.int32)
    cv2.fillPoly(mask_pred, [shifted_pred], 255)
    return mask_gql, mask_pred, ox, oy


def combo_metrics(
    gql_polys_subset: list[np.ndarray],
    pred_poly: np.ndarray,
) -> dict:
    """Score the union of *gql_polys_subset* against *pred_poly* for all metrics.

    Returns ``{"iou": float, "coverage": float, "containment": float}``.
    All values are 0.0 when the subset is empty or rasterisation fails.

    coverage   = intersection / GQL area    (how much of GQL is covered)
    containment = intersection / pred area  (how much of pred overlaps GQL)
    """
    zero = {"iou": 0.0, "coverage": 0.0, "containment": 0.0}
    if not gql_polys_subset or len(pred_poly) < 3:
        return zero
    try:
        mask_g, mask_s, _, _ = rasterize_shapes(gql_polys_subset, pred_poly)
        gql_px  = max(int(np.count_nonzero(mask_g)), 1)
        pred_px = max(int(np.count_nonzero(mask_s)), 1)
        inter   = int(np.count_nonzero(cv2.bitwise_and(mask_g, mask_s)))
        union   = int(np.count_nonzero(cv2.bitwise_or(mask_g, mask_s)))
        return {
            "iou":         round(inter / union   if union   > 0 else 0.0, 4),
            "coverage":    round(inter / gql_px,                          4),
            "containment": round(inter / pred_px,                         4),
        }
    except Exception:
        return zero


def bbox_to_polygon(bbox: list[float]) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` bbox to a 4-corner polygon array."""
    x1, y1, x2, y2 = bbox
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def pred_to_polygon(pred: dict) -> np.ndarray:
    """Convert a prediction dict to a polygon array.

    Supports both ``{"bbox": [x1,y1,x2,y2]}`` and ``{"polygon": [[x,y], ...]}``
    inputs.
    """
    if "polygon" in pred:
        return np.array(pred["polygon"], dtype=np.float64)
    return bbox_to_polygon(pred["bbox"])


def best_aabb_combo_for_box(
    gql_polys: list[np.ndarray],
    pred_poly: np.ndarray,
    metric: str,
) -> tuple[list[int], float, dict]:
    """Greedily find the subset of GQL polygons whose AABB best matches *pred_poly*.

    Starting from the empty set, repeatedly adds the GQL part whose inclusion
    yields the highest *metric* value when scored as the AABB of all selected
    parts combined.  A part is kept only when it strictly improves the current
    best.  Stops when no remaining part helps.

    Returns ``(selected_indices, best_value, best_metrics)``.
    """
    zero: dict = {"iou": 0.0, "coverage": 0.0, "containment": 0.0}
    selected: list[int] = []
    best = 0.0
    best_metrics: dict = zero
    remaining = list(range(len(gql_polys)))

    while remaining:
        best_candidate: int | None = None
        best_candidate_val = best
        best_candidate_metrics = best_metrics
        for idx in remaining:
            candidate_polys = [gql_polys[i] for i in selected] + [gql_polys[idx]]
            all_pts = np.concatenate(candidate_polys, axis=0)
            x1 = float(all_pts[:, 0].min())
            y1 = float(all_pts[:, 1].min())
            x2 = float(all_pts[:, 0].max())
            y2 = float(all_pts[:, 1].max())
            aabb = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
            try:
                m = combo_metrics([aabb], pred_poly)
            except Exception:
                continue
            val = m[metric]
            if val > best_candidate_val:
                best_candidate_val = val
                best_candidate = idx
                best_candidate_metrics = m
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best = best_candidate_val
        best_metrics = best_candidate_metrics

    return selected, best, best_metrics


def project_parts_polys_with_offset(
    gql_parts_with_ids: list[dict],
    offset_mm: float,
    mapper: CameraModelMapper,
    iw: int,
    ih: int,
    saw_x_offset: float = SAW_X_OFFSET,
) -> tuple[list[int], list[np.ndarray]]:
    """Re-project **every** GQL part with an added world-space x offset.

    Unlike :func:`reproject_parts_with_offset` (which envelopes a fixed subset
    into a single AABB), this returns the individual per-part pixel polygons for
    all parts so that the greedy subset match can be *re-run* at each offset.

    This is what lets a part that is fully off-frame at offset 0 (and therefore
    never selected by the base match) be picked up once the offset shifts it into
    view.

    Returns ``(orig_indices, polys)`` where ``orig_indices[k]`` is the index into
    *gql_parts_with_ids* of ``polys[k]``.  Parts that cannot be projected are
    omitted (so the two lists stay aligned and index-mappable).
    """
    orig_indices: list[int] = []
    polys: list[np.ndarray] = []
    for ci, part in enumerate(gql_parts_with_ids):
        world = part.get("world")
        if world is None:
            continue
        x = float(world.get("x", 0.0))
        y = float(world.get("y", 0.0))
        sx = float(world.get("sizeX", 0.0))
        sy = float(world.get("sizeY", 0.0))
        r = float(world.get("rotation", world.get("r", 0.0)) or 0.0)
        if sx <= 0 or sy <= 0:
            continue
        if world.get("table_kind") == "saw":
            x += saw_x_offset
        x += offset_mm
        corners = rect_corners_world(x, y, sx, sy, r)
        pts = subdivide_polygon(corners, EDGE_SUBDIVISIONS)
        try:
            proj = mapper.project(pts)
        except Exception:
            continue
        proj[:, 0] = np.clip(proj[:, 0], 0, iw)
        proj[:, 1] = np.clip(proj[:, 1], 0, ih)
        orig_indices.append(ci)
        polys.append(proj)
    return orig_indices, polys


def auto_match_boxes_to_parts(
    gql_parts_with_ids: list[dict],
    predictions: list[dict],
    metric: str,
    threshold: float,
    iw: int = 1920,
    ih: int = 1024,
    correction_offset_mm: float = 0.0,
    mapper: CameraModelMapper | None = None,
    next_run_gql_parts: list[dict] | None = None,
    saw_x_offset: float = SAW_X_OFFSET,
    two_sided_offset: bool = False,
) -> dict:
    """Per-prediction greedy AABB-envelope subset search with optional correction.

    *two_sided_offset* makes the x-offset correction also try negative offsets
    (see :func:`correction_offsets_mm`); False reproduces the production
    matcher's one-sided search exactly.

    For each prediction box, greedily selects the subset of GQL parts whose
    combined AABB envelope best matches the prediction under *metric*.  Parts
    are added one at a time; a part is kept only when it strictly improves the
    score.

    Only prediction boxes are evaluated — unmatched GQL parts are NOT errors
    (box-centric semantics).

    Correction strategies (applied in order when a match is below *threshold*):

    1. **X-offset sweep** (when *correction_offset_mm* > 0 and *mapper* is not
       None): re-projects the selected parts with increasing world-space x
       offsets (5 mm steps up to *correction_offset_mm*).  The first offset that
       meets *threshold* is recorded as ``is_corrected_error``.

    2. **Temporal ordering** (when *next_run_gql_parts* is not None): retries the
       full greedy match against the next sync-run's GQL projection.  Models the
       short delay between GQL snapshot capture and image capture.

    Parameters
    ----------
    gql_parts_with_ids:
        Output of :func:`build_gql_flat` for the current run_id.
    predictions:
        List of prediction dicts with either a ``"bbox"`` or ``"polygon"`` key.
    metric:
        One of ``"iou"``, ``"coverage"``, ``"containment"``.
    threshold:
        Minimum metric value for a match to be considered correct.
    iw, ih:
        Image dimensions for clipping during x-offset re-projection.
    correction_offset_mm:
        Maximum world-space x offset (mm) to try during sweep correction.
        0 = disabled.
    mapper:
        Camera model required for x-offset re-projection.  Must be supplied
        when *correction_offset_mm* > 0.
    next_run_gql_parts:
        GQL flat parts from the **next** sync-run for the same camera, already
        projected to pixel space.  Pass ``None`` to disable temporal correction.
    saw_x_offset:
        Fixed world-space x shift applied to SAW parts before projection.

    Returns
    -------
    dict with keys:

    - ``groups``: one entry per prediction box.
    - ``unmatched_boxes``: indices of boxes where ``is_error`` is true.
    - ``corrected_boxes``: indices of boxes where ``is_corrected_error`` is true.
    - ``has_errors``: true when any box is a true error.
    - ``metric_used``, ``threshold``, ``correction_offset_mm``.
    """
    zero: dict = {"iou": 0.0, "coverage": 0.0, "containment": 0.0}
    gql_polys = [np.array(p["polygon"], dtype=np.float64) for p in gql_parts_with_ids]

    if not predictions:
        return {
            "groups": [],
            "unmatched_boxes": [],
            "corrected_boxes": [],
            "has_errors": False,
            "metric_used": metric,
            "threshold": threshold,
            "correction_offset_mm": correction_offset_mm,
        }

    groups: list[dict] = []

    for bi, pred in enumerate(predictions):
        pred_poly = pred_to_polygon(pred)

        if not gql_polys:
            groups.append({
                "box_index": bi,
                "parts": [],
                "best_value": 0.0,
                "metrics": zero,
                "is_error": True,
                "is_corrected_error": False,
                "corrected_offset_mm": 0.0,
                "corrected_polygons": [],
                "correction_method": None,
            })
            continue

        selected_indices, best_val, best_metrics = best_aabb_combo_for_box(
            gql_polys, pred_poly, metric
        )

        is_error = best_val < threshold
        is_corrected_error = False
        corrected_offset: float = 0.0
        corrected_polygons: list[list[list[float]]] = []
        correction_method: str | None = None
        final_selected = selected_indices

        # ── 1. X-offset sweep correction ──────────────────────────────────────
        # Re-run the greedy match over ALL parts at each offset step (not just
        # the parts selected at offset 0).  This lets parts that are clipped
        # off-frame at offset 0 — and therefore never selected by the base
        # match — be picked up once the offset shifts them into view.
        if (
            is_error
            and correction_offset_mm > 0.0
            and mapper is not None
            and gql_parts_with_ids
        ):
            for trial_mm in correction_offsets_mm(correction_offset_mm, two_sided_offset):
                orig_idx, off_polys = project_parts_polys_with_offset(
                    gql_parts_with_ids, trial_mm, mapper, iw, ih, saw_x_offset,
                )
                if off_polys:
                    sel, retry_val, retry_metrics = best_aabb_combo_for_box(
                        off_polys, pred_poly, metric
                    )
                    if retry_val >= threshold:
                        is_error = False
                        is_corrected_error = True
                        corrected_offset = trial_mm
                        corrected_polygons = [off_polys[s].tolist() for s in sel]
                        best_val = retry_val
                        best_metrics = retry_metrics
                        correction_method = "x_offset"
                        final_selected = [orig_idx[s] for s in sel]
                        break

        # ── 2. Temporal ordering correction ───────────────────────────────────
        if is_error and next_run_gql_parts is not None:
            next_polys = [np.array(p["polygon"], dtype=np.float64) for p in next_run_gql_parts]
            if next_polys:
                next_indices, next_val, next_metrics = best_aabb_combo_for_box(
                    next_polys, pred_poly, metric
                )
                if next_val >= threshold:
                    is_error = False
                    is_corrected_error = True
                    best_val = next_val
                    best_metrics = next_metrics
                    correction_method = "temporal"
                    final_selected = next_indices

        # ── Build parts list ──────────────────────────────────────────────────
        source_parts = (
            next_run_gql_parts if correction_method == "temporal" else gql_parts_with_ids
        )
        parts: list[dict] = []
        for gi in final_selected:
            if gi < len(source_parts):
                part = source_parts[gi]
                parts.append({
                    "part_id": part["id"],
                    "part_kind": part["kind"],
                    "gql_index": gi,
                    "gql_kind_index": part.get("kind_index", gi),
                })

        groups.append({
            "box_index": bi,
            "parts": parts,
            "best_value": round(best_val, 4),
            "metrics": best_metrics,
            "is_error": is_error,
            "is_corrected_error": is_corrected_error,
            "corrected_offset_mm": corrected_offset,
            "corrected_polygons": corrected_polygons,
            "correction_method": correction_method,
        })

    unmatched_boxes = [g["box_index"] for g in groups if g["is_error"]]
    corrected_boxes = [g["box_index"] for g in groups if g["is_corrected_error"]]
    # Image is an error when ANY box is unmatched (every prediction/annotation
    # must be explained by GQL after corrections).
    has_errors = bool(unmatched_boxes)

    return {
        "groups": groups,
        "unmatched_boxes": unmatched_boxes,
        "corrected_boxes": corrected_boxes,
        "has_errors": has_errors,
        "metric_used": metric,
        "threshold": threshold,
        "correction_offset_mm": correction_offset_mm,
    }
