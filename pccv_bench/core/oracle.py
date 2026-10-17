"""Is an injected perturbation an *observable* inconsistency?

Changing the record is not the same as creating a fault a camera could have
caught, and the study was scoring the difference as matcher error:

- ``inject_missing_part`` deleted recorded parts without checking that the
  deleted part overlapped anything that was actually observed.  Removing a part
  the camera never saw leaves nothing visible unexplained, so passing such a
  pair is *correct* behaviour being counted as a miss.
- the desynchronization generator rejected two empty records but not two
  non-empty records with the same visible geometry — a donor that looks
  identical is not a desynchronization.
- ``composition_swap`` decided severity from the **world-space** envelope, which
  can move while every projected group stays exactly where it was.

Independence
------------
Validity is decided from projected geometry alone.  Nothing here calls
:func:`~pccv_bench.core.matching.auto_match_boxes_to_parts` or
:func:`~pccv_bench.core.twoway.record_side_unmatched`: a perturbation whose
ground truth came from the method under evaluation is precisely the circularity
that an independent review objects to.  The rasterisation primitives are shared (polygon algebra is not
a decision rule) and so is
:func:`~pccv_bench.core.twoway.is_part_visible` — forking the definition of
"visible" would let the oracle and the channel it validates disagree about which
parts a camera can even see, which is worse than the coupling."""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from pccv_bench.core.inject import expected_action_for
from pccv_bench.core.twoway import is_part_visible, record_part_coverage


#: Visible footprints at least this similar before and after count as unchanged.
#: Not 1.0: rasterisation is integer-pixel, so a genuinely identical layout can
#: score a shade under unity.
NO_CHANGE_IOU: float = 0.98


#: A deleted record part must have covered at least this fraction of itself with
#: real observation to have been observable at all.  Deliberately far below any
#: matching threshold — the question is "was there anything there to miss", not
#: "would the matcher have matched it".
MIN_DELETED_PART_COVERAGE: float = 0.10


#: A ghost part overlapping an observation by more than this is explainable by
#: that detection, so it is not a ghost.
MAX_GHOST_OVERLAP: float = 0.05


#: Per-part movement below this IoU counts as a visibly moved group.
SWAP_MOVED_IOU: float = 0.9


@dataclass(frozen=True)
class InjectionValidity:
    """Why a perturbation may or may not be scored.

    ``expected_action`` is repeated from the class taxonomy so a caller reading
    a rejection never has to look up what the instance would have demanded.
    """

    is_observable: bool
    changed_visible_geometry: bool
    expected_action: str
    rejection_reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.rejection_reason


def _polys(parts: list[dict], image_wh: tuple[int, int] | None) -> list[np.ndarray]:
    """Projected polygons of the parts a camera can actually see."""
    out: list[np.ndarray] = []
    for p in parts:
        poly = p.get("polygon")
        if poly is None or not is_part_visible(poly, image_wh):
            continue
        arr = np.asarray(poly, dtype=np.float64)
        if len(arr) >= 3:
            out.append(arr)
    return out


def footprint_iou(before: list[np.ndarray], after: list[np.ndarray]) -> float:
    """IoU of the two visible footprints, as unions of polygon sets.

    1.0 when both are empty: nothing visible either way is, correctly, no
    visible change.
    """
    if not before and not after:
        return 1.0
    if not before or not after:
        return 0.0
    all_pts = np.concatenate(before + after, axis=0)
    ox = max(0, int(all_pts[:, 0].min()) - 4)
    oy = max(0, int(all_pts[:, 1].min()) - 4)
    w = int(all_pts[:, 0].max()) - ox + 4
    h = int(all_pts[:, 1].max()) - oy + 4
    if w <= 0 or h <= 0:
        return 1.0
    mask_b = np.zeros((h, w), dtype=np.uint8)
    mask_a = np.zeros((h, w), dtype=np.uint8)
    for polys, mask in ((before, mask_b), (after, mask_a)):
        for poly in polys:
            cv2.fillPoly(mask, [(poly - np.array([ox, oy])).astype(np.int32)], 255)
    inter = int(np.count_nonzero(cv2.bitwise_and(mask_b, mask_a)))
    union = int(np.count_nonzero(cv2.bitwise_or(mask_b, mask_a)))
    return float(inter / union) if union else 1.0


def _by_id(parts: list[dict]) -> dict[str, dict]:
    return {str(p.get("id", "")): p for p in parts}


def validate_injection(
    before: list[dict],
    after: list[dict],
    observations: list[dict],
    camera: str,
    injection_class: str,
    image_wh: tuple[int, int] | None = None,
) -> InjectionValidity:
    """Decide whether this perturbation is scorable, and why not if it is not.

    *before* and *after* are the record parts **already projected into pixel
    space** — the same flat form the providers build — because observability is
    a property of the image, not of the world coordinates.  (The plan's
    four-argument sketch omitted *injection_class* and *image_wh*; neither
    "which perturbation is this" nor "what can this camera see" is recoverable
    from the geometry alone.)
    """
    action = expected_action_for(injection_class)
    vis_before = _polys(before, image_wh)
    vis_after = _polys(after, image_wh)
    iou = footprint_iou(vis_before, vis_after)
    changed = len(vis_before) != len(vis_after) or iou < NO_CHANGE_IOU

    if injection_class == "clean":
        # The control is defined as an untouched record; it has nothing to
        # observe and nothing to reject.
        return InjectionValidity(
            is_observable=True, changed_visible_geometry=False, expected_action=action
        )

    def reject(reason: str) -> InjectionValidity:
        return InjectionValidity(
            is_observable=False,
            changed_visible_geometry=changed,
            expected_action=action,
            rejection_reason=reason,
        )

    if not changed:
        return reject("no_visible_change")

    before_by_id, after_by_id = _by_id(before), _by_id(after)

    removed = [p for pid, p in before_by_id.items() if pid not in after_by_id]
    if removed:
        # Only a removal that took away *observed* evidence leaves something
        # visible unexplained.
        observable_removals = [
            p
            for p in removed
            if is_part_visible(p.get("polygon"), image_wh)
            and record_part_coverage(p["polygon"], observations) >= MIN_DELETED_PART_COVERAGE
        ]
        if not observable_removals:
            return reject("deleted_part_unobserved")

    added = [p for pid, p in after_by_id.items() if pid not in before_by_id]
    if added and injection_class == "ghost_part":
        placed = [
            p
            for p in added
            if is_part_visible(p.get("polygon"), image_wh)
            and record_part_coverage(p["polygon"], observations) <= MAX_GHOST_OVERLAP
        ]
        if not placed:
            return reject("ghost_overlaps_observation")

    if injection_class == "composition_swap":
        # Per affected group in projected space, not the global world envelope —
        # and as a *set*, not part by part.  Detection is single-class, so a part
        # sitting exactly where another part was is observationally identical to
        # the original scene however far each individual part moved.  Asking
        # whether each part moved would accept precisely the swaps no detector
        # could express.
        touched = [
            pid
            for pid, p in before_by_id.items()
            if pid in after_by_id
            and footprint_iou(
                _polys([p], image_wh), _polys([after_by_id[pid]], image_wh)
            )
            < NO_CHANGE_IOU
        ]
        if footprint_iou(
            _polys([before_by_id[pid] for pid in touched], image_wh),
            _polys([after_by_id[pid] for pid in touched], image_wh),
        ) >= SWAP_MOVED_IOU:
            return reject("swap_invisible_in_projection")

    return InjectionValidity(
        is_observable=True, changed_visible_geometry=True, expected_action=action
    )
