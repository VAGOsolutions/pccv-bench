"""Synthetic detection model.

Truth boxes come from projecting each scene group's parts through a fitted
synthetic camera; this module degrades them into ``predictions`` the same
shape :func:`~pccv_bench.core.matching.pred_to_polygon` expects
(``{"bbox": [x1,y1,x2,y2], "score": s}``, absolute crop pixels, 1920x1024
canvas), via the noise mechanisms in :mod:`pccv_bench.noise`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.core.camera_model import CameraModelMapper

from pccv_bench.core.projection import project_world_parts
from pccv_bench.noise import NoiseParams
from pccv_bench.scene import Scene

BBox = tuple[float, float, float, float]

#: A group whose projected AABB is smaller than this on either axis has
#: clipped to a degenerate sliver at the frame border (cameras.py's synthetic
#: geometry puts ~90% of the table in-frame, not 100% — see that module's
#: docstring) rather than being genuinely visible. Treated as "not observed
#: this frame" — the same real phenomenon
#: :func:`~pccv_bench.core.twoway.is_part_visible` guards against on real data
#: — instead of emitting a spurious sliver detection for it.
MIN_VISIBLE_BOX_PX = 4.0


def truth_boxes_per_group(
    scene: Scene, mapper: CameraModelMapper, iw: int, ih: int
) -> dict[int, BBox]:
    """AABB per *visible* scene group, from projecting that group's parts.

    Groups with multiple parts (the "parts-per-group" draw) get
    the union AABB of their parts' projected polygons — the same envelope the
    matcher's ``best_aabb_combo_for_box`` would assemble from the matching
    record parts, so a group with 1 part behaves identically to the earlier
    ``pred_to_polygon`` single-box case and a group with >1 part exercises the
    multi-part combination path. Groups that clip to a degenerate
    near-frame-border sliver (see :data:`MIN_VISIBLE_BOX_PX`) are omitted.
    """
    boxes: dict[int, BBox] = {}
    for gid in scene.group_ids():
        parts = scene.parts_in_group(gid)
        world_dicts = [p.to_world_dict() for p in parts]
        flat = project_world_parts(world_dicts, mapper, iw, ih)
        if not flat:
            continue
        pts = np.concatenate([np.asarray(f["polygon"], dtype=np.float64) for f in flat], axis=0)
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        if (x2 - x1) < MIN_VISIBLE_BOX_PX or (y2 - y1) < MIN_VISIBLE_BOX_PX:
            continue
        boxes[gid] = (x1, y1, x2, y2)
    return boxes


def _jitter_box(rng: np.random.Generator, box: BBox, sigma: float, iw: int, ih: int) -> BBox:
    x1, y1, x2, y2 = box
    x1 += rng.normal(0.0, sigma)
    y1 += rng.normal(0.0, sigma)
    x2 += rng.normal(0.0, sigma)
    y2 += rng.normal(0.0, sigma)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    # A degenerate (near-zero-area) box after jitter is not a useful detection;
    # enforce a floor rather than emitting a sliver the matcher would reject.
    if x2 - x1 < 2.0:
        x2 = x1 + 2.0
    if y2 - y1 < 2.0:
        y2 = y1 + 2.0
    return (
        float(np.clip(x1, 0, iw)), float(np.clip(y1, 0, ih)),
        float(np.clip(x2, 0, iw)), float(np.clip(y2, 0, ih)),
    )


def _random_fp_box(rng: np.random.Generator, iw: int, ih: int, params: NoiseParams) -> BBox:
    area_frac = rng.uniform(*params.fp_size_frac)
    area = area_frac * iw * ih
    aspect = rng.uniform(0.3, 3.0)
    h = float(np.sqrt(area / aspect))
    w = float(area / max(h, 1.0))
    x1 = rng.uniform(0, max(iw - w, 1.0))
    y1 = rng.uniform(0, max(ih - h, 1.0))
    return (x1, y1, min(x1 + w, iw), min(y1 + h, ih))


def _score(rng: np.random.Generator, params: NoiseParams) -> float:
    return float(np.clip(rng.normal(params.score_mean, params.score_sigma), 0.02, 0.999))


def observe(
    rng: np.random.Generator,
    scene: Scene,
    mapper: CameraModelMapper,
    iw: int,
    ih: int,
    params: NoiseParams,
) -> list[dict]:
    """Truth -> noisy ``predictions`` for one instance.

    Applies, per group in a random order: miss (drop entirely), merge (fold
    into the next remaining group's box), split (emit two adjacent boxes
    instead of one), then per-box jitter and a score draw. A single false
    positive is added with probability ``fp_rate``.
    """
    truth = truth_boxes_per_group(scene, mapper, iw, ih)
    remaining = list(truth.keys())
    rng.shuffle(remaining)

    preds: list[dict] = []
    while remaining:
        gid = remaining.pop(0)
        box = truth[gid]

        if rng.random() < params.miss_rate:
            continue

        if remaining and rng.random() < params.merge_rate:
            other_gid = remaining.pop(0)
            obox = truth[other_gid]
            box = (
                min(box[0], obox[0]), min(box[1], obox[1]),
                max(box[2], obox[2]), max(box[3], obox[3]),
            )

        boxes_to_emit: list[BBox] = [box]
        if rng.random() < params.split_rate:
            x1, y1, x2, y2 = box
            if (x2 - x1) >= (y2 - y1):
                mid = (x1 + x2) / 2.0
                boxes_to_emit = [(x1, y1, mid, y2), (mid, y1, x2, y2)]
            else:
                mid = (y1 + y2) / 2.0
                boxes_to_emit = [(x1, y1, x2, mid), (x1, mid, x2, y2)]

        for b in boxes_to_emit:
            jb = _jitter_box(rng, b, params.jitter_px_sigma, iw, ih)
            preds.append({"bbox": list(jb), "score": _score(rng, params)})

    if rng.random() < params.fp_rate:
        fp = _random_fp_box(rng, iw, ih, params)
        preds.append({"bbox": list(fp), "score": _score(rng, params)})

    return preds
