"""Analytic-truth tests against the production matcher and camera model.

These are the **first tests for ``matching.py`` and ``camera_model.py``**
— everything here is a case where the correct
answer is known by construction, not measured. Where the module says a case
must give a specific answer, the test asserts that exact answer, not "roughly".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.core.camera_model import CameraModelMapper
from pccv_bench.core.matching import (
    CORRECTION_STEP_MM,
    ORANGE_THRESHOLD,
    auto_match_boxes_to_parts,
    orange_ratio,
)

from pccv_bench.core.projection import project_world_parts

from pccv_bench.cameras import fit_synthetic_camera
from pccv_bench.observe import truth_boxes_per_group
from pccv_bench.scene import generate_scene

IMAGE_WH = (1920, 1024)


@pytest.fixture(scope="module")
def mapper() -> CameraModelMapper:
    rng = np.random.default_rng(123)
    return fit_synthetic_camera(rng, "cam1", n_samples=40)


def _one_part_world(x: float = 0.0, y: float = 0.0) -> list[dict]:
    return [{
        "id": "p1", "x": x, "y": y, "sizeX": 80.0, "sizeY": 240.0,
        "rotation": 0.0, "table_kind": "saw",
    }]


def _record_and_box(mapper: CameraModelMapper, x: float, y: float):
    parts = _one_part_world(x, y)
    record = project_world_parts(parts, mapper, *IMAGE_WH)
    pts = np.asarray(record[0]["polygon"], dtype=np.float64)
    box = (
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    )
    return record, box


# ── 1. Identity ──────────────────────────────────────────────────────────────


def test_identity_a_part_matches_its_own_projection(mapper):
    record, box = _record_and_box(mapper, 0.0, 0.0)
    detections = [{"bbox": list(box)}]
    result = auto_match_boxes_to_parts(record, detections, metric="containment", threshold=0.3)
    assert result["groups"][0]["best_value"] >= 0.98
    assert result["has_errors"] is False


# ── 2. Known translation, recovered ─────────────────────────────────────────


def test_known_translation_is_recovered_within_the_correction_step(mapper):
    """Shift the record by exactly 200mm in x; a 300mm-bound sweep must
    recover it, landing within one step (CORRECTION_STEP_MM) of 200mm.

    The sweep can pass at a step slightly *before* the true offset — at
    threshold 0.9 (not 1.0), a residual under-correction of one step (5mm on
    an 80mm-wide part, a ~6% fractional misalignment) can already clear 90%
    containment. That is the metric's threshold tolerance working as
    designed, not a sweep-precision bug, so the window below is centred on
    200mm rather than requiring recovery only at or after it.
    """
    _, box = _record_and_box(mapper, 0.0, 0.0)  # detection stays at x=0
    shifted_record, _ = _record_and_box(mapper, -200.0, 0.0)  # record moved -200mm
    detections = [{"bbox": list(box)}]
    result = auto_match_boxes_to_parts(
        shifted_record, detections, metric="containment", threshold=0.9,
        correction_offset_mm=300.0, mapper=mapper,
    )
    g = result["groups"][0]
    assert g["is_error"] is False
    assert g["is_corrected_error"] is True
    assert g["correction_method"] == "x_offset"
    assert 200.0 - CORRECTION_STEP_MM <= g["corrected_offset_mm"] <= 200.0 + CORRECTION_STEP_MM


# ── 3. Beyond bound ──────────────────────────────────────────────────────────


def test_shift_beyond_the_bound_stays_an_error(mapper):
    _, box = _record_and_box(mapper, 0.0, 0.0)
    shifted_record, _ = _record_and_box(mapper, -700.0, 0.0)
    detections = [{"bbox": list(box)}]
    result = auto_match_boxes_to_parts(
        shifted_record, detections, metric="containment", threshold=0.9,
        correction_offset_mm=600.0, mapper=mapper,
    )
    assert result["has_errors"] is True


# ── 4. Y-shift unrecoverable by the x-only sweep ────────────────────────────


def test_y_shift_is_not_recovered_by_the_x_only_sweep(mapper):
    _, box = _record_and_box(mapper, 0.0, 0.0)
    shifted_record, _ = _record_and_box(mapper, 0.0, -200.0)
    detections = [{"bbox": list(box)}]
    result = auto_match_boxes_to_parts(
        shifted_record, detections, metric="containment", threshold=0.9,
        correction_offset_mm=600.0, mapper=mapper,
    )
    assert result["has_errors"] is True, "the x-offset sweep must not silently fix a y-shift"


# ── 5. Missing part ──────────────────────────────────────────────────────────


def test_dropping_the_only_explaining_part_is_an_error(mapper):
    _, box = _record_and_box(mapper, 0.0, 0.0)
    detections = [{"bbox": list(box)}]
    result = auto_match_boxes_to_parts([], detections, metric="containment", threshold=0.3)
    assert result["has_errors"] is True
    assert result["groups"][0]["is_error"] is True


# ── 6. Ghost part — two-sided ────────────────────────────────────────────────


def test_ghost_part_is_invisible_box_centric_but_caught_two_way(mapper):
    """The central finding, as a test rather than a claim."""
    ghost_record, _ = _record_and_box(mapper, 500.0, 500.0)  # a part with nothing observed there
    detections: list[dict] = []  # nothing detected anywhere

    box_centric = auto_match_boxes_to_parts(
        ghost_record, detections, metric="containment", threshold=0.3
    )
    assert box_centric["has_errors"] is False, (
        "box-centric semantics: unmatched record parts are not errors "
        "(matching.py L471-473) — this is the documented limitation, not a bug"
    )

    from pccv_bench.core.twoway import record_side_unmatched
    unmatched = record_side_unmatched(
        ghost_record, detections, threshold=0.3, image_wh=IMAGE_WH,
    )
    assert unmatched == [0], "the two-way channel must catch what the box-centric path cannot"


# ── 7. Temporal ──────────────────────────────────────────────────────────────


def test_temporal_retry_recovers_from_the_next_snapshot(mapper):
    _, box = _record_and_box(mapper, 0.0, 0.0)
    wrong_record, _ = _record_and_box(mapper, 900.0, 900.0)  # unrelated location
    correct_next_record, _ = _record_and_box(mapper, 0.0, 0.0)  # matches the detection
    detections = [{"bbox": list(box)}]

    result = auto_match_boxes_to_parts(
        wrong_record, detections, metric="containment", threshold=0.9,
        next_run_gql_parts=correct_next_record,
    )
    g = result["groups"][0]
    assert g["is_error"] is False
    assert g["correction_method"] == "temporal"


def test_temporal_retry_does_not_fire_without_a_next_snapshot(mapper):
    _, box = _record_and_box(mapper, 0.0, 0.0)
    wrong_record, _ = _record_and_box(mapper, 900.0, 900.0)
    detections = [{"bbox": list(box)}]
    result = auto_match_boxes_to_parts(
        wrong_record, detections, metric="containment", threshold=0.9,
        next_run_gql_parts=None,
    )
    assert result["has_errors"] is True


# ── 8. Occlusion ─────────────────────────────────────────────────────────────


def test_orange_fraction_just_below_threshold_is_not_filtered():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # ORANGE_THRESHOLD=0.002 of 10,000px = 20px; colour 19 to stay just under.
    img[0, :19] = (0, 140, 255)
    assert orange_ratio(img) < ORANGE_THRESHOLD


def test_orange_fraction_just_above_threshold_is_filtered():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0, :21] = (0, 140, 255)  # 21px > 20px threshold
    assert orange_ratio(img) >= ORANGE_THRESHOLD


# ── 9. Camera round-trip ─────────────────────────────────────────────────────


@pytest.mark.parametrize("model_type", ["polynomial", "auto"])
def test_camera_fit_save_load_project_round_trips(tmp_path, model_type):
    rng = np.random.default_rng(7)
    m1 = CameraModelMapper("test_cam", model_type=model_type)
    # Reuse the same synthetic correspondence generator cameras.py uses, just
    # with an explicit model_type override for this test's parametrization.
    from pccv_bench.cameras import _synthetic_ground_truth_project
    from pccv_bench.scene import TABLE_HALF_X, TABLE_HALF_Y

    n = 40
    wx = rng.uniform(-TABLE_HALF_X, TABLE_HALF_X, size=n)
    wy = rng.uniform(-TABLE_HALF_Y, TABLE_HALF_Y, size=n)
    world_xy = np.stack([wx, wy], axis=1)
    pix = _synthetic_ground_truth_project(world_xy, *IMAGE_WH) + rng.normal(0, 1.0, size=(n, 2))
    samples = [
        {"pixel_u": float(u), "pixel_v": float(v), "world_x": float(x), "world_y": float(y)}
        for (u, v), (x, y) in zip(pix, world_xy)
    ]
    m1.fit(samples)

    test_pts = np.array([[0.0, 0.0], [100.0, -50.0], [-200.0, 150.0]])
    proj_before = m1.project(test_pts)

    save_path = tmp_path / "cam.json"
    m1.save(save_path)
    m2 = CameraModelMapper.load(save_path)
    proj_after = m2.project(test_pts)

    assert np.allclose(proj_before, proj_after, atol=1e-6)


def test_synthetic_camera_fit_hits_the_right_order_of_magnitude_rms():
    """Not exact (see cameras.py's docstring on the fit's own bias), but the
    fitted RMS must be within a small factor of the target, not off by 10x+
    the way an unrealistic camera geometry produced before this was tuned."""
    from pccv_bench.cameras import TARGET_RMS_PX

    rng = np.random.default_rng(1)
    for cam, target in TARGET_RMS_PX.items():
        m = fit_synthetic_camera(rng, cam, n_samples=40)
        rms = m.fit_meta.get("global_reproj_rms_px", float("inf"))
        assert rms < max(target * 3.0, 2.0), f"{cam}: rms={rms} target={target}"


# ── scene / observe sanity (load-bearing for everything above) ─


def test_generated_group_boxes_have_positive_area(mapper):
    scene = generate_scene(np.random.default_rng(9), "cam1", n_groups=3)
    truth = truth_boxes_per_group(scene, mapper, *IMAGE_WH)
    for x1, y1, x2, y2 in truth.values():
        assert x2 > x1 and y2 > y1
