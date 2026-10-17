"""Unit tests for the generator components (scene/observe/provider mechanics),
distinct from the analytic matcher tests in ``test_matcher_analytic.py``.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.core.harness import ArmConfig, run_arm
from pccv_bench.core.inject import EXPECTED_ACTION, INJECTION_CLASSES

from pccv_bench.cameras import fit_synthetic_camera
from pccv_bench.instance_io import instance_from_dict, instance_to_dict, load_instances, save_instances
from pccv_bench.observe import observe, truth_boxes_per_group
from pccv_bench.scene import BOX_COUNT_DIST, CAMERAS, generate_scene, sample_box_count
from pccv_bench.synthetic_provider import IMAGE_WH, SyntheticProvider


# ── scene ──────────────────────────────────────────────────────────────────────


def test_every_camera_has_a_measured_distribution():
    assert set(CAMERAS) == set(BOX_COUNT_DIST)
    for cam, dist in BOX_COUNT_DIST.items():
        assert sum(dist.values()) > 0
        assert 0 in dist, f"{cam} distribution should include the empty-table case"


def test_sampled_box_counts_stay_within_the_measured_range():
    rng = np.random.default_rng(0)
    for cam in CAMERAS:
        allowed = set(BOX_COUNT_DIST[cam])
        for _ in range(50):
            assert sample_box_count(rng, cam) in allowed


def test_unknown_camera_is_rejected():
    with pytest.raises(ValueError):
        generate_scene(np.random.default_rng(0), "not_a_camera")


def test_generate_scene_respects_an_explicit_group_count():
    scene = generate_scene(np.random.default_rng(0), "cam1", n_groups=4)
    assert len(scene.group_ids()) == 4


def test_generated_parts_have_only_geometric_fields():
    """Scope constraint: no type/category/product attribute."""
    scene = generate_scene(np.random.default_rng(0), "cam1", n_groups=2)
    for p in scene.parts_world():
        assert set(p) <= {"id", "x", "y", "sizeX", "sizeY", "rotation", "table_kind", "z", "mass"}


# ── observe ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mapper():
    return fit_synthetic_camera(np.random.default_rng(1), "cam1")


def test_truth_boxes_cover_every_group_or_skip_it_for_visibility(mapper):
    scene = generate_scene(np.random.default_rng(2), "cam1", n_groups=3)
    truth = truth_boxes_per_group(scene, mapper, 1920, 1024)
    assert set(truth) <= set(scene.group_ids())
    for x1, y1, x2, y2 in truth.values():
        assert x2 - x1 >= 1 and y2 - y1 >= 1


def test_observe_emits_at_most_one_prediction_per_group_absent_split(mapper):
    from pccv_bench.noise import NoiseParams

    zero_noise = NoiseParams(miss_rate=0.0, fp_rate=0.0, split_rate=0.0, merge_rate=0.0)
    scene = generate_scene(np.random.default_rng(3), "cam1", n_groups=3)
    preds = observe(np.random.default_rng(4), scene, mapper, 1920, 1024, zero_noise)
    assert len(preds) == len(truth_boxes_per_group(scene, mapper, 1920, 1024))


def test_high_miss_rate_can_drop_every_prediction(mapper):
    from pccv_bench.noise import NoiseParams

    always_miss = NoiseParams(miss_rate=1.0, fp_rate=0.0)
    scene = generate_scene(np.random.default_rng(5), "cam1", n_groups=2)
    preds = observe(np.random.default_rng(6), scene, mapper, 1920, 1024, always_miss)
    assert preds == []


# ── synthetic provider ─────────────────────────────────────────────────────────


def test_provider_yields_the_requested_class_and_correct_labels():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        provider = SyntheticProvider("missing_part", n_instances=5, seed=0)
        instances = list(provider)
    assert len(instances) == 5
    for inst in instances:
        assert inst.injection_class == "missing_part"
        assert inst.expected_action == EXPECTED_ACTION["missing_part"]


def test_clean_instances_are_never_labelled_inconsistent():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        instances = list(SyntheticProvider("clean", n_instances=10, seed=1))
    assert all(i.expected_action == "pass" for i in instances)


def test_every_injection_class_can_generate_at_least_one_instance():
    """A smoke check across the whole class list, not just one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mappers = None
        for cls in INJECTION_CLASSES:
            p = SyntheticProvider(cls, n_instances=3, seed=2, mappers=mappers)
            mappers = p.mappers
            instances = list(p)
            assert len(instances) == 3, f"{cls} failed to generate any instances"


def test_provider_is_deterministic_for_a_fixed_seed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = [i.instance_id for i in SyntheticProvider("missing_part", n_instances=5, seed=7)]
        b = [i.instance_id for i in SyntheticProvider("missing_part", n_instances=5, seed=7)]
    assert a == b


# ── per-camera noise dispatch (2026-09-16, per-camera calibration pass) ────────


def test_noise_for_camera_returns_the_single_noiseparams_unchanged():
    """Backward compatibility: a plain NoiseParams (every call site in this
    codebase before per-camera dispatch existed) must still mean exactly one
    set of parameters shared by every camera."""
    from pccv_bench.noise import NoiseParams

    pooled = NoiseParams(miss_rate=0.2)
    provider = SyntheticProvider("clean", n_instances=1, seed=0, noise=pooled)
    assert provider._noise_for_camera("cam1") is pooled
    assert provider._noise_for_camera("cam6") is pooled


def test_noise_for_camera_dispatches_a_dict_by_camera_name():
    from pccv_bench.noise import NoiseParams

    noisy_camera_noise = NoiseParams(miss_rate=0.9)
    quiet_camera_noise = NoiseParams(miss_rate=0.1)
    provider = SyntheticProvider(
        "clean", n_instances=1, seed=0,
        noise={"cam1": noisy_camera_noise, "cam4": quiet_camera_noise},
    )
    assert provider._noise_for_camera("cam1") is noisy_camera_noise
    assert provider._noise_for_camera("cam4") is quiet_camera_noise


def test_noise_for_camera_dict_falls_back_to_default_noise_for_missing_key():
    """noise.py's documented convention: a camera key absent from the dict
    falls back to DEFAULT_NOISE, not an error or the hand-set NoiseParams()
    default of some other field."""
    from pccv_bench.noise import DEFAULT_NOISE, NoiseParams

    provider = SyntheticProvider(
        "clean", n_instances=1, seed=0,
        noise={"cam1": NoiseParams(miss_rate=0.9)},
    )
    assert provider._noise_for_camera("cam6") is DEFAULT_NOISE


def test_per_camera_noise_dict_reaches_the_generated_detections():
    """End-to-end: not just that the accessor resolves the right object, but
    that a per-camera override actually changes what observe() produces for
    that camera specifically, leaving another camera in the same provider
    unaffected."""
    from pccv_bench.noise import NoiseParams

    always_miss = NoiseParams(miss_rate=1.0, fp_rate=0.0, split_rate=0.0, merge_rate=0.0)
    never_miss = NoiseParams(
        miss_rate=0.0, fp_rate=0.0, split_rate=0.0, merge_rate=0.0, jitter_px_sigma=0.0
    )
    provider = SyntheticProvider(
        "clean", n_instances=1, seed=11,
        cameras=("cam1", "cam3"),
        noise={"cam1": always_miss, "cam3": never_miss},
    )

    rng_a = provider._rng_for(1)
    _, dets_always_miss = provider._scene_with_detections(rng_a, "cam1", require_nonempty=True)
    assert dets_always_miss == []

    rng_b = provider._rng_for(2)
    scene_b, dets_never_miss = provider._scene_with_detections(rng_b, "cam3", require_nonempty=True)
    truth_b = truth_boxes_per_group(scene_b, provider.mappers["cam3"], *IMAGE_WH)
    assert len(dets_never_miss) == len(truth_b)


# ── serialization ──────────────────────────────────────────────────────────────


def test_instance_json_round_trip_is_lossless():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        insts = list(SyntheticProvider("position_shift_x_inbound", n_instances=3, seed=3))
    for inst in insts:
        d = instance_to_dict(inst)
        back = instance_from_dict(d)
        assert back == inst


def test_save_and_load_instances_round_trips(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        insts = list(SyntheticProvider("snapshot_desync", n_instances=4, seed=4))
    path = tmp_path / "instances.json"
    save_instances(path, insts, meta={"seed": 4})
    loaded, meta = load_instances(path)
    assert len(loaded) == len(insts)
    assert meta["seed"] == 4
    assert [i.instance_id for i in loaded] == [i.instance_id for i in insts]


def test_saved_instances_are_scoreable_by_the_harness(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        provider = SyntheticProvider("missing_part", n_instances=5, seed=5)
        insts = list(provider)
    path = tmp_path / "instances.json"
    save_instances(path, insts)
    loaded, _ = load_instances(path)
    verdicts = run_arm(loaded, ArmConfig(name="geometry", metric="containment", threshold=0.3),
                        mappers=provider.mappers)
    assert len(verdicts) == len(insts)
