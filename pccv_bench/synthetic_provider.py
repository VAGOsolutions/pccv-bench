"""``SyntheticProvider`` — the second ``InstanceProvider`` for the shared harness.

Plugs fully-synthetic scenes into ``pccv_bench.core.harness.run_arm`` exactly the
way ``pccv_bench.core.projection.RealProvider`` plugs in real ones, so both
studies score through the identical matcher-invocation + metric layer. This is
the concrete realisation of the "same code, two data sources" requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.core.camera_model import CameraModelMapper
from pccv_bench.core.matching import ORANGE_THRESHOLD, orange_ratio

from pccv_bench.core.harness import Instance, MAX_X_OFFSET_MM
from pccv_bench.core.inject import EXPECTED_ACTION, INJECTORS, InjectionContext, apply_ops
from pccv_bench.core.oracle import validate_injection
from pccv_bench.core.projection import project_world_parts
from pccv_bench.core.seeding import stable_seed

from pccv_bench.cameras import fit_synthetic_camera
from pccv_bench.noise import NoiseParams, DEFAULT_NOISE
from pccv_bench.observe import observe
from pccv_bench.scene import CAMERAS, Scene, generate_scene

IMAGE_WH: tuple[int, int] = (1920, 1024)

#: Classes needing a non-empty scene to perturb — the synthetic mirror of
#: ``pccv_bench.core.projection.CLASS_NEEDS_DETECTIONS``, same rationale:
#: the box-centric matcher has nothing to evaluate on an empty table.
CLASS_NEEDS_DETECTIONS: dict[str, bool] = {
    "clean": False,
    "missing_part": True,
    "ghost_part": False,
    "position_shift_x_inbound": True,
    "position_shift_out": True,
    "snapshot_desync": False,
    "composition_swap": True,
}


def _make_occlusion_frame(
    rng: np.random.Generator, iw: int, ih: int, occluded: bool
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A synthetic BGR frame whose orange-pixel fraction straddles
    ``ORANGE_THRESHOLD``, so the real :func:`orange_ratio` runs unmodified on
    it rather than being mocked.
    """
    img = np.zeros((ih, iw, 3), dtype=np.uint8)
    frac = rng.uniform(0.01, 0.20) if occluded else rng.uniform(0.0, 0.0008)
    area = max(frac * iw * ih, 1.0)
    w = max(1, int(np.sqrt(area * rng.uniform(0.5, 2.0))))
    h = max(1, int(area / w))
    x0 = int(rng.uniform(0, max(iw - w, 1)))
    y0 = int(rng.uniform(0, max(ih - h, 1)))
    # BGR ~ orange; matching.ORANGE_LOWER/UPPER are HSV(5-25, 100-255, 100-255).
    img[y0 : y0 + h, x0 : x0 + w] = (0, 140, 255)
    return img, (x0, y0, x0 + w, y0 + h)


def _small_drift(
    rng: np.random.Generator, parts_world: list[dict], sigma_mm: float = 3.0
) -> list[dict]:
    """A plausible "next run" of the *same* true record: same parts (same
    ids), positions nudged by a small natural-production amount.

    Deliberately reuses the input parts' ids rather than generating a fresh
    independent scene. ``apply_ops`` (the replay of a deliberate injection
    onto the next snapshot) matches parts **by id**; an independently
    generated scene restarts its id sequence from ``synth#000001`` every call,
    so replaying an op against one would — at best coincidentally, at worst
    silently wrongly — hit an unrelated part that happens to share an id,
    not "the same physical part a moment later". This keeps genuine identity
    continuity, the same guarantee a real next-run GQL snapshot gives for
    free (see ``pccv_bench.core.projection``).
    """
    out = []
    for p in parts_world:
        q = dict(p)
        q["x"] = float(q.get("x", 0.0)) + float(rng.normal(0.0, sigma_mm))
        q["y"] = float(q.get("y", 0.0)) + float(rng.normal(0.0, sigma_mm))
        out.append(q)
    return out


def _truncate_under_occlusion(
    detections: list[dict], region: tuple[int, int, int, int]
) -> list[dict]:
    """Drop or shrink detections whose box overlaps the occluding region —
    the observation-side half of "occlusion suppresses/truncates detections
    beneath it"."""
    rx1, ry1, rx2, ry2 = region
    out: list[dict] = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        ix1, iy1 = max(x1, rx1), max(y1, ry1)
        ix2, iy2 = min(x2, rx2), min(y2, ry2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max((x2 - x1) * (y2 - y1), 1.0)
        if inter / area > 0.6:
            continue  # mostly covered -> dropped, mirrors a real occluded miss
        out.append(d)
    return out


class SyntheticProvider:
    """Generates labelled :class:`Instance` objects for one injection class.

    Mirrors :class:`~pccv_bench.core.projection.RealProvider`'s constructor
    shape so the two are interchangeable call sites in the ablation/report
    tooling.
    """

    def __init__(
        self,
        injection_class: str,
        n_instances: int,
        seed: int = 42,
        cameras: tuple[str, ...] = CAMERAS,
        noise: NoiseParams | dict[str, NoiseParams] = DEFAULT_NOISE,
        occlusion_rate: float = 0.10,
        mappers: dict[str, CameraModelMapper] | None = None,
    ) -> None:
        if injection_class not in EXPECTED_ACTION:
            raise ValueError(f"unknown injection class {injection_class!r}")
        self.injection_class = injection_class
        self.n_instances = n_instances
        self.seed = seed
        self.cameras = cameras
        self.noise = noise
        self.occlusion_rate = occlusion_rate
        # A real camera is calibrated once, not per photo — fit each requested
        # camera's mapper a single time (from a fixed early seed, independent
        # of per-instance randomness) and reuse it across every instance on
        # that camera, exactly as the real study reuses one CameraModelMapper
        # per camera for its whole run.
        if mappers is not None:
            self.mappers = mappers
        else:
            cam_rng = np.random.default_rng(seed)
            self.mappers = {c: fit_synthetic_camera(cam_rng, c) for c in cameras}
        self.rejected: list[dict] = []

    def _rng_for(self, i: int) -> np.random.Generator:
        return np.random.default_rng(stable_seed(self.seed, self.injection_class, i))

    def _noise_for_camera(self, camera: str) -> NoiseParams:
        """Resolve ``self.noise`` (``NoiseParams | dict[str, NoiseParams]``,
        see ``noise.py``'s module docstring for the convention) to the one
        :class:`NoiseParams` to use for *camera*.

        A plain ``NoiseParams`` — every call site in this codebase before
        this method existed — is returned unchanged for every camera,
        preserving that behaviour exactly. A dict is looked up by camera
        name, falling back to :data:`DEFAULT_NOISE` (the same pooled/default
        params the single-``NoiseParams`` form already means) for any
        camera key the dict doesn't cover.
        """
        if isinstance(self.noise, dict):
            return self.noise.get(camera, DEFAULT_NOISE)
        return self.noise

    def _scene_with_detections(
        self,
        rng: np.random.Generator,
        camera: str,
        require_nonempty: bool,
        min_total_parts: int = 0,
    ) -> tuple[Scene, list[dict]]:
        """Regenerate until the scene meets *require_nonempty* /
        *min_total_parts*, so a class-specific requirement (e.g.
        ``composition_swap`` needing >=2 parts to even attempt a swap) doesn't
        depend on the caller happening to draw a big-enough scene by luck.
        Most groups are single-part (scene.py's parts-per-group draw is 70%
        weight-1), so requiring only >=1 *group* here — as the other five
        classes do — routinely handed composition_swap a 1-part scene and
        rejected it for having nothing to pair, before its own genuine
        envelope-delta check ever ran.
        """
        mapper = self.mappers[camera]
        for _ in range(20):
            scene = generate_scene(rng, camera)
            if require_nonempty and not scene.group_ids():
                continue
            if len(scene.parts) < min_total_parts:
                continue
            detections = observe(rng, scene, mapper, *IMAGE_WH, self._noise_for_camera(camera))
            return scene, detections
        return generate_scene(rng, camera), []

    def __iter__(self) -> Iterator[Instance]:
        need_nonempty = CLASS_NEEDS_DETECTIONS.get(self.injection_class, True)
        emitted = 0
        attempt = 0
        while emitted < self.n_instances:
            attempt += 1
            rng = self._rng_for(attempt)
            camera = self.cameras[attempt % len(self.cameras)]
            mapper = self.mappers[camera]
            min_parts = 2 if self.injection_class == "composition_swap" else 0
            scene, detections = self._scene_with_detections(
                rng, camera, need_nonempty, min_parts
            )
            parts_world = scene.parts_world()

            occluded = rng.random() < self.occlusion_rate
            occl_img, occl_region = _make_occlusion_frame(rng, *IMAGE_WH, occluded)
            is_occluded = orange_ratio(occl_img) >= ORANGE_THRESHOLD
            if is_occluded:
                detections = _truncate_under_occlusion(detections, occl_region)
            stratum = "occluded" if is_occluded else "none"

            if self.injection_class == "snapshot_desync":
                donor_scene, _ = self._scene_with_detections(rng, camera, False)
                donor_parts = donor_scene.parts_world()
                if not donor_parts and not parts_world:
                    self.rejected.append(
                        {"attempt": attempt, "reason": "both_records_empty"}
                    )
                    continue
                record_parts = project_world_parts(donor_parts, mapper, *IMAGE_WH)
                next_parts = project_world_parts(parts_world, mapper, *IMAGE_WH)
                # Same validity gate the real study applies: a
                # donor whose projected geometry matches the true record is
                # not a desynchronization, just two empty or identical frames.
                validity = validate_injection(
                    next_parts, record_parts, detections,
                    camera=camera, injection_class="snapshot_desync",
                    image_wh=IMAGE_WH,
                )
                if not validity.ok:
                    self.rejected.append(
                        {"attempt": attempt, "reason": validity.rejection_reason}
                    )
                    continue
                inst = Instance(
                    instance_id=f"synthetic|{camera}|{attempt}|snapshot_desync",
                    camera=camera,
                    run_id=f"attempt-{attempt}",
                    detections=detections,
                    record_parts=record_parts,
                    image_wh=IMAGE_WH,
                    injection_class="snapshot_desync",
                    severity=1.0,
                    next_record_parts=next_parts,
                    occlusion_stratum=stratum,
                )
                emitted += 1
                yield inst
                continue

            def _project(pw: list[dict]) -> list[list[list[float]]]:
                flat = project_world_parts(pw, mapper, *IMAGE_WH)
                return [f["polygon"] for f in flat]

            donor_pool_scene, _ = self._scene_with_detections(rng, camera, False)
            ctx = InjectionContext(
                camera=camera,
                image_wh=IMAGE_WH,
                project=_project,
                detections=detections,
                donor_parts=donor_pool_scene.parts_world(),
                max_offset_mm=MAX_X_OFFSET_MM,
            )
            injector = INJECTORS[self.injection_class]
            res = injector(parts_world, rng, ctx)
            if not res.ok:
                self.rejected.append({"attempt": attempt, "reason": res.reason})
                continue

            record_parts = project_world_parts(res.parts_world, mapper, *IMAGE_WH)
            # The injector changed the record; the oracle decides whether that
            # change is something a camera could actually have caught — the
            # same rule the real study applies, kept out of reach of the
            # matcher being evaluated.
            validity = validate_injection(
                project_world_parts(parts_world, mapper, *IMAGE_WH), record_parts,
                detections, camera=camera,
                injection_class=res.injection_class, image_wh=IMAGE_WH,
            )
            if not validity.ok:
                self.rejected.append(
                    {"attempt": attempt, "reason": validity.rejection_reason}
                )
                continue
            next_true_world = _small_drift(rng, parts_world)
            next_world = apply_ops(next_true_world, res.ops)
            next_parts = (
                project_world_parts(next_world, mapper, *IMAGE_WH)
                if next_world
                else None
            )

            yield Instance(
                instance_id=f"synthetic|{camera}|{attempt}|{self.injection_class}",
                camera=camera,
                run_id=f"attempt-{attempt}",
                detections=detections,
                record_parts=record_parts,
                image_wh=IMAGE_WH,
                injection_class=res.injection_class,
                severity=res.severity,
                next_record_parts=next_parts,
                occlusion_stratum=stratum,
                meta=res.meta,
            )
            emitted += 1
