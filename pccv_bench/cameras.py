"""Synthetic camera calibration.

Deliberately does **not** ship any real camera model, since those may reveal
cell geometry. Instead this module synthesizes a
ground-truth pinhole+distortion camera, samples noisy correspondences from it,
and fits the production :class:`~pccv_bench.core.camera_model.CameraModelMapper` on those
— exercising the *real* fitting and projection code (including the
out-of-domain homography fallback) while containing zero real geometry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.core.camera_model import CameraModelMapper

#: Per-camera target reprojection RMS (pixels), measured on the real
#: installation. Reproducing this
#: spread — including one deliberately poorly-calibrated view — matters
#: because the residual-misalignment story is exactly what the bounded
#: x-offset correction exists for; a benchmark where every camera is perfect
#: cannot exercise that mechanism.
TARGET_RMS_PX: dict[str, float] = {
    "cam1": 3.17,
    "cam2": 2.80,
    "cam3": 8.54,   # the deliberately poorly-calibrated view
    "cam4": 3.88,
    "cam5": 4.44,
    "cam6": 1.10,
}

#: model_type used per camera when fitting — mirrors the real fleet (mostly
#: polynomial, one "auto"/Brown+RBF).
MODEL_TYPE: dict[str, str] = {
    "cam1": "polynomial",
    "cam2": "polynomial",
    "cam3": "polynomial",
    "cam4": "polynomial",
    "cam5": "polynomial",
    "cam6": "auto",
}

DEFAULT_IMAGE_WH: tuple[int, int] = (1920, 1024)

#: Synthetic ground-truth camera geometry: a downward-tilted pinhole several
#: metres above the table, in the same world-mm frame as scene.py's
#: TABLE_HALF_X/Y (a fixed camera's own **local**, elongated view of one
#: station — matching the real longitudinal cameras, which need
#: enough length to hold up to 7 groups without excessive overlap — not the
#: real span of any station). Deliberately not derived from any real camera
#: pose. Tuned (not measured) jointly with scene.py's table extent so: (a) a
#: generated 80mm part projects to a realistic box-width scale (~40-75px,
#: vs. the measured real median of ~143px for larger multi-part groups); (b)
#: the projection is close enough to affine over the domain that a quadratic
#: ``polynomial`` fit leaves a small systematic residual (~0.01px noiseless)
#: rather than one dominating the fit's reported RMS over the injected noise,
#: which would make TARGET_RMS_PX above unreachable; (c) ~90% of sampled world
#: points land inside a 1920x1024 frame — the remaining ~10% clip to the
#: border like a real camera's edge effects, which observe.py's near-frame
#: filter treats as "not visible" rather than emitting a spurious sliver
#: detection for it. Checked at build time: a wide-FOV camera sized to see an
#: 1800x2800mm table in one frame keeps the residual just as low but shrinks
#: an 80mm part to ~7-11px — too small for the matcher's
#: IoU/coverage/containment thresholds to behave realistically, and
#: disproportionate to a fixed jitter_px_sigma.
_GT_FOCAL_PX = 3500.0
_GT_HEIGHT_MM = 6500.0
_GT_TILT_DEG = 2.0


def _synthetic_ground_truth_project(
    world_xy: np.ndarray, iw: int, ih: int
) -> np.ndarray:
    """A simple downward-tilted pinhole projection, purely synthetic.

    Not the model under test — this exists only to generate self-consistent
    (world, pixel) correspondence *pairs* for ``CameraModelMapper.fit`` to
    recover from, the same way a real calibration pass would.
    """
    x, y = world_xy[:, 0], world_xy[:, 1]
    tilt = np.radians(_GT_TILT_DEG)
    # Camera looks along +y, tilted down; z is the fixed table plane (0).
    cam_y = y * np.cos(tilt) - _GT_HEIGHT_MM * np.sin(tilt)
    cam_z = y * np.sin(tilt) + _GT_HEIGHT_MM * np.cos(tilt)
    cam_z = np.clip(cam_z, 1.0, None)  # guard against behind-camera math
    u = iw / 2.0 + _GT_FOCAL_PX * x / cam_z
    v = ih / 2.0 + _GT_FOCAL_PX * cam_y / cam_z
    return np.stack([u, v], axis=1)


def fit_synthetic_camera(
    rng: np.random.Generator,
    camera: str,
    n_samples: int = 40,
    image_wh: tuple[int, int] = DEFAULT_IMAGE_WH,
) -> CameraModelMapper:
    """Fit a :class:`CameraModelMapper` on synthetic correspondences for *camera*.

    Samples *n_samples* world points over the table plane, projects them
    through a synthetic ground-truth camera, adds pixel noise sized to hit
    ``TARGET_RMS_PX[camera]``, and fits the production mapper on the result —
    the same ``fit()`` entry point a real calibration pass uses.
    """
    if camera not in TARGET_RMS_PX:
        raise ValueError(f"unknown synthetic camera {camera!r}")
    iw, ih = image_wh

    from pccv_bench.scene import TABLE_HALF_X, TABLE_HALF_Y

    world_x = rng.uniform(-TABLE_HALF_X, TABLE_HALF_X, size=n_samples)
    world_y = rng.uniform(-TABLE_HALF_Y, TABLE_HALF_Y, size=n_samples)
    world_xy = np.stack([world_x, world_y], axis=1)

    pixel_uv = _synthetic_ground_truth_project(world_xy, iw, ih)
    noise_sigma = TARGET_RMS_PX[camera]
    pixel_uv_noisy = pixel_uv + rng.normal(0.0, noise_sigma, size=pixel_uv.shape)

    samples = [
        {"pixel_u": float(u), "pixel_v": float(v), "world_x": float(x), "world_y": float(y)}
        for (u, v), (x, y) in zip(pixel_uv_noisy, world_xy)
    ]

    mapper = CameraModelMapper(camera, model_type=MODEL_TYPE[camera])
    mapper.fit(samples)
    return mapper


def fit_all_synthetic_cameras(
    rng: np.random.Generator, n_samples: int = 40
) -> dict[str, CameraModelMapper]:
    from pccv_bench.scene import CAMERAS

    return {cam: fit_synthetic_camera(rng, cam, n_samples=n_samples) for cam in CAMERAS}
