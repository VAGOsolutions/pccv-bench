"""Camera model fitting: world-to-pixel mapping with multiple model types.

Supported model_type values
---------------------------
auto          Brown camera model (cv2.calibrateCamera) when n >= 8, homography
              fallback for smaller sets or when Brown is unstable. RBF residual
              correction applied on top. (default — preserves existing behaviour)
brown         Brown model only, no RBF. Falls back to homography when Brown fails.
brown_rbf     Brown model + RBF residual correction.
homography    Plain perspective matrix (8 DOF), no RBF.
affine        Affine (2×3) transformation fitted via cv2.estimateAffine2D, stored
              as a 3×3 matrix and routed through the homography prediction path.
polynomial    Quadratic least-squares: 6-term feature vector
              [1, x, y, x², xy, y²] on normalised world coords.

Pipeline (auto)
---------------
1. Normalise world coords (centre + scale) for numerical stability.
2. Fit a global Brown model (cv2.calibrateCamera, planar scene at z=0).
   Falls back to homography when Brown fitting is unstable or n < 8.
3. Fit RBFInterpolator on the residuals (requires scipy).
4. project(world_xy) = global_predict(world_norm) + rbf_correct(world_norm)"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np


try:
    from scipy.interpolate import RBFInterpolator as _RBFInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


_VALID_MODEL_TYPES = {"auto", "brown", "brown_rbf", "homography", "affine", "polynomial"}


class CameraModelMapper:
    """Maps world coords (mm, planar) → crop pixel coords."""

    # Brown distortion polynomials (k1*r^2 + k2*r^4 + k3*r^6) diverge outside
    # the calibration domain.  Points beyond this margin (fraction of the
    # calibration bbox extent on each axis) are projected with a homography
    # fallback instead.
    _DOMAIN_MARGIN = 0.25

    def __init__(
        self,
        camera: str,
        smoothing: float = 0.0,
        epsilon: float = 1.0,
        max_control_points: int = 200,
        model_type: str = "auto",
    ) -> None:
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(f"Unknown model_type {model_type!r}. Valid: {sorted(_VALID_MODEL_TYPES)}")
        self.camera = camera
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.max_control_points = max_control_points
        self.model_type = model_type

        self._world_center: np.ndarray | None = None
        self._world_scale: float = 1.0

        # Brown model (cv2.calibrateCamera output)
        self._K: np.ndarray | None = None
        self._dist: np.ndarray | None = None
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None

        # Homography / affine (primary when Brown fails, extrapolation fallback otherwise)
        self._H: np.ndarray | None = None
        self._H_fallback: np.ndarray | None = None
        self._use_H: bool = True

        # Polynomial coefficients — shape (2, 6): coef[:, i] for u and v
        self._poly_coef: np.ndarray | None = None

        # Calibration domain in normalised coords (bbox of training points)
        self._calib_norm_min: np.ndarray | None = None
        self._calib_norm_max: np.ndarray | None = None

        # RBF correction
        self._rbf: Any = None
        self._rbf_X: np.ndarray | None = None
        self._rbf_Y: np.ndarray | None = None
        self._rbf_kernel: str = "thin_plate_spline"

        self.status: str = "unfitted"
        self.reason: str = ""
        self.fit_meta: dict[str, Any] = {}

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, samples: list[dict]) -> None:
        """Fit from list of {pixel_u, pixel_v, world_x, world_y} dicts."""
        n = len(samples)
        if n < 4:
            self.status = "failed"
            self.reason = f"too_few_samples ({n} < 4)"
            self.fit_meta = {"n_samples": n, "global_reproj_rms_px": float("nan"), "model_type": self.model_type}
            return

        world_xy = np.array([[s["world_x"], s["world_y"]] for s in samples], dtype=np.float64)
        pixel_uv = np.array([[s["pixel_u"], s["pixel_v"]] for s in samples], dtype=np.float64)

        # Normalise world to ≈ [-1, 1] for numerical stability
        self._world_center = world_xy.mean(axis=0)
        self._world_scale = float(max(np.abs(world_xy - self._world_center).max(), 1.0))
        world_norm = (world_xy - self._world_center) / self._world_scale

        mt = self.model_type
        use_rbf = False

        if mt == "auto":
            if n >= 8 and self._try_brown(world_norm, pixel_uv, n):
                self._use_H = False
                self._fit_homography_fallback(world_norm, pixel_uv)
            else:
                self._fit_homography(world_norm, pixel_uv)
                self._use_H = True
            use_rbf = True

        elif mt in ("brown", "brown_rbf"):
            if n >= 8 and self._try_brown(world_norm, pixel_uv, n):
                self._use_H = False
                self._fit_homography_fallback(world_norm, pixel_uv)
            else:
                # Brown not feasible — fall back gracefully
                self._fit_homography(world_norm, pixel_uv)
                self._use_H = True
            use_rbf = (mt == "brown_rbf")

        elif mt == "homography":
            self._fit_homography(world_norm, pixel_uv)
            self._use_H = True
            use_rbf = False

        elif mt == "affine":
            self._fit_affine(world_norm, pixel_uv)
            self._use_H = True
            use_rbf = False

        elif mt == "polynomial":
            self._fit_poly(world_norm, pixel_uv)
            use_rbf = False

        # Store calibration domain for extrapolation detection
        self._calib_norm_min = world_norm.min(axis=0).copy()
        self._calib_norm_max = world_norm.max(axis=0).copy()

        # RBF correction on residuals (only for model types that request it)
        if use_rbf and _HAS_SCIPY:
            self._fit_rbf(world_norm, pixel_uv)

        # Compute in-sample fit metrics
        pred = self._global_predict(world_norm)
        if self._rbf is not None:
            pred = pred + self._rbf_predict(world_norm)
        errs = np.sqrt(((pixel_uv - pred) ** 2).sum(axis=1))
        self.fit_meta = {
            "n_samples": n,
            "global_reproj_rms_px": float(np.sqrt(np.mean(errs ** 2))),
            "reproj_median_px": float(np.median(errs)),
            "reproj_max_px": float(errs.max()),
            "use_homography_fallback": self._use_H,
            "has_rbf": self._rbf is not None,
            "model_type": self.model_type,
        }
        self.status = "ok"
        # Build a human-readable reason string
        if mt == "auto":
            self.reason = "homography+rbf" if self._use_H else "brown+rbf"
        elif mt == "polynomial":
            self.reason = "polynomial"
        elif mt == "affine":
            self.reason = "affine"
        elif mt == "homography":
            self.reason = "homography"
        elif mt == "brown":
            self.reason = "brown" if not self._use_H else "brown→homography_fallback"
        elif mt == "brown_rbf":
            base = "brown" if not self._use_H else "brown→homography_fallback"
            self.reason = f"{base}+rbf" if self._rbf is not None else base

    # ── Model-specific fitting helpers ────────────────────────────────────────

    def _try_brown(self, world_norm: np.ndarray, pixel_uv: np.ndarray, n: int) -> bool:
        """Attempt Brown model via cv2.calibrateCamera. Returns True on success."""
        obj_pts = np.zeros((n, 1, 3), dtype=np.float32)
        obj_pts[:, 0, :2] = world_norm.astype(np.float32)
        img_pts = pixel_uv[:, np.newaxis, :].astype(np.float32)

        px_span = pixel_uv.max(axis=0) - pixel_uv.min(axis=0)
        w_span = world_norm.max(axis=0) - world_norm.min(axis=0)
        f = float(np.max(px_span) / max(float(np.max(w_span)), 1e-6))
        cx, cy = float(pixel_uv[:, 0].mean()), float(pixel_uv[:, 1].mean())
        K_init = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist_init = np.zeros((5, 1), dtype=np.float64)

        iw = max(int(pixel_uv[:, 0].max()) + 32, 64)
        ih = max(int(pixel_uv[:, 1].max()) + 32, 64)

        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
        if n < 12:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3

        try:
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                [obj_pts], [img_pts], (iw, ih), K_init, dist_init,
                flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-8),
            )
        except Exception:
            return False

        if not math.isfinite(rms) or rms > 1e5:
            return False

        self._K = K
        self._dist = dist
        self._rvec = rvecs[0]
        self._tvec = tvecs[0]
        return True

    def _fit_homography(self, world_norm: np.ndarray, pixel_uv: np.ndarray) -> None:
        src = world_norm.astype(np.float32)
        dst = pixel_uv.astype(np.float32)
        for method, thr in [(cv2.RANSAC, 10.0), (cv2.LMEDS, 0.0), (0, 0.0)]:
            kw = {"ransacReprojThreshold": thr} if method == cv2.RANSAC else {}
            H, _ = cv2.findHomography(src, dst, method, **kw)
            if H is not None:
                self._H = H
                return
        raise RuntimeError("findHomography failed for all methods")

    def _fit_homography_fallback(self, world_norm: np.ndarray, pixel_uv: np.ndarray) -> None:
        """Fit a homography alongside Brown for extrapolation outside the calibration domain."""
        src = world_norm.astype(np.float32)
        dst = pixel_uv.astype(np.float32)
        for method, thr in [(cv2.RANSAC, 10.0), (cv2.LMEDS, 0.0), (0, 0.0)]:
            kw = {"ransacReprojThreshold": thr} if method == cv2.RANSAC else {}
            H, _ = cv2.findHomography(src, dst, method, **kw)
            if H is not None:
                self._H_fallback = H
                return

    def _fit_affine(self, world_norm: np.ndarray, pixel_uv: np.ndarray) -> None:
        """Fit a 2×3 affine transform and store it as a 3×3 homogeneous matrix."""
        src = world_norm.astype(np.float32)
        dst = pixel_uv.astype(np.float32)
        # estimateAffine2D needs at least 3 non-collinear points (4 recommended)
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
        if M is None:
            # Fall back to least-squares affine via findHomography with constrained DOF
            # by solving the 2×3 system directly
            n = len(world_norm)
            A = np.hstack([world_norm, np.ones((n, 1))]).astype(np.float64)
            coef, _, _, _ = np.linalg.lstsq(A, pixel_uv, rcond=None)  # (3,2)
            M = coef.T.astype(np.float32)  # (2,3)
        # Embed 2×3 into 3×3
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M.astype(np.float64)
        self._H = H

    def _fit_poly(self, world_norm: np.ndarray, pixel_uv: np.ndarray) -> None:
        """Fit a quadratic polynomial: features [1, x, y, x², xy, y²]."""
        x, y = world_norm[:, 0], world_norm[:, 1]
        A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])  # (n, 6)
        coef, _, _, _ = np.linalg.lstsq(A, pixel_uv, rcond=None)  # (6, 2)
        self._poly_coef = coef.T  # (2, 6): row 0 → u, row 1 → v

    def _fit_rbf(self, world_norm: np.ndarray, pixel_uv: np.ndarray) -> None:
        pred = self._global_predict(world_norm)
        residuals = pixel_uv - pred
        X, Y = world_norm, residuals
        if len(X) > self.max_control_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), self.max_control_points, replace=False)
            X, Y = X[idx], Y[idx]
        try:
            self._rbf = _RBFInterpolator(
                X, Y,
                kernel=self._rbf_kernel,
                smoothing=self.smoothing,
                epsilon=self.epsilon,
            )
            self._rbf_X, self._rbf_Y = X, Y
        except Exception:
            self._rbf = None

    # ── Cross-validation ─────────────────────────────────────────────────────

    def _fit_global_only(
        self,
        world_norm: np.ndarray,
        pixel_uv: np.ndarray,
        n: int,
    ) -> bool:
        """Fit global model (Brown or homography) only — no RBF. Returns True on success."""
        if n >= 8 and self._try_brown(world_norm, pixel_uv, n):
            self._use_H = False
            self._rbf = None
            return True
        try:
            self._fit_homography(world_norm, pixel_uv)
            self._use_H = True
            self._rbf = None
            return True
        except RuntimeError:
            return False

    @staticmethod
    def global_loo_cv_rms(samples: list[dict]) -> float:
        """Leave-one-out cross-validation RMS using the global model only.

        Fits the global model (Brown or homography) N times, each time holding
        out one point, predicts it, and returns the RMS of those held-out errors.
        The RBF is deliberately excluded so this measures generalisation of the
        underlying geometric model without interpolation overfitting.

        Returns inf when there are not enough samples for a stable LOO fit.
        """
        n = len(samples)
        if n < 5:
            return float("inf")

        world_xy = np.array([[s["world_x"], s["world_y"]] for s in samples], dtype=np.float64)
        pixel_uv = np.array([[s["pixel_u"], s["pixel_v"]] for s in samples], dtype=np.float64)

        center = world_xy.mean(axis=0)
        scale = float(max(np.abs(world_xy - center).max(), 1.0))
        world_norm = (world_xy - center) / scale

        errors: list[float] = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            wn_train = world_norm[mask]
            pv_train = pixel_uv[mask]
            n_train = n - 1

            tmp = CameraModelMapper.__new__(CameraModelMapper)
            tmp._K = None
            tmp._dist = None
            tmp._rvec = None
            tmp._tvec = None
            tmp._H = None
            tmp._use_H = True
            tmp._rbf = None
            tmp._poly_coef = None
            tmp.model_type = "auto"
            tmp._world_center = center
            tmp._world_scale = scale

            if not tmp._fit_global_only(wn_train, pv_train, n_train):
                return float("inf")

            pred = tmp._global_predict(world_norm[[i]])
            err = float(np.sqrt(((pixel_uv[[i]] - pred) ** 2).sum()))
            errors.append(err)

        return float(np.sqrt(np.mean(np.array(errors) ** 2)))

    # ── Prediction ───────────────────────────────────────────────────────────

    def _global_predict(self, world_norm: np.ndarray) -> np.ndarray:
        if self._poly_coef is not None:
            return self._poly_predict(world_norm)
        if self._use_H or self._K is None:
            pts = world_norm[:, np.newaxis, :].astype(np.float32)
            out = cv2.perspectiveTransform(pts, self._H)
            return out[:, 0, :].astype(np.float64)
        pts3d = np.zeros((len(world_norm), 1, 3), dtype=np.float32)
        pts3d[:, 0, :2] = world_norm.astype(np.float32)
        proj, _ = cv2.projectPoints(pts3d, self._rvec, self._tvec, self._K, self._dist)
        return proj[:, 0, :].astype(np.float64)

    def _poly_predict(self, world_norm: np.ndarray) -> np.ndarray:
        x, y = world_norm[:, 0], world_norm[:, 1]
        A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
        return (self._poly_coef @ A.T).T  # (n, 2)

    def _rbf_predict(self, world_norm: np.ndarray) -> np.ndarray:
        if self._rbf is None:
            return np.zeros((len(world_norm), 2), dtype=np.float64)
        return self._rbf(world_norm)

    def _in_calibration_domain(self, world_norm: np.ndarray) -> np.ndarray:
        """Boolean mask: True for points inside the calibration bbox + margin."""
        if self._calib_norm_min is None:
            return np.ones(len(world_norm), dtype=bool)
        extent = self._calib_norm_max - self._calib_norm_min
        margin = extent * self._DOMAIN_MARGIN
        lo = self._calib_norm_min - margin
        hi = self._calib_norm_max + margin
        return ((world_norm >= lo) & (world_norm <= hi)).all(axis=1)

    def _fallback_predict(self, world_norm: np.ndarray) -> np.ndarray:
        """Project via homography fallback (no distortion, no RBF)."""
        H = self._H_fallback if self._H_fallback is not None else self._H
        pts = world_norm[:, np.newaxis, :].astype(np.float32)
        out = cv2.perspectiveTransform(pts, H)
        return out[:, 0, :].astype(np.float64)

    def project(self, world_xy: np.ndarray) -> np.ndarray:
        """Project world coords (N, 2) in mm → pixel coords (N, 2)."""
        if self._world_center is None:
            raise RuntimeError(f"CameraModelMapper({self.camera!r}) not fitted")
        world_xy = np.atleast_2d(np.asarray(world_xy, dtype=np.float64))
        norm = (world_xy - self._world_center) / self._world_scale

        # Polynomial has no fallback concept — always uses poly predict
        if self._poly_coef is not None:
            return self._poly_predict(norm)

        # When already using homography/affine as primary, no fallback needed.
        if self._use_H or self._H_fallback is None:
            pred = self._global_predict(norm)
            if self._rbf is not None:
                pred = pred + self._rbf_predict(norm)
            return pred

        # Brown model: route out-of-domain points through the homography
        # fallback to avoid distortion-polynomial blow-up.
        in_dom = self._in_calibration_domain(norm)

        if in_dom.all():
            pred = self._global_predict(norm)
            if self._rbf is not None:
                pred = pred + self._rbf_predict(norm)
            return pred

        pred = np.empty((len(norm), 2), dtype=np.float64)

        if in_dom.any():
            p = self._global_predict(norm[in_dom])
            if self._rbf is not None:
                p = p + self._rbf_predict(norm[in_dom])
            pred[in_dom] = p

        pred[~in_dom] = self._fallback_predict(norm[~in_dom])
        return pred

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        data: dict[str, Any] = {
            "camera": self.camera,
            "model_type": self.model_type,
            "status": self.status,
            "reason": self.reason,
            "fit_meta": self.fit_meta,
            "world_center": self._world_center.tolist() if self._world_center is not None else None,
            "world_scale": self._world_scale,
            "use_homography": self._use_H,
        }
        if self._use_H and self._H is not None:
            data["H"] = self._H.tolist()
        if not self._use_H and self._K is not None:
            data.update({
                "K": self._K.tolist(),
                "dist": self._dist.tolist(),
                "rvec": self._rvec.tolist(),
                "tvec": self._tvec.tolist(),
            })
        if self._H_fallback is not None:
            data["H_fallback"] = self._H_fallback.tolist()
        if self._calib_norm_min is not None:
            data["calib_norm_min"] = self._calib_norm_min.tolist()
            data["calib_norm_max"] = self._calib_norm_max.tolist()
        if self._poly_coef is not None:
            data["poly_coef"] = self._poly_coef.tolist()
        if self._rbf_X is not None:
            data.update({
                "rbf_X": self._rbf_X.tolist(),
                "rbf_Y": self._rbf_Y.tolist(),
                "rbf_kernel": self._rbf_kernel,
                "rbf_smoothing": self.smoothing,
                "rbf_epsilon": self.epsilon,
            })
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CameraModelMapper":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mt = data.get("model_type", "auto")
        m = cls(data["camera"], model_type=mt)
        m.status = data.get("status", "ok")
        m.reason = data.get("reason", "")
        m.fit_meta = data.get("fit_meta", {})
        c = data.get("world_center")
        m._world_center = np.array(c, dtype=np.float64) if c is not None else None
        m._world_scale = float(data.get("world_scale", 1.0))
        m._use_H = data.get("use_homography", True)
        if m._use_H and "H" in data:
            m._H = np.array(data["H"], dtype=np.float64)
        elif not m._use_H and "K" in data:
            m._K = np.array(data["K"], dtype=np.float64)
            m._dist = np.array(data["dist"], dtype=np.float64)
            m._rvec = np.array(data["rvec"], dtype=np.float64)
            m._tvec = np.array(data["tvec"], dtype=np.float64)
        if "H_fallback" in data:
            m._H_fallback = np.array(data["H_fallback"], dtype=np.float64)
        if "calib_norm_min" in data:
            m._calib_norm_min = np.array(data["calib_norm_min"], dtype=np.float64)
            m._calib_norm_max = np.array(data["calib_norm_max"], dtype=np.float64)
        if "poly_coef" in data:
            m._poly_coef = np.array(data["poly_coef"], dtype=np.float64)
        if "rbf_X" in data and _HAS_SCIPY:
            X = np.array(data["rbf_X"], dtype=np.float64)
            Y = np.array(data["rbf_Y"], dtype=np.float64)
            m._rbf_X, m._rbf_Y = X, Y
            m._rbf_kernel = data.get("rbf_kernel", "thin_plate_spline")
            m.smoothing = float(data.get("rbf_smoothing", 0.0))
            m.epsilon = float(data.get("rbf_epsilon", 1.0))
            try:
                m._rbf = _RBFInterpolator(
                    X, Y,
                    kernel=m._rbf_kernel,
                    smoothing=m.smoothing,
                    epsilon=m.epsilon,
                )
            except Exception:
                pass
        # Backward compat: reconstruct fallback from RBF control points
        # when loading a model saved before the fallback was introduced.
        if not m._use_H and m._H_fallback is None and m._rbf_X is not None and m._rbf_Y is not None:
            pixel_at_ctrl = m._global_predict(m._rbf_X) + m._rbf_Y
            src = m._rbf_X.astype(np.float32)
            dst = pixel_at_ctrl.astype(np.float32)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 10.0)
            if H is not None:
                m._H_fallback = H
        if m._calib_norm_min is None and m._rbf_X is not None:
            m._calib_norm_min = m._rbf_X.min(axis=0).copy()
            m._calib_norm_max = m._rbf_X.max(axis=0).copy()
        return m
