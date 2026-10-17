from __future__ import annotations

import math
import numpy as np


def rect_corners_world(cx: float, cy: float, sx: float, sy: float, r_deg: float) -> np.ndarray:
    """Return (4, 2) float64 array of world-space corners (TL, TR, BR, BL, CW when y-up)."""
    r = math.radians(r_deg)
    c, s = math.cos(r), math.sin(r)
    hw, hh = sx / 2.0, sy / 2.0
    local = np.array([[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy], dtype=np.float64)
