"""Synthetic world-space scene generation.

Reproduces the *structure* of real per-camera part records — the group/box-count
distribution, tiling pitch, rotation bias — without reproducing any real
geometry. Coordinates, table extents and part sizes below are plausible
synthetic values in the same units and order of magnitude as the real system
(millimeters, part footprints ~60-100mm x ~200-300mm), not measurements of the
real installation.

Scope constraint: parts are instances of **one class**. A
generated part carries only the fields load-bearing for projection and
matching — ``x, y, sizeX, sizeY, rotation, table_kind`` plus an ``id`` — and
nothing that could be read as a type/category/product attribute.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Per-camera box-count (= detected-group-count) distribution, measured
#: directly from all 3,000 real coarse labels (one line per detected group).
#: Reproduced per camera rather than pooled —
#: empty-table share alone spans 19% (cam3) to 61% (cam4/cam6), and
#: the max group count spans 2 to 7 — because that 3x spread across six views
#: of one process is a real property of the installation the benchmark exists
#: to expose; a single pooled shape would erase it.
BOX_COUNT_DIST: dict[str, dict[int, int]] = {
    "cam1": {0: 162, 1: 159, 2: 88, 3: 37, 4: 50, 5: 4},
    "cam2":   {0: 251, 1: 210, 2: 27, 3: 9, 4: 3},
    "cam3": {0: 96, 1: 169, 2: 126, 3: 65, 4: 21, 5: 19, 6: 4},
    "cam4":   {0: 306, 1: 166, 2: 23, 3: 5},
    "cam5": {0: 160, 1: 186, 2: 80, 3: 25, 4: 10, 5: 11, 6: 14, 7: 14},
    "cam6":   {0: 307, 1: 149, 2: 44},
}

CAMERAS: tuple[str, ...] = tuple(BOX_COUNT_DIST)

#: Synthetic table extent, world mm, centred at the origin — a fixed camera's
#: own local view of one station (matching the real six-camera setup, where
#: each camera watches its own table, not the whole cell), not the
#: real span of any station. Sized jointly with ``pccv_bench.cameras``'
#: synthetic camera geometry so the whole extent projects inside a 1920x1024
#: frame at a realistic box-pixel scale — see that module's docstring for the
#: numeric search behind these two files' paired constants.
TABLE_HALF_X = 420.0
TABLE_HALF_Y = 900.0

#: Example pitches taken from the installation (a ~80mm part footprint tiled
#: at a ~240mm pitch within a group) — used as the mean of a jittered draw so
#: generated groups aren't perfectly regular.
PART_PITCH_X_MEAN = 80.0
PART_SIZE_X_RANGE = (55.0, 105.0)
PART_SIZE_Y_RANGE = (200.0, 300.0)

#: Fraction of parts placed axis-aligned (r in {0, 180}) vs. off-axis, matching
#: the real sample's rotation bias.
AXIS_ALIGNED_FRACTION = 0.85

#: Minimum centre-to-centre separation between two groups' bounding boxes,
#: mm. Occasionally sampled small enough to produce touching/near-touching
#: groups — a known real failure mode the matcher's greedy subset search must
#: cope with — rather than always keeping groups comfortably apart.
GROUP_SEPARATION_RANGE = (20.0, 400.0)


@dataclass
class ScenePart:
    id: str
    x: float
    y: float
    sizeX: float
    sizeY: float
    rotation: float
    table_kind: str
    group_id: int

    def to_world_dict(self) -> dict:
        """The flat world-space dict shape ``eval.injection`` consumes
        everywhere (``project_world_parts``, the injectors' ``parts_world``)."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "sizeX": self.sizeX,
            "sizeY": self.sizeY,
            "rotation": self.rotation,
            "table_kind": self.table_kind,
            # Schema realism only — never load-bearing.
            "z": 35.0,
            "mass": round(float(self.sizeX * self.sizeY * 35.0 * 7.85e-6), 3),
        }


@dataclass
class Scene:
    camera: str
    parts: list[ScenePart] = field(default_factory=list)
    n_groups: int = 0

    def parts_world(self) -> list[dict]:
        return [p.to_world_dict() for p in self.parts]

    def group_ids(self) -> list[int]:
        return sorted({p.group_id for p in self.parts})

    def parts_in_group(self, group_id: int) -> list[ScenePart]:
        return [p for p in self.parts if p.group_id == group_id]


def sample_box_count(rng: np.random.Generator, camera: str) -> int:
    """Draw a group count from *camera*'s measured distribution."""
    dist = BOX_COUNT_DIST[camera]
    values = list(dist.keys())
    weights = np.array(list(dist.values()), dtype=np.float64)
    return int(rng.choice(values, p=weights / weights.sum()))


def sample_parts_per_group(rng: np.random.Generator) -> int:
    """Mostly single-part groups, occasionally 2-3 parts tiled together.

    A group/parts-per-group draw is used; most real
    coarse-label groups reduce to a single bounding box, so a group is most
    often exactly one part, with a shrinking tail up to 3 so the matcher's
    greedy multi-part subset search (``best_aabb_combo_for_box``) is actually
    exercised by some fraction of instances.
    """
    return int(rng.choice([1, 2, 3], p=[0.70, 0.22, 0.08]))


def sample_rotation(rng: np.random.Generator) -> float:
    if rng.random() < AXIS_ALIGNED_FRACTION:
        return float(rng.choice([0.0, 180.0]))
    return float(rng.uniform(-45.0, 45.0))


def _group_footprint(n_parts: int, pitch_x: float, size_x: float, size_y: float) -> tuple[float, float]:
    width = size_x + (n_parts - 1) * pitch_x
    return width, size_y


def _sample_group_center(
    rng: np.random.Generator,
    placed: list[tuple[float, float, float, float]],  # (cx, cy, half_w, half_h)
    half_w: float,
    half_h: float,
    max_tries: int = 40,
) -> tuple[float, float]:
    """Place a new group's centre, retrying to honour the separation draw."""
    for _ in range(max_tries):
        cx = rng.uniform(-TABLE_HALF_X + half_w, TABLE_HALF_X - half_w)
        cy = rng.uniform(-TABLE_HALF_Y + half_h, TABLE_HALF_Y - half_h)
        gap = rng.uniform(*GROUP_SEPARATION_RANGE)
        ok = True
        for pcx, pcy, phw, phh in placed:
            # Required clearance between this group's bbox and an existing one.
            dx = max(0.0, abs(cx - pcx) - (half_w + phw))
            dy = max(0.0, abs(cy - pcy) - (half_h + phh))
            if dx == 0.0 and dy == 0.0:
                continue  # already overlapping in projection — reject below
            if (dx * dx + dy * dy) ** 0.5 < gap:
                ok = False
                break
        if ok:
            return cx, cy
    # Give up honouring separation on the last try rather than looping forever;
    # a rare accidental overlap is itself a realistic edge case, not a bug.
    return (
        rng.uniform(-TABLE_HALF_X + half_w, TABLE_HALF_X - half_w),
        rng.uniform(-TABLE_HALF_Y + half_h, TABLE_HALF_Y - half_h),
    )


def generate_scene(
    rng: np.random.Generator,
    camera: str,
    table_kind: str = "saw",
    n_groups: int | None = None,
) -> Scene:
    """Generate one synthetic scene for *camera*.

    ``n_groups=None`` (the default) draws from the camera's measured
    box-count distribution; pass an explicit value to force a scene (e.g. for
    an analytic test that needs exactly one group).
    """
    if camera not in BOX_COUNT_DIST:
        raise ValueError(f"unknown synthetic camera {camera!r}; choices: {CAMERAS}")
    if n_groups is None:
        n_groups = sample_box_count(rng, camera)

    placed: list[tuple[float, float, float, float]] = []
    parts: list[ScenePart] = []
    part_seq = 0

    for gid in range(n_groups):
        n_parts = sample_parts_per_group(rng)
        pitch_x = float(rng.normal(PART_PITCH_X_MEAN, 8.0))
        size_x = float(rng.uniform(*PART_SIZE_X_RANGE))
        size_y = float(rng.uniform(*PART_SIZE_Y_RANGE))
        width, height = _group_footprint(n_parts, pitch_x, size_x, size_y)
        half_w, half_h = width / 2.0, height / 2.0
        cx, cy = _sample_group_center(rng, placed, half_w, half_h)
        placed.append((cx, cy, half_w, half_h))

        rotation = sample_rotation(rng)
        start_x = cx - (n_parts - 1) * pitch_x / 2.0
        for k in range(n_parts):
            part_seq += 1
            parts.append(
                ScenePart(
                    id=f"synth#{part_seq:06d}",
                    x=start_x + k * pitch_x,
                    y=cy,
                    sizeX=size_x,
                    sizeY=size_y,
                    rotation=rotation,
                    table_kind=table_kind,
                    group_id=gid,
                )
            )

    return Scene(camera=camera, parts=parts, n_groups=n_groups)
