"""Projection of world-space record parts into image polygons.

Reuses the production projection so that a perturbed or synthetic record is
structurally identical to one the production pipeline would build."""

from __future__ import annotations

from pccv_bench.core.camera_model import CameraModelMapper
from pccv_bench.core.matching import SAW_X_OFFSET, project_parts_to_pixels_with_ids


def project_world_parts(
    parts_world: list[dict],
    mapper: CameraModelMapper,
    iw: int,
    ih: int,
) -> list[dict]:
    """Project world-space parts into the flat record format used everywhere.

    Reuses the production projection
    (:func:`~pccv_bench.core.matching.project_parts_to_pixels_with_ids`)
    and mirrors :func:`~pccv_bench.core.matching.build_gql_flat`'s
    flattening, so a perturbed record is structurally indistinguishable from one
    the production pipeline would build.
    """
    shapes = project_parts_to_pixels_with_ids(parts_world, mapper, iw, ih, SAW_X_OFFSET)
    by_id: dict[str, dict] = {}
    for p in parts_world:
        by_id.setdefault(str(p.get("id", "")), p)
    flat: list[dict] = []
    for kind in ("saw", "pla"):
        for kind_idx, entry in enumerate(shapes[kind]):
            pid = entry["id"]
            flat.append(
                {
                    "id": pid,
                    "kind": kind,
                    "kind_index": kind_idx,
                    "polygon": entry["polygon"],
                    "world": by_id.get(pid),
                }
            )
    return flat
