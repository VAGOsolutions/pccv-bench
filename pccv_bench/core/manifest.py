"""Content checksums for a generated instance set.

A reported rate is only reproducible if the instances behind it are.  The
generation path is seeded (:mod:`pccv_bench.core.seeding`), but nothing detected
a *regenerated* benchmark that came out different — a changed injector, a
re-exported record, or a pool rebuilt from different frames would all produce
plausible numbers against a silently different set.  This module pins the set:
every instance gets a content digest, the digests fix one checksum for the whole
file, and evaluation verifies that checksum before it scores anything.

The digest covers the label-bearing content only — identity, class, ground
truth, geometry and provenance.  It deliberately excludes nothing that would
change a verdict."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Sequence


#: Bump whenever the generation semantics change, so an old manifest is
#: recognisably not comparable rather than merely mismatched.
GENERATOR_VERSION = "2026-09-08.outcome-model"


#: Coordinates are rounded before hashing so a harmless last-bit difference in
#: an accelerated BLAS does not read as a changed benchmark.  Far finer than any
#: geometric tolerance in the matcher.
_COORD_DP = 6


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, _COORD_DP)
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def _part_ids(parts: Sequence[dict] | None) -> list[str]:
    return [str(p.get("id", "")) for p in (parts or [])]


def _geometry(parts: Sequence[dict] | None) -> list[Any]:
    return [_round(p.get("polygon")) for p in (parts or [])]


def instance_digest(inst) -> str:
    """SHA-256 over one instance's label-bearing content."""
    payload = {
        "instance_id": inst.instance_id,
        "camera": inst.camera,
        "run_id": inst.run_id,
        "injection_class": inst.injection_class,
        "fault_type": inst.fault_type,
        "expected_action": inst.expected_action,
        "severity": _round(float(inst.severity)),
        "image_wh": list(inst.image_wh),
        "detections": [_round(d.get("polygon")) for d in inst.detections],
        "record_part_ids": _part_ids(inst.record_parts),
        "record_geometry": _geometry(inst.record_parts),
        "next_record_part_ids": _part_ids(inst.next_record_parts),
        "next_record_geometry": _geometry(inst.next_record_parts),
        "donor_run": (inst.meta or {}).get("donor_run_id"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def manifest_checksum(instances: Iterable) -> str:
    """One checksum for the whole set, independent of generation order."""
    digests = sorted(instance_digest(i) for i in instances)
    return sha256("".join(digests).encode("ascii")).hexdigest()


def build_manifest(
    instances: Iterable,
    generator_version: str = GENERATOR_VERSION,
    extra: dict | None = None,
) -> dict:
    instances = list(instances)
    rows = [
        {
            "instance_id": i.instance_id,
            "camera": i.camera,
            "injection_class": i.injection_class,
            "expected_action": i.expected_action,
            "source_run": i.run_id,
            "donor_run": (i.meta or {}).get("donor_run_id"),
            "current_record_ids": _part_ids(i.record_parts),
            "next_record_ids": _part_ids(i.next_record_parts),
            "sha256": instance_digest(i),
        }
        for i in instances
    ]
    return {
        "generator_version": generator_version,
        "n_instances": len(instances),
        "checksum": manifest_checksum(instances),
        **(extra or {}),
        "rows": rows,
    }


def save_manifest(
    path: Path,
    instances: Iterable,
    generator_version: str = GENERATOR_VERSION,
    extra: dict | None = None,
) -> dict:
    manifest = build_manifest(instances, generator_version, extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
