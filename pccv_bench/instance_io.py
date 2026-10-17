"""JSON (de)serialization for ``pccv_bench.core.harness.Instance``.

Kept out of ``pccv_bench.core.harness`` itself so that module stays a pure
matcher-invocation layer with no I/O — this is PCCV-Bench-specific plumbing
for the CLI's ``generate`` -> ``ablate`` split.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.core.harness import Instance


def _json_default(obj):
    """Handle numpy scalars leaking through the (shared, not-forked) injector
    code — e.g. ``position_shift_*``'s ``severity``/``meta['shift_mm']`` are
    computed from ``rng.uniform(...)``, and ``round(np.float64, n)`` stays a
    ``np.float64`` rather than becoming a plain Python float. Never surfaced
    before because nothing previously round-tripped an ``Instance`` through
    JSON; fixed here rather than in ``eval/injection/inject.py`` so that
    shared, already-tested code stays unforked.
    """
    if hasattr(obj, "item"):  # numpy scalar types all implement .item()
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def instance_to_dict(inst: Instance) -> dict:
    d = asdict(inst)
    d["image_wh"] = list(d["image_wh"])  # tuples aren't valid JSON
    return d


def instance_from_dict(d: dict) -> Instance:
    d = dict(d)
    d["image_wh"] = tuple(d["image_wh"])
    return Instance(**d)


def save_instances(path: Path, instances: list[Instance], meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta or {},
        "n_instances": len(instances),
        "instances": [instance_to_dict(i) for i in instances],
    }
    path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")


def load_instances(path: Path) -> tuple[list[Instance], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = [instance_from_dict(d) for d in payload["instances"]]
    return instances, payload.get("meta", {})
