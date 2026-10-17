"""The released seed-42 instance set must be what the generator produces."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pccv_bench.core.manifest import manifest_checksum
from pccv_bench.instance_io import load_instances

_ROOT = Path(__file__).resolve().parent.parent.parent
PINNED_INSTANCES = _ROOT / "instances" / "medium_v1_2_fitted_seed42.json"
PINNED_CHECKSUM = "8b2bee72cb8c01bda429156583b8aaac49956b5d5bb990e480ff3be0f76d3986"

#: Regenerated geometry must agree with the pinned set to this many pixels or
#: millimetres. OpenCV 4.5 to 4.13 agree to about 1e-4 in the fitted camera model
#: of one camera; OpenCV 4.14 and 5.x change it by about 0.2 px, which this test rejects.
REGENERATION_TOLERANCE = 1e-3


@pytest.mark.skipif(not PINNED_INSTANCES.exists(), reason="pinned instance set not present")
def test_pinned_instance_set_has_the_documented_checksum() -> None:
    instances, meta = load_instances(PINNED_INSTANCES)
    assert manifest_checksum(instances) == PINNED_CHECKSUM
    assert meta["instance_checksum"] == PINNED_CHECKSUM


def _largest_difference(expected, actual) -> float:
    """Largest absolute float difference, or infinity if any other value differs."""
    if isinstance(expected, float) and isinstance(actual, float):
        return abs(expected - actual)
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return float("inf")
        return max((_largest_difference(expected[k], actual[k]) for k in expected), default=0.0)
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return float("inf")
        return max((_largest_difference(e, a) for e, a in zip(expected, actual)), default=0.0)
    return 0.0 if expected == actual else float("inf")


@pytest.mark.skipif(not PINNED_INSTANCES.exists(), reason="pinned instance set not present")
def test_regenerating_seed_42_reproduces_the_pinned_set(tmp_path: Path) -> None:
    out = tmp_path / "instances.json"
    subprocess.run(
        [
            sys.executable, "-m", "pccv_bench.cli", "generate",
            "--config", str(_ROOT / "pccv_bench" / "configs" / "medium_v1_2_fitted.yaml"),
            "--seed", "42", "--out", str(out),
        ],
        cwd=_ROOT, check=True, capture_output=True,
    )
    pinned = json.loads(PINNED_INSTANCES.read_text(encoding="utf-8"))["instances"]
    regenerated = json.loads(out.read_text(encoding="utf-8"))["instances"]
    assert [i["instance_id"] for i in regenerated] == [i["instance_id"] for i in pinned]
    assert _largest_difference(pinned, regenerated) <= REGENERATION_TOLERANCE
