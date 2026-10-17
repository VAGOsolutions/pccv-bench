"""Derive PCCV-Bench observation noise from frozen detector results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pccv_bench.noise import NoiseParams


@dataclass(frozen=True)
class FittedNoise:
    """Noise parameters and the detector evidence used to derive them."""

    params: NoiseParams
    provenance: dict[str, Any]


def fit_noise_from_frozen_evaluation(evaluation: Mapping[str, Any]) -> FittedNoise:
    """Fit observable noise rates from one frozen detector evaluation.

    The benchmark's miss mechanism represents a ground-truth group with no
    usable detection, so its rate is the frozen evaluation's one-minus-recall.
    Its false-positive mechanism adds at most one spurious box per image, so
    the empty-image false-positive rate is the directly comparable estimate.

    Args:
        evaluation: Parsed frozen-evaluation JSON with ``test_global`` rates.

    Returns:
        Fitted parameters and traceable source metadata.

    Raises:
        ValueError: If the expected frozen evaluation fields are absent or out
            of the inclusive probability range.
    """
    test_global = evaluation.get("test_global")
    if not isinstance(test_global, Mapping):
        raise TypeError("frozen evaluation must contain a test_global mapping")

    recall = _probability(test_global.get("recall"), "test_global.recall")
    empty_fp_rate = _probability(
        test_global.get("empty_fp_rate"), "test_global.empty_fp_rate"
    )
    model = evaluation.get("model")
    threshold = evaluation.get("frozen_threshold")
    if not isinstance(model, str) or not model:
        raise ValueError("frozen evaluation must contain a non-empty model name")
    if not isinstance(threshold, (int, float)):
        raise TypeError("frozen evaluation must contain frozen_threshold")

    return FittedNoise(
        params=NoiseParams(
            version="v1.0-fitted",
            miss_rate=1.0 - recall,
            fp_rate=empty_fp_rate,
        ),
        provenance={
            "noise_version": "v1.0-fitted",
            "source_model": model,
            "frozen_threshold": float(threshold),
            "source_recall": recall,
            "source_empty_fp_rate": empty_fp_rate,
            "fit_method": "miss_rate=1-recall; fp_rate=empty_image_fp_rate",
        },
    )


def _probability(value: object, field_name: str) -> float:
    """Validate and normalize one probability value."""
    if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be a probability, got {value!r}")
    return float(value)
