"""Compare public PCCV-Bench ablations with the companion real-data study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isnan, sqrt


@dataclass(frozen=True)
class RankCorrelation:
    """Spearman comparison over the ablation arms common to two studies."""

    shared_arms: tuple[str, ...]
    spearman_rho: float


def rank_correlation(
    real_rates: Mapping[str, float], synthetic_rates: Mapping[str, float]
) -> RankCorrelation:
    """Calculate Spearman's rho using only ablation arms shared by both inputs.

    Args:
        real_rates: Real-study detection rate by arm.
        synthetic_rates: PCCV-Bench detection rate by arm.

    Returns:
        The sorted shared arm names and their Spearman rank correlation.

    Raises:
        ValueError: If fewer than two arms are shared or a rate is invalid.
    """
    shared_arms = tuple(sorted(set(real_rates) & set(synthetic_rates)))
    if len(shared_arms) < 2:
        raise ValueError("at least two shared arms are required for rank correlation")

    real_values = [
        _probability(real_rates[arm], f"real_rates[{arm!r}]") for arm in shared_arms
    ]
    synthetic_values = [
        _probability(synthetic_rates[arm], f"synthetic_rates[{arm!r}]")
        for arm in shared_arms
    ]
    rho = _pearson_correlation(
        _average_ranks(real_values), _average_ranks(synthetic_values)
    )
    if isnan(rho):
        raise ValueError(
            "rank correlation is undefined because one study has no rank variation"
        )
    return RankCorrelation(shared_arms=shared_arms, spearman_rho=float(rho))


def _probability(value: object, field_name: str) -> float:
    """Validate one rate supplied to the comparison."""
    if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be a probability, got {value!r}")
    return float(value)


def _average_ranks(values: list[float]) -> list[float]:
    """Return one-based ranks, assigning tied values their average rank."""
    ordered_indices = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered_indices):
        end = start + 1
        while (
            end < len(ordered_indices)
            and values[ordered_indices[end]] == values[ordered_indices[start]]
        ):
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in ordered_indices[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson_correlation(left: list[float], right: list[float]) -> float:
    """Calculate Pearson correlation for two equally sized numeric vectors."""
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return numerator / (left_norm * right_norm)
