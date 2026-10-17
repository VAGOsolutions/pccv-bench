"""Tests for PCCV-Bench release provenance utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_fit_noise_uses_frozen_detector_error_rates() -> None:
    """The fitted miss and empty-frame FP rates must come from the input."""
    from pccv_bench.fit_noise import fit_noise_from_frozen_evaluation

    evaluation = {
        "model": "example",
        "frozen_threshold": 0.4,
        "test_global": {
            "recall": 0.76,
            "empty_fp_rate": 0.03,
        },
    }

    fitted = fit_noise_from_frozen_evaluation(evaluation)

    assert fitted.params.miss_rate == pytest.approx(0.24)
    assert fitted.params.fp_rate == pytest.approx(0.03)
    assert fitted.params.to_dict()["version"] == "v1.0-fitted"
    assert fitted.provenance["source_model"] == "example"
    assert fitted.provenance["frozen_threshold"] == 0.4


def test_spearman_correlation_ranks_shared_arms() -> None:
    """Only shared arms participate in the reported rank correlation."""
    from pccv_bench.release_validation import rank_correlation

    real = {"geometry": 0.2, "+offset": 0.4, "+twoway": 0.8}
    synthetic = {"geometry": 0.1, "+offset": 0.3, "+twoway": 0.9, "extra": 1.0}

    result = rank_correlation(real, synthetic)

    assert result.shared_arms == ("+offset", "+twoway", "geometry")
    assert result.spearman_rho == pytest.approx(1.0)


def test_configured_noise_version_overrides_the_development_default() -> None:
    """Released v1.0 output must not be labelled as the v0.1 default."""
    from pccv_bench.cli import noise_version_for_config

    assert noise_version_for_config({"noise_version": "v1.0-fitted"}) == "v1.0-fitted"


def test_git_revision_marks_a_dirty_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    from pccv_bench import cli

    responses = iter(["abc123\n", " M pccv_bench/cli.py\n"])
    monkeypatch.setattr(
        cli.subprocess, "check_output", lambda *args, **kwargs: next(responses)
    )

    assert cli._git_rev() == "abc123-dirty"
