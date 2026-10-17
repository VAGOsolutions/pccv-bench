"""Unit tests for pccv_bench.agreement_stats (E4/E6 rank-agreement machinery)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.agreement_stats import (
    _rank_rows,
    classify_reading_rule,
    joint_bootstrap_rho_ci,
    permutation_pvalue,
    rank_agreement_point,
)


def test_perfect_agreement_gives_rho_one():
    x = list(range(13))
    stats = rank_agreement_point(x, x)
    assert stats["spearman_rho"] == 1.0
    assert stats["kendall_tau_b"] > 0.999


def test_perfect_anti_agreement_gives_rho_minus_one():
    x = list(range(13))
    stats = rank_agreement_point(x, list(reversed(x)))
    assert stats["spearman_rho"] == -1.0


def test_permutation_pvalue_is_tiny_for_perfect_agreement():
    x = list(range(13))
    result = permutation_pvalue(x, x, n_perm=20_000, seed=0)
    assert result["rho_observed"] == 1.0
    assert result["p_value"] < 0.001


def test_permutation_pvalue_is_large_for_a_typical_random_pairing():
    rng = np.random.default_rng(3)
    x = list(range(13))
    y = list(rng.permutation(13))
    result = permutation_pvalue(x, y, n_perm=20_000, seed=0)
    assert result["p_value"] > 0.05


def test_permutation_pvalue_never_exactly_zero():
    x = list(range(13))
    result = permutation_pvalue(x, x, n_perm=1000, seed=0)
    assert result["p_value"] > 0.0


def test_joint_bootstrap_ci_contains_the_point_estimate_for_strong_agreement():
    arms = [f"a{i}" for i in range(13)]
    # monotonically separated true rates on both sides -> strong agreement
    real = {a: [True] * (50 + 5 * i) + [False] * (150 - 5 * i) for i, a in enumerate(arms)}
    synth = {a: [True] * (50 + 5 * i) + [False] * (150 - 5 * i) for i, a in enumerate(arms)}
    real_clusters = [f"run{i}" for i in range(200)]
    ci = joint_bootstrap_rho_ci(real, real_clusters, synth, arms, n_boot=2000, seed=0)
    lo, hi = ci["ci95"]
    assert lo <= 1.0 <= hi + 1e-9
    assert lo > 0.5


def test_joint_bootstrap_ci_is_wide_for_unrelated_rates():
    rng = np.random.default_rng(7)
    arms = [f"a{i}" for i in range(13)]
    real = {a: list(rng.random(300) < 0.3) for a in arms}
    synth = {a: list(rng.random(300) < 0.3) for a in arms}
    real_clusters = [f"run{i}" for i in range(300)]
    ci = joint_bootstrap_rho_ci(real, real_clusters, synth, arms, n_boot=2000, seed=1)
    lo, hi = ci["ci95"]
    assert hi - lo > 0.3  # unrelated random rates should not give a tight CI


def test_classify_reading_rule():
    assert classify_reading_rule(rho=0.9, rho_ci_lo=0.6, p_value=0.001) == "supported"
    assert classify_reading_rule(rho=0.3, rho_ci_lo=0.1, p_value=0.2) == "weak"
    assert classify_reading_rule(rho=-0.1, rho_ci_lo=-0.4, p_value=0.5) == "absent"
    # rho CI lower bound > 0.5 but p not significant -> weak, not supported
    assert classify_reading_rule(rho=0.7, rho_ci_lo=0.55, p_value=0.2) == "weak"


def test_rank_rows_averages_ties():
    ranks = _rank_rows(np.array([[0.2, 0.2, 0.5], [0.9, 0.1, 0.9]]))
    assert ranks.tolist() == [[1.5, 1.5, 3.0], [2.5, 1.0, 2.5]]


def test_joint_bootstrap_ci_contains_the_tie_aware_point_estimate():
    """Resamples that cannot vary must reproduce the tie-aware point rho
    exactly (regression: position-broken ties gave a degenerate CI of a
    different statistic)."""
    arms = ["a", "b", "c", "d", "e"]
    real_true_counts = {"a": 2, "b": 4, "c": 6, "d": 8, "e": 10}
    real = {a: [True] * n + [False] * (10 - n) for a, n in real_true_counts.items()}
    real_clusters = ["run0"] * 10  # one cluster: every real resample is the full data
    synth = {a: [a == "e"] * 10 for a in arms}  # constant per arm: rates 0,0,0,0,1 (four-way tie)
    point = rank_agreement_point([0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0])["spearman_rho"]
    ci = joint_bootstrap_rho_ci(real, real_clusters, synth, arms, n_boot=200, seed=0)
    assert np.allclose(ci["ci95"], [point, point])


def test_permutation_pvalue_respects_ties():
    """With five arms (120 orderings) no permutation p-value can fall below
    about 1/120 for a non-perfect rho; position-broken ties allowed ~1e-5."""
    real = [0.5047, 0.4429, 0.4841, 0.4381, 0.9083]
    synth = [0.7115, 0.6948, 0.7115, 0.6948, 0.9719]
    result = permutation_pvalue(real, synth, n_perm=20_000, seed=0)
    assert result["p_value"] > 1 / 120
