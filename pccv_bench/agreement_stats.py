"""Rank-agreement statistics for the PCCV-Bench agreement study (E4, E6).

``pccv_bench.release_validation`` already has a plain Spearman rho (no CI, no
p-value, no Kendall). PREREGISTRATION.md §3 needs more: Kendall tau-b, a
100,000-permutation p-value, and a 10,000-replicate joint bootstrap CI for
rho that resamples the synthetic instances and the real capture-run clusters
*independently* (PREREGISTRATION.md §2: instance-level bootstrap on the
synthetic side, capture-run cluster bootstrap on the real side, same as the
real study, ``eval/statistics.py``).

Every synthetic ``Instance``/``Verdict`` carries a unique ``run_id``
(``synthetic_provider.py``: ``run_id=f"attempt-{attempt}"``), so a synthetic
"cluster" is always a singleton — cluster-bootstrapping the synthetic side
with ``pccv_bench.core.statistics`` machinery already *is* a plain instance bootstrap.
That is used directly (via ``pccv_bench.core.statistics.paired_difference`` with
per-instance cluster keys) for the paired synthetic differences in E3/E6; the
functions here are for the cross-study rank-correlation statistics only,
which need both sides resampled together per replicate.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


def _rank_rows(x: np.ndarray) -> np.ndarray:
    """Rank each row of a 2D array independently, ties given their average rank.

    Tie-averaging is required, not optional: arms that share a mechanism
    (e.g. every two-way arm flags every ghost part, or ``geometry`` and
    ``+temporal`` flag the same synthetic instances) produce *identical*
    rates in every resample, so ties are common. Breaking them by position
    computes a different statistic from the tie-aware point estimate, and the
    resulting CI and permutation p-value need not even contain or match it.
    """
    return rankdata(x, axis=1, method="average")


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between corresponding rows of two 2D arrays."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def rank_agreement_point(real: Sequence[float], synthetic: Sequence[float]) -> dict[str, float]:
    """Observed Spearman rho and Kendall tau-b (exact, tie-aware; scipy)."""
    rho, _ = spearmanr(real, synthetic)
    tau, tau_p = kendalltau(real, synthetic, variant="b")
    return {"spearman_rho": float(rho), "kendall_tau_b": float(tau), "kendall_tau_b_asymptotic_p": float(tau_p)}


def permutation_pvalue(
    real: Sequence[float],
    synthetic: Sequence[float],
    *,
    n_perm: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Two-sided permutation p-value for Spearman rho.

    Fixes the real ranks and repeatedly re-pairs them against a random
    permutation of the synthetic ranks (PREREGISTRATION.md §3: "100,000
    random permutations of the synthetic ranks" -- 13! is too large to
    enumerate). ``p = (1 + #{|rho_perm| >= |rho_obs|}) / (n_perm + 1)``, the
    standard +1 correction so a permutation p-value is never reported as
    exactly 0.
    """
    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    k = real.size
    if synthetic.size != k:
        raise ValueError("real and synthetic must have the same length")
    rho_obs = float(spearmanr(real, synthetic)[0])

    rng = np.random.default_rng(seed)
    real_rank = _rank_rows(real.reshape(1, -1)).repeat(n_perm, axis=0)
    synth_tile = np.tile(synthetic, (n_perm, 1))
    synth_perm = rng.permuted(synth_tile, axis=1)
    synth_rank = _rank_rows(synth_perm)
    rho_perm = _pearson_rows(real_rank, synth_rank)

    p = float((1 + np.sum(np.abs(rho_perm) >= abs(rho_obs) - 1e-12)) / (n_perm + 1))
    return {"rho_observed": rho_obs, "p_value": p, "n_perm": n_perm, "seed": seed}


def _cluster_index(clusters: Sequence[Hashable]) -> tuple[np.ndarray, int]:
    index: OrderedDict[Hashable, int] = OrderedDict()
    for k in clusters:
        index.setdefault(k, len(index))
    return np.array([index[k] for k in clusters]), len(index)


def _cluster_bootstrap_rates(
    values_per_arm: Mapping[str, Sequence[bool]],
    clusters: Sequence[Hashable],
    arm_order: Sequence[str],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_boot, len(arm_order)) resampled rates, one shared cluster draw per
    replicate reused across every arm (so a replicate's rate vector is
    internally consistent -- the same drawn capture runs for every arm)."""
    cluster_of, m = _cluster_index(clusters)
    idx = rng.integers(0, m, size=(n_boot, m))
    out = np.empty((n_boot, len(arm_order)), dtype=float)
    for j, arm in enumerate(arm_order):
        vals = np.asarray(values_per_arm[arm], dtype=float)
        sums = np.zeros(m)
        sizes = np.zeros(m)
        np.add.at(sums, cluster_of, vals)
        np.add.at(sizes, cluster_of, 1.0)
        out[:, j] = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return out


def _instance_bootstrap_rates(
    values_per_arm: Mapping[str, Sequence[bool]],
    arm_order: Sequence[str],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_boot, len(arm_order)) resampled rates, one shared instance draw per
    replicate reused across every arm."""
    n = len(next(iter(values_per_arm.values())))
    idx = rng.integers(0, n, size=(n_boot, n))
    out = np.empty((n_boot, len(arm_order)), dtype=float)
    for j, arm in enumerate(arm_order):
        vals = np.asarray(values_per_arm[arm], dtype=float)
        out[:, j] = vals[idx].mean(axis=1)
    return out


def joint_bootstrap_rho_ci(
    real_values_per_arm: Mapping[str, Sequence[bool]],
    real_clusters: Sequence[Hashable],
    synthetic_values_per_arm: Mapping[str, Sequence[bool]],
    arm_order: Sequence[str],
    *,
    n_boot: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """95% CI for Spearman rho from a joint bootstrap: the real side is
    resampled by capture-run cluster, the synthetic side by instance, each
    independently drawn (separate ``rng`` calls) but both varying within the
    same replicate index, so replicate *b*'s rho reflects one coherent
    resample of both studies at once.
    """
    rng = np.random.default_rng(seed)
    real_reps = _cluster_bootstrap_rates(real_values_per_arm, real_clusters, arm_order, n_boot, rng)
    synth_reps = _instance_bootstrap_rates(synthetic_values_per_arm, arm_order, n_boot, rng)
    rho_reps = _pearson_rows(_rank_rows(real_reps), _rank_rows(synth_reps))
    rho_reps = rho_reps[~np.isnan(rho_reps)]
    alpha = 100.0 * (1.0 - confidence) / 2.0
    lo, hi = np.percentile(rho_reps, [alpha, 100.0 - alpha])
    return {
        "ci95": [float(lo), float(hi)],
        "n_boot": n_boot,
        "n_valid_replicates": int(rho_reps.size),
        "seed": seed,
    }


def classify_reading_rule(rho: float, rho_ci_lo: float, p_value: float) -> str:
    """PREREGISTRATION.md §3's reading rule.

    - **supported**: rho CI lower bound > 0.5 and p < 0.05.
    - **weak**: rho > 0 but either condition above fails.
    - **absent**: otherwise.
    """
    if rho_ci_lo > 0.5 and p_value < 0.05:
        return "supported"
    if rho > 0:
        return "weak"
    return "absent"
