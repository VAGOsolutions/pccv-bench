"""Uncertainty quantification for the injection study.

An independent review rejected the previous ``_proportion_ci`` for three
separate reasons, and this module exists to answer each of them:

**The resampling unit was wrong.**  Evaluated instances reuse source frames,
and those frames come from far fewer *capture events*: one
production run is photographed by six fixed cameras at once, and every
injection class is applied to the same frames.  Verdicts sharing a ``run_id``
are therefore not independent draws, and resampling individual flags reports an
interval for a study with several times more information than was collected.
The resampling unit here is the capture run, and whole clusters are drawn with
replacement (Field & Welsh 2007, "Bootstrapping clustered data").

**The interval degenerated at the boundary.**  A percentile bootstrap over
observations that all agree returns ``[0, 0]``, which the manuscript then read
as "no risk".  Zero observed failures is not zero possible failures: whenever
the resampled statistic has no variation left to describe, the interval falls
back to an exact Clopper-Pearson limit taken over the *clusters*, which is the
number of independent things actually observed.  :func:`wilson_interval` is
provided alongside for the same boundary problem in its score-interval form.

**Method comparison was done by eye.**  Two overlapping marginal intervals do
not test whether the arms differ; the arms are evaluated on identical
instances, so the difference is a *paired* quantity and must be estimated
directly.  :func:`paired_difference` resamples clusters once and reads both
arms out of the same drawn clusters, which keeps every instance's two verdicts
together and removes the between-instance variance that dominates the marginal
intervals."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Hashable, Iterable, Sequence
import numpy as np
from scipy.stats import beta as _beta


#: Bootstrap settings.  Fixed so that a rerun of the study reproduces the
#: published intervals exactly; every entry point takes an override.
DEFAULT_N_BOOT: int = 2000


DEFAULT_SEED: int = 42


DEFAULT_CONFIDENCE: float = 0.95


#: Reported values are rounded here rather than at each call site, so a
#: bootstrap CI and an exact CI are never printed at different precisions.
_ROUND = 4


def cluster_key(verdict: Any) -> str:
    """The independent capture event a verdict belongs to.

    ``Verdict.run_id`` is carried through from ``Instance.run_id`` precisely so
    this does not have to be parsed back out of ``instance_id`` (which encodes
    camera and frame stem, not the run).  A verdict without a ``run_id`` — only
    hand-built ones, since both providers set it — is treated as its own
    cluster, which reproduces the discredited per-flag behaviour; that fallback
    exists so old fixtures still evaluate, not because it is defensible.
    """
    return getattr(verdict, "run_id", "") or verdict.instance_id


def frame_key(verdict: Any) -> tuple[str, str]:
    """The source *frame*: one camera's view of one capture event.

    Distinct from the cluster because six frames share a capture event, and
    distinct from the instance because every injection class reuses the frame.
    """
    return (verdict.camera, cluster_key(verdict))


def clopper_pearson_interval(
    successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial interval.

    Chosen over Wilson for the degenerate fallback because it never
    under-covers: when the study observed no failures at all, the reported
    upper limit should be the most conservative honest one.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - confidence
    lo = 0.0 if successes == 0 else float(_beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(_beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return (lo, hi)


def _cluster_totals(
    values: Sequence[bool], clusters: Sequence[Hashable] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster (sum, size), in first-appearance order for reproducibility."""
    arr = np.asarray(list(values), dtype=float)
    if clusters is None:
        return arr, np.ones_like(arr)
    keys = list(clusters)
    if len(keys) != arr.size:
        raise ValueError("values and clusters must be the same length")
    index: OrderedDict[Hashable, int] = OrderedDict()
    for k in keys:
        index.setdefault(k, len(index))
    m = len(index)
    sums = np.zeros(m, dtype=float)
    sizes = np.zeros(m, dtype=float)
    for value, key in zip(arr, keys):
        i = index[key]
        sums[i] += value
        sizes[i] += 1.0
    return sums, sizes


def _percentile_ci(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 100.0 * (1.0 - confidence) / 2.0
    return (float(np.percentile(samples, alpha)), float(np.percentile(samples, 100.0 - alpha)))


def _proportion_interval(
    values: Sequence[bool],
    clusters: Sequence[Hashable] | None,
    *,
    n_boot: int,
    seed: int,
    confidence: float,
) -> tuple[float, float, str]:
    sums, sizes = _cluster_totals(values, clusters)
    m = sums.size
    if m == 0:
        return (float("nan"), float("nan"), "undefined")
    if m == 1:
        # One capture event carries no information about between-event
        # variability; saying so is more useful than resampling inside it.
        return (0.0, 1.0, "single_cluster")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    means = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = _percentile_ci(means, confidence)
    if hi > lo:
        return (lo, hi, "cluster_bootstrap")

    # No variation survived the resampling — the [0, 0] failure mode.  The
    # clusters are then the only countable evidence, so bound the rate exactly
    # over them.
    rate = float(sums.sum() / sizes.sum())
    exact_lo, exact_hi = clopper_pearson_interval(int(round(rate * m)), m, confidence)
    return (exact_lo, exact_hi, "clopper_pearson_clusters")


def proportion_summary(
    values: Sequence[bool],
    clusters: Sequence[Hashable] | None = None,
    frames: Sequence[Hashable] | None = None,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """A rate with its denominators and a cluster-aware interval.

    The number of source frames and independent capture
    clusters beside every result, so ``n`` alone is never returned: ``n`` counts
    evaluated instances, ``n_frames`` the camera views they were drawn from, and
    ``n_clusters`` the independent capture events — the last being the one that
    determines how much the interval can narrow.
    """
    arr = list(values)
    n = len(arr)
    if n == 0:
        return {
            "rate": None,
            "n": 0,
            "n_frames": 0,
            "n_clusters": 0,
            "ci95": [None, None],
            "ci_method": None,
        }
    keys = list(clusters) if clusters is not None else list(range(n))
    lo, hi, method = _proportion_interval(
        arr, keys, n_boot=n_boot, seed=seed, confidence=confidence
    )
    return {
        "rate": round(sum(bool(v) for v in arr) / n, _ROUND),
        "n": n,
        "n_frames": len(set(frames)) if frames is not None else len(set(keys)),
        "n_clusters": len(set(keys)),
        "ci95": [round(lo, _ROUND), round(hi, _ROUND)],
        "ci_method": method,
    }


def verdict_proportion(
    verdicts: Iterable[Any],
    value: Callable[[Any], bool],
    **kwargs: Any,
) -> dict[str, Any]:
    """:func:`proportion_summary` over verdicts, using their capture runs."""
    vs = list(verdicts)
    return proportion_summary(
        [bool(value(v)) for v in vs],
        clusters=[cluster_key(v) for v in vs],
        frames=[frame_key(v) for v in vs],
        **kwargs,
    )


def paired_difference_replicates(
    values_a: Sequence[bool],
    values_b: Sequence[bool],
    clusters: Sequence[Hashable],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Bootstrap replicates of ``rate(a) - rate(b)`` with pairing preserved.

    One set of cluster indices is drawn and *both* arms are read out of it, so
    an instance can never contribute to arm A's replicate without also
    contributing to arm B's.  That is what makes this an estimate of the
    difference rather than of two unrelated rates: when the arms agree
    everywhere the replicates are identically zero, whereas resampling them
    independently would manufacture spread out of nothing.
    """
    if len(values_a) != len(values_b):
        raise ValueError("paired arms must be evaluated on the same instances")
    sums_a, sizes = _cluster_totals(values_a, clusters)
    sums_b, _ = _cluster_totals(values_b, clusters)
    m = sums_a.size
    if m == 0:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    n_drawn = sizes[idx].sum(axis=1)
    return (sums_a[idx].sum(axis=1) - sums_b[idx].sum(axis=1)) / n_drawn


def paired_difference(
    values_a: Sequence[bool],
    values_b: Sequence[bool],
    clusters: Sequence[Hashable],
    frames: Sequence[Hashable] | None = None,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Paired, cluster-aware difference between two arms on identical instances.

    ``significant`` is True only when the interval excludes zero.  It replaces
    the "their intervals overlap, so they tie" reading that misleads: the
    marginal intervals answer a different question and can overlap widely while
    the paired difference is unambiguous.
    """
    a = [bool(v) for v in values_a]
    b = [bool(v) for v in values_b]
    if len(a) != len(b):
        raise ValueError("paired arms must be evaluated on the same instances")
    n = len(a)
    keys = list(clusters)
    if n == 0:
        return {
            "difference": None,
            "rate_a": None,
            "rate_b": None,
            "n": 0,
            "n_frames": 0,
            "n_clusters": 0,
            "ci95": [None, None],
            "ci_method": None,
            "significant": False,
        }
    rate_a = sum(a) / n
    rate_b = sum(b) / n
    diff = rate_a - rate_b
    m = len(set(keys))

    if m == 1:
        lo, hi, method = -1.0, 1.0, "single_cluster"
    else:
        reps = paired_difference_replicates(a, b, keys, n_boot=n_boot, seed=seed)
        lo, hi = _percentile_ci(reps, confidence)
        method = "cluster_bootstrap"
        if hi <= lo:
            # Every cluster showed the same effect, so there is no
            # between-cluster variation for the bootstrap to describe.  Bound
            # the effect instead by an exact interval on how often clusters
            # disagreed at all — the cluster-level analogue of resting a paired
            # binary comparison on its discordant pairs.  Deliberately
            # conservative: it will call a perfectly constant effect
            # non-significant rather than infer precision from m clusters that
            # happened to be identical.
            per_cluster = {}
            for va, vb, key in zip(a, b, keys):
                per_cluster[key] = per_cluster.get(key, 0) + (int(va) - int(vb))
            k = sum(1 for total in per_cluster.values() if total != 0)
            k_lo, k_hi = clopper_pearson_interval(k, m, confidence)
            half = max(k_hi - k / m, k / m - k_lo)
            lo, hi = max(-1.0, diff - half), min(1.0, diff + half)
            method = "clopper_pearson_discordant_clusters"

    return {
        "difference": round(diff, _ROUND),
        "rate_a": round(rate_a, _ROUND),
        "rate_b": round(rate_b, _ROUND),
        "n": n,
        "n_frames": len(set(frames)) if frames is not None else len(set(keys)),
        "n_clusters": m,
        "ci95": [round(lo, _ROUND), round(hi, _ROUND)],
        "ci_method": method,
        "significant": bool(lo > 0.0 or hi < 0.0),
    }


def paired_verdict_difference(
    verdicts_a: Iterable[Any],
    verdicts_b: Iterable[Any],
    value: Callable[[Any], bool],
    **kwargs: Any,
) -> dict[str, Any]:
    """Paired difference between two arms, matched by ``instance_id``.

    Matching by identifier rather than by list position is what makes the claim
    "evaluated on identical instances" checkable: a mismatched population is an
    error here rather than a silently misaligned comparison.
    """
    by_id_a = {v.instance_id: v for v in verdicts_a}
    by_id_b = {v.instance_id: v for v in verdicts_b}
    if set(by_id_a) != set(by_id_b):
        raise ValueError(
            "paired comparison requires the same instances in both arms "
            f"({len(by_id_a)} vs {len(by_id_b)}, "
            f"{len(set(by_id_a) ^ set(by_id_b))} unmatched)"
        )
    ids = sorted(by_id_a)
    ordered_a = [by_id_a[i] for i in ids]
    return paired_difference(
        [bool(value(by_id_a[i])) for i in ids],
        [bool(value(by_id_b[i])) for i in ids],
        clusters=[cluster_key(v) for v in ordered_a],
        frames=[frame_key(v) for v in ordered_a],
        **kwargs,
    )
