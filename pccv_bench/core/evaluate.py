"""Metrics for the injection study.

Every rate is reported with its denominators and a confidence interval, and
every breakdown carries its own.  A pooled recall without
per-class and per-camera counts is not reportable, and frames reused
across classes must not be presented as independent instances.  That last point
is why a rate here carries three counts and not one: ``n`` evaluated instances,
``n_frames`` distinct camera views, and ``n_clusters`` independent capture
events — the last being the sample size the interval is actually built on
.  All interval machinery lives in :mod:`pccv_bench.core.statistics`."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Sequence
import numpy as np
from pccv_bench.core.harness import Verdict
from pccv_bench.core.statistics import DEFAULT_N_BOOT, DEFAULT_SEED, cluster_key, frame_key, verdict_proportion


BOOTSTRAP_N: int = DEFAULT_N_BOOT


BOOTSTRAP_SEED: int = DEFAULT_SEED


#: How each expected action is read off the flag rate.
INTERPRETATION = {
    "alarm": "detection_rate",
    "pass": "false_error_rate",
    # A timing condition the arm could not re-explain is escalated to a human,
    # which is defensible — so its flag rate is neither a hit nor a false error.
    "recover": "escalation_rate",
}


SEVERITY_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 0.34),
    ("mid", 0.34, 0.67),
    ("high", 0.67, 1.01),
)


def _rate(
    verdicts: Sequence[Verdict], value: Callable[[Verdict], bool]
) -> dict[str, Any]:
    """One reported rate, clustered by capture event.

    Takes verdicts rather than a list of booleans because the interval needs
    each observation's capture run: resampling the booleans on their own is the
    per-flag bootstrap an independent review rejected.
    """
    return verdict_proportion(verdicts, value, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)


def _counts(verdicts: Sequence[Verdict]) -> dict[str, int]:
    """Instance / frame / cluster denominators for a population."""
    return {
        "n_instances": len(verdicts),
        "n_frames": len({frame_key(v) for v in verdicts}),
        "n_clusters": len({cluster_key(v) for v in verdicts}),
    }


def _severity_bucket(sev: float) -> str:
    for name, lo, hi in SEVERITY_BUCKETS:
        if lo <= sev < hi:
            return name
    return "high"


def per_class_rates(verdicts: Iterable[Verdict]) -> dict[str, Any]:
    """Flag rate per injection class, with the correct interpretation attached.

    For an ``alarm`` class the rate is **detection rate** (recall); for a
    ``pass`` class it is the **false-error rate**; for a ``recover`` class it is
    an **escalation rate** and belongs to neither.  All three are the same
    underlying quantity — the fraction flagged — so they are computed once and
    labelled, rather than being three numbers a reader has to reconcile.
    """
    by_class: dict[str, list[Verdict]] = defaultdict(list)
    for v in verdicts:
        by_class[v.injection_class].append(v)

    out: dict[str, Any] = {}
    for cls, vs in sorted(by_class.items()):
        expected_action = vs[0].expected_action
        entry: dict[str, Any] = {
            "expected_action": expected_action,
            "fault_type": vs[0].fault_type,
            "interpretation": INTERPRETATION.get(expected_action, "detection_rate"),
            **_counts(vs),
            "flag_rate": _rate(vs, lambda v: v.flagged),
            "box_side_rate": _rate(vs, lambda v: v.flagged_box_side),
            "record_side_rate": _rate(vs, lambda v: v.flagged_record_side),
        }
        # Severity gradient — the point of giving every class a severity.
        buckets: dict[str, Any] = {}
        by_bucket: dict[str, list[Verdict]] = defaultdict(list)
        for v in vs:
            by_bucket[_severity_bucket(v.severity)].append(v)
        for name, _, _ in SEVERITY_BUCKETS:
            if by_bucket.get(name):
                buckets[name] = _rate(by_bucket[name], lambda v: v.flagged)
        entry["by_severity"] = buckets
        out[cls] = entry
    return out


def per_camera_rates(verdicts: Iterable[Verdict]) -> dict[str, Any]:
    """Flag rate per camera per class, with both breakdowns."""
    by_key: dict[tuple[str, str], list[Verdict]] = defaultdict(list)
    for v in verdicts:
        by_key[(v.injection_class, v.camera)].append(v)
    out: dict[str, Any] = defaultdict(dict)
    for (cls, cam), vs in sorted(by_key.items()):
        out[cls][cam] = _rate(vs, lambda v: v.flagged)
    return dict(out)


def diagnosis_states(verdicts: Iterable[Verdict]) -> dict[str, Any]:
    """Box-side outcome crossed with ground truth, counted per class and camera.

    States: ``matched_at_t`` / ``recovered_by_offset`` / ``recovered_temporally``
    / ``fault_masked_by_offset`` / ``fault_masked_temporally`` / ``unresolved``.

    The two ``fault_masked_*`` states are the dangerous ones — a correction
    explained away a real physical fault — and they are why a correction
    mechanism must be reported with a cost and not just a recovery rate.  They
    are counted separately from the recoveries they look identical to
    mechanically, which is the distinction the previous ``temporal_state`` field
    got backwards for every ``snapshot_desync`` instance.
    """
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_camera: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for v in verdicts:
        st = v.diagnosis
        by_class[v.injection_class][st] += 1
        by_camera[v.camera][st] += 1
        totals[st] += 1
    return {
        "totals": dict(totals),
        "by_class": {k: dict(v) for k, v in sorted(by_class.items())},
        "by_camera": {k: dict(v) for k, v in sorted(by_camera.items())},
    }


def recovery_and_masking(verdicts: Iterable[Verdict]) -> dict[str, Any]:
    """Per-class recovery and masking rates, each on its own denominator.

    Recovery is only defined where the expected action is ``pass`` or
    ``recover``; masking only where it is ``alarm``.  Reporting one rate over
    the other's population is how a successful temporal retry came to be counted
    as a masked fault.
    """
    by_class: dict[str, list[Verdict]] = defaultdict(list)
    for v in verdicts:
        by_class[v.injection_class].append(v)

    out: dict[str, Any] = {}
    for cls, vs in sorted(by_class.items()):
        entry: dict[str, Any] = {"expected_action": vs[0].expected_action, **_counts(vs)}
        if vs[0].expected_action in ("pass", "recover"):
            entry["recovery_rate"] = _rate(vs, lambda v: v.was_recovered)
        else:
            entry["masking_rate"] = _rate(vs, lambda v: v.was_fault_masked)
            entry["masked_by"] = dict(
                Counter(v.masking_mechanism for v in vs if v.was_fault_masked)
            )
        entry["mixed_correction_rate"] = _rate(vs, lambda v: v.mixed_corrections)
        out[cls] = entry
    return out


def offset_diagnostics(verdicts: Iterable[Verdict]) -> dict[str, Any]:
    """Selected-offset distribution plus how often the correction was used.

    ``wrongly_explained_rate`` is the masking cost: genuinely inconsistent
    instances that the x-offset sweep managed to explain away.
    """
    vs = list(verdicts)
    used = [v for v in vs if v.correction_outcome == "recovered_by_offset"]
    offsets = [v.corrected_offset_mm for v in used if v.corrected_offset_mm > 0]

    masked = [v for v in vs if v.masking_mechanism == "offset"]
    alarm_population = [v for v in vs if v.should_alarm]

    q = {}
    if offsets:
        arr = np.asarray(offsets, dtype=float)
        q = {
            "median_mm": float(np.median(arr)),
            "p25_mm": float(np.percentile(arr, 25)),
            "p75_mm": float(np.percentile(arr, 75)),
            "p90_mm": float(np.percentile(arr, 90)),
            "max_mm": float(arr.max()),
        }
    return {
        "n_used_x_offset": len(used),
        "selected_offset": q,
        "wrongly_explained_rate": _rate(
            alarm_population, lambda v: v.masking_mechanism == "offset"
        ),
        "n_masked_by_offset": len(masked),
    }


def occlusion_strata(verdicts: Iterable[Verdict]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[Verdict]] = defaultdict(list)
    for v in verdicts:
        by_key[(v.occlusion_stratum, v.injection_class)].append(v)
    out: dict[str, Any] = defaultdict(dict)
    for (stratum, cls), vs in sorted(by_key.items()):
        out[stratum][cls] = _rate(vs, lambda v: v.flagged)
    return dict(out)


def summarise_arm(arm_name: str, verdicts: list[Verdict]) -> dict[str, Any]:
    """Everything reportable for one arm.

    The three populations are disjoint and defined by the *expected action*, not
    by whether a record was modified: alarm conditions carry the detection rate,
    pass conditions the false-error rate, and timing conditions their own
    recovery rate.  Mixing the last into either of the first two — as a single
    ``is_inconsistent`` boolean forced — both understated detection and
    overstated false errors for every arm without a temporal retry.
    """
    alarm_pop = [v for v in verdicts if v.expected_action == "alarm"]
    pass_pop = [v for v in verdicts if v.expected_action == "pass"]
    recover_pop = [v for v in verdicts if v.expected_action == "recover"]
    clean_only = [v for v in verdicts if v.injection_class == "clean"]

    return {
        "arm": arm_name,
        "n_instances": len(verdicts),
        # Instances reuse source frames, and frames share capture events.
        # Publishing the instance count alone overstates the sample by that
        # factor, so all three denominators travel with the summary.
        "n_frames": len({frame_key(v) for v in verdicts}),
        "n_clusters": len({cluster_key(v) for v in verdicts}),
        "headline": {
            # Pooled numbers are reported but must never be quoted without the
            # per-class table.
            "detection_rate_pooled": _rate(alarm_pop, lambda v: v.flagged),
            "false_error_rate_pooled": _rate(pass_pop, lambda v: v.flagged),
            "false_error_rate_clean_only": _rate(clean_only, lambda v: v.flagged),
            "recovery_rate_pooled": _rate(recover_pop, lambda v: v.was_recovered),
            "masking_rate_pooled": _rate(alarm_pop, lambda v: v.was_fault_masked),
        },
        "per_class": per_class_rates(verdicts),
        "per_camera": per_camera_rates(verdicts),
        "diagnosis": diagnosis_states(verdicts),
        "recovery_and_masking": recovery_and_masking(verdicts),
        "offset": offset_diagnostics(verdicts),
        "occlusion": occlusion_strata(verdicts),
    }


def format_summary_table(summaries: list[dict[str, Any]]) -> str:
    """Compact text table across arms — the at-a-glance result."""
    lines: list[str] = []
    lines.append("")
    lines.append(
        f"{'arm':22s} {'n':>6s} {'frames':>7s} {'clust':>6s} "
        f"{'detect':>16s} {'false-err(clean)':>18s}"
    )
    lines.append("-" * 80)
    for s in summaries:
        d = s["headline"]["detection_rate_pooled"]
        f = s["headline"]["false_error_rate_clean_only"]
        d_s = "—" if d["rate"] is None else f"{d['rate']:.3f} (n={d['n']})"
        f_s = "—" if f["rate"] is None else f"{f['rate']:.3f} (n={f['n']})"
        lines.append(
            f"{s['arm']:22s} {s['n_instances']:6d} {s.get('n_frames', 0):7d} "
            f"{s.get('n_clusters', 0):6d} {d_s:>16s} {f_s:>18s}"
        )
    lines.append("")
    return "\n".join(lines)


def format_per_class_table(summary: dict[str, Any]) -> str:
    """Per-class table for one arm, with counts and the box/record split."""
    lines: list[str] = []
    lines.append(f"\n  per-class — arm '{summary['arm']}'")
    lines.append(
        f"    {'class':28s} {'expect':>8s} {'flagged':>16s} {'95% CI':>16s} "
        f"{'frames':>7s} {'clust':>6s} {'box':>8s} {'record':>8s}"
    )
    lines.append("    " + "-" * 104)
    for cls, e in summary["per_class"].items():
        r = e["flag_rate"]
        rate_s = "—" if r["rate"] is None else f"{r['rate']:.3f} (n={r['n']})"
        ci = r["ci95"]
        ci_s = "—" if ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        b = e["box_side_rate"]["rate"]
        rc = e["record_side_rate"]["rate"]
        lines.append(
            f"    {cls:28s} {e['expected_action']:>8s} "
            f"{rate_s:>16s} {ci_s:>16s} {e['n_frames']:7d} {e['n_clusters']:6d} "
            f"{('—' if b is None else f'{b:.3f}'):>8s} "
            f"{('—' if rc is None else f'{rc:.3f}'):>8s}"
        )
    return "\n".join(lines)
