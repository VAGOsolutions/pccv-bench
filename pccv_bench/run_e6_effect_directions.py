"""PCCV-Bench agreement study, E6 — effect-direction agreement (PREREGISTRATION.md §4).

For each of the 15 pre-registered contrasts C1-C15: the synthetic paired
difference (95% bootstrap CI, instance-level -- every synthetic instance is
its own cluster), the real value (from ``injection_summary.json``, or for
C13-C15 from E3's matched-budget result, itself sourced from
the real study's matched-budget file), and the outcome (agree /
inconclusive / disagree) by the §4 rule.

C1-C12 need the seed-42 aligned synthetic verdicts for two arms each (all
scored on the same instance set, so paired by ``instance_id`` directly --
unlike E3, a single seed has no cross-seed ``instance_id`` collision risk).
Reuses the verdicts cache E4 writes (``--dump-verdicts-cache``) rather than
re-running the ladder a third time. C13-C15 are read from E3's own paired-
difference output (already instance-bootstrapped, pooled across eval seeds).

Usage::

    python -m pccv_bench.run_e6_effect_directions \\
        --verdicts-cache /tmp/pccv_seed42_aligned_verdicts.json \\
        --e3-result results/agreement_v2/e3_matched_budget.json \\
        --out-dir results/agreement_v2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.core.statistics import DEFAULT_N_BOOT, DEFAULT_SEED, paired_verdict_difference, verdict_proportion
from pccv_bench.run_e4_rank_agreement import _metric_predicate, _real_rate

#: E3/E6 metric label -> population label in the real study's matched-budget file.
E3_TO_REAL_POPULATION = {
    "missing_part_recall": "missing_part_recall",
    "pooled_detection": "detection_rate_pooled",
    "pooled_false_error": "false_error_rate_pooled",
}

#: (id, claim, kind, ...) -- kind is "pair" (arm_a - arm_b), "level" (C10:
#: single arm vs a fixed threshold), or "e3" (pulled from E3's own output).
CONTRASTS: list[dict[str, Any]] = [
    {"id": "C1", "claim": "The record side catches added parts", "kind": "pair",
     "arm_a": "+all+twoway", "arm_b": "+offset+temporal", "metric": "class:ghost_part", "expected_sign": "+"},
    {"id": "C2", "claim": "A second alarm channel raises clean disagreement", "kind": "pair",
     "arm_a": "+all+twoway", "arm_b": "+offset+temporal", "metric": "clean_flag_rate", "expected_sign": "+"},
    {"id": "C3", "claim": "Spatial correction lowers clean disagreement", "kind": "pair",
     "arm_a": "+offset+twoway", "arm_b": "geometry+twoway", "metric": "clean_flag_rate", "expected_sign": "-"},
    {"id": "C4", "claim": "Spatial correction lowers in-bound flags", "kind": "pair",
     "arm_a": "+offset+twoway", "arm_b": "geometry+twoway", "metric": "class:position_shift_x_inbound", "expected_sign": "-"},
    {"id": "C5", "claim": "Spatial correction masks missing parts", "kind": "pair",
     "arm_a": "+offset+twoway", "arm_b": "geometry+twoway", "metric": "class:missing_part", "expected_sign": "-"},
    {"id": "C6", "claim": "Temporal retry recovers desync", "kind": "pair",
     "arm_a": "+temporal+twoway", "arm_b": "geometry+twoway", "metric": "class:snapshot_desync", "expected_sign": "-"},
    {"id": "C7", "claim": "The offset-aware record side lowers in-bound flags", "kind": "pair",
     "arm_a": "+all+twoway+offset_aware", "arm_b": "+all+twoway", "metric": "class:position_shift_x_inbound", "expected_sign": "-"},
    {"id": "C8", "claim": "...and lowers clean disagreement", "kind": "pair",
     "arm_a": "+all+twoway+offset_aware", "arm_b": "+all+twoway", "metric": "clean_flag_rate", "expected_sign": "-"},
    {"id": "C9", "claim": "...at some missing-part cost", "kind": "pair",
     "arm_a": "+all+twoway+offset_aware", "arm_b": "+all+twoway", "metric": "class:missing_part", "expected_sign": "-"},
    {"id": "C10", "claim": "The box side alone rarely sees added parts", "kind": "level",
     "arm_a": "+offset+temporal", "metric": "class:ghost_part", "level": 0.10},
    {"id": "C11", "claim": "At native threshold, symmetric alarms far more on clean data", "kind": "pair",
     "arm_a": "baseline_symmetric", "arm_b": "+all+twoway+offset_aware", "metric": "clean_flag_rate", "expected_sign": "+"},
    {"id": "C12", "claim": "...and so does coherent", "kind": "pair",
     "arm_a": "baseline_coherent", "arm_b": "+all+twoway+offset_aware", "metric": "clean_flag_rate", "expected_sign": "+"},
    {"id": "C13", "claim": "At a matched budget the verifier leads on missing parts (from E3)", "kind": "e3",
     "e3_metric": "missing_part_recall", "expected_sign": "+"},
    {"id": "C14", "claim": "At a matched budget the verifier leads on pooled detection (from E3)", "kind": "e3",
     "e3_metric": "pooled_detection", "expected_sign": "+"},
    {"id": "C15", "claim": "At a matched budget coherent keeps the lower pooled false-error (from E3)", "kind": "e3_negated",
     "e3_metric": "pooled_false_error", "baseline": "coherent", "expected_sign": "-"},
]


def _git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(_ROOT), text=True)
        return f"{rev}-dirty" if dirty else rev
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_verdicts_cache(path: Path) -> dict[str, list[SimpleNamespace]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {arm: [SimpleNamespace(**row) for row in rows] for arm, rows in payload.items()}


def classify(diff_ci: list[float], expected_sign: str) -> str:
    lo, hi = diff_ci
    if lo <= 0.0 <= hi:
        return "inconclusive"
    sign = "+" if lo > 0.0 else "-"
    return "agree" if sign == expected_sign else "disagree"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verdicts-cache", required=True)
    ap.add_argument("--e3-result", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--real-summary", required=True, help="injection_summary.json of the real-data study (not released)")
    ap.add_argument("--real-matched-budget", required=True, help="matched-budget comparison of the real-data study (not released)")
    args = ap.parse_args()

    synth_by_arm = load_verdicts_cache(Path(args.verdicts_cache))
    e3 = json.loads(Path(args.e3_result).read_text(encoding="utf-8"))
    real_summary = json.loads(Path(args.real_summary).read_text(encoding="utf-8"))
    real_headline_by_arm = {s["arm"]: s for s in real_summary["summaries"]}
    real_matched_budget = json.loads(Path(args.real_matched_budget).read_text(encoding="utf-8"))
    real_matched_by_baseline = {c["baseline"]: c for c in real_matched_budget["comparisons"]}

    rows: list[dict[str, Any]] = []
    for c in CONTRASTS:
        if c["kind"] == "pair":
            pred = _metric_predicate(c["metric"])
            va = [v for v in synth_by_arm[c["arm_a"]] if pred(v)]
            vb = [v for v in synth_by_arm[c["arm_b"]] if pred(v)]
            d = paired_verdict_difference(va, vb, value=lambda v: v.flagged, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
            real_a = _real_rate(real_headline_by_arm, c["arm_a"], c["metric"])
            real_b = _real_rate(real_headline_by_arm, c["arm_b"], c["metric"])
            real_diff = real_a - real_b
            outcome = classify(d["ci95"], c["expected_sign"])
            rows.append(
                {
                    "id": c["id"], "claim": c["claim"], "contrast": f"{c['arm_a']} - {c['arm_b']}", "metric": c["metric"],
                    "expected_sign": c["expected_sign"], "real_value": f"{real_a:.4f} - {real_b:.4f} = {real_diff:+.4f}",
                    "synthetic_difference": d["difference"], "synthetic_ci95": d["ci95"], "outcome": outcome,
                }
            )
        elif c["kind"] == "level":
            pred = _metric_predicate(c["metric"])
            va = [v for v in synth_by_arm[c["arm_a"]] if pred(v)]
            p = verdict_proportion(va, value=lambda v: v.flagged, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
            real_a = _real_rate(real_headline_by_arm, c["arm_a"], c["metric"])
            outcome = "agree" if p["ci95"][1] < c["level"] else ("inconclusive" if p["rate"] < c["level"] else "disagree")
            rows.append(
                {
                    "id": c["id"], "claim": c["claim"], "contrast": f"{c['arm_a']} vs level {c['level']}", "metric": c["metric"],
                    "expected_sign": f"below {c['level']}", "real_value": f"{real_a:.4f}",
                    "synthetic_difference": p["rate"], "synthetic_ci95": p["ci95"], "outcome": outcome,
                }
            )
        elif c["kind"] in ("e3", "e3_negated"):
            baseline = c.get("baseline")
            baselines_to_check = [baseline] if baseline else ["symmetric", "coherent"]
            real_pop = E3_TO_REAL_POPULATION[c["e3_metric"]]
            sub_outcomes = []
            real_parts = []
            synth_parts = []
            for b in baselines_to_check:
                d = e3["paired_differences_verifier_minus_baseline"][b][c["e3_metric"]]
                diff, ci = d["difference"], d["ci95"]
                real_diff = real_matched_by_baseline[b]["metrics"][real_pop]["difference"]
                if c["kind"] == "e3_negated":
                    diff, ci, real_diff = -diff, [-ci[1], -ci[0]], -real_diff
                sub_outcomes.append(classify(ci, c["expected_sign"]))
                synth_parts.append(f"{b}: {diff:+.4f} {ci}")
                real_parts.append(f"{b}: {real_diff:+.4f}")
            outcome = "agree" if all(o == "agree" for o in sub_outcomes) else (
                "disagree" if any(o == "disagree" for o in sub_outcomes) else "inconclusive"
            )
            rows.append(
                {
                    "id": c["id"], "claim": c["claim"], "contrast": f"verifier vs {baselines_to_check} (E3)",
                    "metric": c["e3_metric"], "expected_sign": c["expected_sign"],
                    "real_value": "; ".join(real_parts) + " (matched-budget comparison of the companion real-data study)",
                    "synthetic_difference": synth_parts, "synthetic_ci95": None,
                    "sub_outcomes": dict(zip(baselines_to_check, sub_outcomes)),
                    "note": "outcome combines both sub-contrasts: agree only if every sub-contrast agrees, disagree if any does",
                    "outcome": outcome,
                }
            )
        else:
            raise ValueError(c["kind"])

    counts = {"agree": 0, "inconclusive": 0, "disagree": 0}
    for r in rows:
        counts[r["outcome"]] += 1

    payload: dict[str, Any] = {
        "git_revision": _git_rev(),
        "contrasts": rows,
        "headline_counts": counts,
        "n_contrasts": len(rows),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "e6_effect_directions.json"
    out_md = out_dir / "e6_effect_directions.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    for r in rows:
        print(f"{r['id']:5s} {r['outcome']:12s} {r['claim']}")
    print(f"\nheadline: {counts} / {len(rows)}")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


def render_markdown(payload: dict) -> str:
    c = payload["headline_counts"]
    lines = [
        "# E6 — effect-direction agreement",
        "",
        f"- git revision: `{payload['git_revision']}`",
        f"- headline: **{c['agree']} agree / {c['inconclusive']} inconclusive / {c['disagree']} disagree** out of {payload['n_contrasts']}",
        "",
        "| ID | claim | contrast | metric | expected sign | real | synthetic diff | synthetic 95% CI | outcome |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["contrasts"]:
        sd = r["synthetic_difference"]
        sd_s = f"{sd:+.4f}" if isinstance(sd, float) else "; ".join(sd)
        ci_s = f"[{r['synthetic_ci95'][0]:.4f}, {r['synthetic_ci95'][1]:.4f}]" if r["synthetic_ci95"] else "—"
        lines.append(
            f"| {r['id']} | {r['claim']} | `{r['contrast']}` | `{r['metric']}` | {r['expected_sign']} | "
            f"{r['real_value']} | {sd_s} | {ci_s} | **{r['outcome']}** |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
