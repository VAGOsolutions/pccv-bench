"""PCCV-Bench agreement study, E4 — rank agreement (PREREGISTRATION.md §3).

13-arm real/synthetic intersection, 9 metrics each (pooled detection,
clean-control flag rate, per-class flag rate x7): Spearman rho, Kendall
tau-b, a 100,000-permutation p-value, and a 10,000-replicate joint bootstrap
95% CI for rho (real side resampled by capture-run cluster, synthetic side
by instance, independently). Also the 5-arm ladder, for continuity, and a
seed 0-9 stability check of the primary metric's (pooled detection) rho and
reading-rule category.

Real side: ``injection_summary.json`` (point rates) + ``verdicts.json``
(raw per-instance verdicts, for the joint bootstrap's cluster resampling).
Synthetic side: the seed-42 aligned instances, re-scored here to get raw
per-instance verdicts (``pccv_bench.cli ablate`` only persists summaries).

Usage::

    python -m pccv_bench.run_e4_rank_agreement \\
        --synthetic-instances results/instances_medium_v1_2_fitted.json \\
        --seed-sweep-dir /tmp/pccv_seed_sweep \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
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

from pccv_bench.core.inject import INJECTION_CLASSES
from pccv_bench.ablate import build_arms, run_all_arms
from pccv_bench.agreement_stats import (
    classify_reading_rule,
    joint_bootstrap_rho_ci,
    permutation_pvalue,
    rank_agreement_point,
)
from pccv_bench.cameras import fit_synthetic_camera
from pccv_bench.cli import load_frozen_config
from pccv_bench.instance_io import load_instances

import numpy as np

#: Outputs of the companion real-data study. They are not part of this release;
#: ``main`` fills them from ``--real-verdicts`` / ``--real-summary``.
REAL_VERDICTS_PATH: Path | None = None
REAL_SUMMARY_PATH: Path | None = None

#: PREREGISTRATION.md §3: the 13-arm intersection (flag_none/flag_all excluded).
ARMS_13 = (
    "baseline_envelope", "baseline_symmetric", "baseline_coherent", "baseline_pairwise",
    "geometry", "+offset", "+temporal", "+offset+temporal",
    "geometry+twoway", "+offset+twoway", "+temporal+twoway", "+all+twoway", "+all+twoway+offset_aware",
)
LADDER_5 = ("geometry", "+offset", "+temporal", "+offset+temporal", "+all+twoway")

METRICS = ["pooled_detection", "clean_flag_rate"] + [f"class:{c}" for c in INJECTION_CLASSES if c != "clean"]
# clean's per-class rate *is* clean_flag_rate; INJECTION_CLASSES also has
# "clean" itself -- avoid reporting it twice under two labels.


def _git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(_ROOT), text=True)
        return f"{rev}-dirty" if dirty else rev
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _metric_predicate(metric: str):
    if metric == "pooled_detection":
        return lambda v: v.expected_action == "alarm"
    if metric == "clean_flag_rate":
        return lambda v: v.injection_class == "clean"
    cls = metric.split(":", 1)[1]
    return lambda v: v.injection_class == cls


def load_real_verdicts_by_arm() -> dict[str, list[SimpleNamespace]]:
    payload = json.loads(REAL_VERDICTS_PATH.read_text(encoding="utf-8"))
    return {arm: [SimpleNamespace(**row) for row in rows] for arm, rows in payload["verdicts"].items()}


def score_synthetic(instances_path: Path, frozen: dict) -> dict[str, list]:
    instances, gen_meta = load_instances(instances_path)
    cfg = gen_meta.get("config", {})
    seed = gen_meta.get("seed", 42)
    cam_rng = np.random.default_rng(seed)
    cameras = cfg.get("cameras") or sorted({i.camera for i in instances})
    mappers = {c: fit_synthetic_camera(cam_rng, c) for c in cameras}
    return run_all_arms(instances, mappers, frozen=frozen)


def _real_rate(headline_by_arm: dict, arm: str, metric: str) -> float:
    h = headline_by_arm[arm]
    if metric == "pooled_detection":
        return h["headline"]["detection_rate_pooled"]["rate"]
    if metric == "clean_flag_rate":
        return h["headline"]["false_error_rate_clean_only"]["rate"]
    cls = metric.split(":", 1)[1]
    return h["per_class"][cls]["flag_rate"]["rate"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic-instances", required=True)
    ap.add_argument("--seed-sweep-summaries", nargs="*", default=[], help="ablate result JSONs for seeds 0-9 (for the stability check)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--real-verdicts", required=True, help="verdicts.json of the real-data study (not released)")
    ap.add_argument("--real-summary", required=True, help="injection_summary.json of the real-data study (not released)")
    ap.add_argument(
        "--dump-verdicts-cache", default=None,
        help="also save the raw seed-42 aligned synthetic verdicts (all arms) as JSON, "
        "so E6 can reuse this same scoring run instead of re-running the ladder.",
    )
    args = ap.parse_args()
    global REAL_VERDICTS_PATH, REAL_SUMMARY_PATH
    REAL_VERDICTS_PATH, REAL_SUMMARY_PATH = Path(args.real_verdicts), Path(args.real_summary)

    frozen = load_frozen_config()

    print("Scoring synthetic seed-42 aligned instances (raw verdicts needed for the joint bootstrap)…")
    synth_verdicts_all = score_synthetic(Path(args.synthetic_instances), frozen)
    real_verdicts_all = load_real_verdicts_by_arm()

    if args.dump_verdicts_cache:
        cache_path = Path(args.dump_verdicts_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({arm: [v.to_dict() for v in vs] for arm, vs in synth_verdicts_all.items()}),
            encoding="utf-8",
        )
        print(f"cached seed-42 aligned synthetic verdicts -> {cache_path}")

    missing_real = set(ARMS_13) - set(real_verdicts_all)
    missing_synth = set(ARMS_13) - set(synth_verdicts_all)
    if missing_real or missing_synth:
        raise SystemExit(f"missing arms -- real: {missing_real}, synthetic: {missing_synth}")

    real_summary = json.loads(REAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    real_headline_by_arm = {s["arm"]: s for s in real_summary["summaries"]}

    results_per_metric: dict[str, dict] = {}
    for arm_set_name, arm_set in (("13_arm", ARMS_13), ("5_arm_ladder", LADDER_5)):
        results_per_metric[arm_set_name] = {}
        for metric in METRICS:
            pred = _metric_predicate(metric)
            real_values_per_arm: dict[str, list[bool]] = {}
            real_clusters: list[str] | None = None
            synth_values_per_arm: dict[str, list[bool]] = {}
            skip = False
            for arm in arm_set:
                rv = [v for v in real_verdicts_all[arm] if pred(v)]
                sv = [v for v in synth_verdicts_all[arm] if pred(v)]
                if not rv or not sv:
                    skip = True
                    break
                real_values_per_arm[arm] = [bool(v.flagged) for v in rv]
                if real_clusters is None:
                    real_clusters = [getattr(v, "run_id", "") or v.instance_id for v in rv]
                synth_values_per_arm[arm] = [bool(v.flagged) for v in sv]
            if skip:
                results_per_metric[arm_set_name][metric] = {"skipped": "empty population for at least one arm"}
                continue

            real_rates = [sum(real_values_per_arm[a]) / len(real_values_per_arm[a]) for a in arm_set]
            synth_rates = [sum(synth_values_per_arm[a]) / len(synth_values_per_arm[a]) for a in arm_set]
            point = rank_agreement_point(real_rates, synth_rates)
            perm = permutation_pvalue(real_rates, synth_rates, n_perm=100_000, seed=42)
            ci = joint_bootstrap_rho_ci(
                real_values_per_arm, real_clusters, synth_values_per_arm, list(arm_set),
                n_boot=10_000, seed=42,
            )
            category = classify_reading_rule(point["spearman_rho"], ci["ci95"][0], perm["p_value"])
            results_per_metric[arm_set_name][metric] = {
                "real_rates": dict(zip(arm_set, real_rates)),
                "synthetic_rates": dict(zip(arm_set, synth_rates)),
                **point,
                "permutation_p_value": perm["p_value"],
                "rho_ci95": ci["ci95"],
                "rho_ci_n_boot": ci["n_boot"],
                "reading_rule_category": category,
            }
            print(
                f"[{arm_set_name}] {metric:30s} rho={point['spearman_rho']:+.4f} tau_b={point['kendall_tau_b']:+.4f} "
                f"p={perm['p_value']:.5f} CI={ci['ci95']} -> {category}"
            )

    # ── seed 0-9 stability check on the primary metric (pooled detection) ──
    stability = None
    if args.seed_sweep_summaries:
        real_pooled = {arm: _real_rate(real_headline_by_arm, arm, "pooled_detection") for arm in ARMS_13}
        per_seed_rho = []
        per_seed_category = []
        for path in args.seed_sweep_summaries:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            by_arm = {s["arm"]: s["headline"]["detection_rate_pooled"]["rate"] for s in payload["summaries"]}
            synth_pooled = [by_arm[a] for a in ARMS_13]
            rho = rank_agreement_point([real_pooled[a] for a in ARMS_13], synth_pooled)["spearman_rho"]
            per_seed_rho.append(rho)
        primary = results_per_metric["13_arm"]["pooled_detection"]
        for rho in per_seed_rho:
            cat = classify_reading_rule(rho, primary["rho_ci95"][0], primary["permutation_p_value"])
            per_seed_category.append(cat)
        stability = {
            "seed_files": args.seed_sweep_summaries,
            "per_seed_rho": per_seed_rho,
            "seed42_rho": primary["spearman_rho"],
            "per_seed_reading_rule_category_using_seed42_ci_and_p": per_seed_category,
            "category_stable_across_seeds": len(set(per_seed_category)) == 1,
        }
        print(f"\nSeed 0-9 pooled-detection rho stability: {per_seed_rho}")
        print(f"  category stable: {stability['category_stable_across_seeds']}")

    payload_out: dict[str, Any] = {
        "git_revision": _git_rev(),
        "frozen_config": frozen,
        "arms_13": list(ARMS_13),
        "ladder_5": list(LADDER_5),
        "metrics": METRICS,
        "results": results_per_metric,
        "seed_stability_check": stability,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "e4_rank_agreement.json"
    out_md = out_dir / "e4_rank_agreement.md"
    out_json.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload_out), encoding="utf-8")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


def render_markdown(payload: dict) -> str:
    lines = [
        "# E4 — rank agreement",
        "",
        f"- git revision: `{payload['git_revision']}`",
        f"- 13 arms: {payload['arms_13']}",
        "",
    ]
    for arm_set_name in ("13_arm", "5_arm_ladder"):
        lines.append(f"## {arm_set_name.replace('_', '-')}")
        lines.append("")
        lines.append("| metric | rho | tau-b | perm p (100k) | rho 95% CI (joint boot, 10k) | category |")
        lines.append("|---|---|---|---|---|---|")
        for metric, r in payload["results"][arm_set_name].items():
            if "skipped" in r:
                lines.append(f"| `{metric}` | — | — | — | — | skipped ({r['skipped']}) |")
                continue
            lines.append(
                f"| `{metric}` | {r['spearman_rho']:.4f} | {r['kendall_tau_b']:.4f} | "
                f"{r['permutation_p_value']:.5f} | [{r['rho_ci95'][0]:.4f}, {r['rho_ci95'][1]:.4f}] | "
                f"**{r['reading_rule_category']}** |"
            )
        lines.append("")
    if payload["seed_stability_check"]:
        s = payload["seed_stability_check"]
        lines.append("## Seed 0-9 stability (primary metric: pooled detection)")
        lines.append("")
        lines.append(f"- seed 42 rho: {s['seed42_rho']:.4f}")
        lines.append(f"- per-seed rho (0-9): {[round(r, 4) for r in s['per_seed_rho']]}")
        lines.append(f"- reading-rule category stable across seeds: **{s['category_stable_across_seeds']}**")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
