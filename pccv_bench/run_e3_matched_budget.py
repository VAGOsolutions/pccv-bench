"""PCCV-Bench agreement study, E3 — synthetic matched-budget protocol.

Mirrors the real tuning-only protocol (PREREGISTRATION.md §5):

1. Generate a tuning set (seed 1000, ``medium_v1_2_fitted.yaml``).
2. Budget = the frozen verifier's (``+all+twoway+offset_aware``, frozen
   config) clean flag rate on tuning.
3. For ``baseline_symmetric``/``baseline_coherent``, sweep threshold over the
   real study's own grid (``pccv_bench.core.matched_budget.
   THRESHOLD_GRID``), metric fixed to the frozen metric; pick the highest
   tuning pooled detection subject to tuning clean flag rate <= budget.
   ``missing_part_recall`` is never consulted.
4. Freeze both thresholds. Evaluate the frozen verifier and both baselines,
   at their frozen/selected points, on evaluation seeds 0-9 (raw instances,
   re-scored here rather than reusing the standard ladder's baseline arms,
   since the ladder's baselines sit at the frozen threshold, not the
   E3-selected one).

Paired differences (verifier minus each baseline) are computed instance-
bootstrapped, pooled across all 10 evaluation seeds -- position-paired within
each seed (never by ``instance_id``, which is only unique *within* one
seed's generation: ``synthetic_provider.py`` builds it from camera/attempt/
class alone, so the same string recurs across different seeds' runs) and
given a seed-qualified singleton cluster key, matching PREREGISTRATION.md
§2's "instance-level bootstrap ... on the synthetic side" (every synthetic
instance already has a unique ``run_id``, i.e. is already its own cluster).

The 36-point threshold sweep (18 grid points x 2 baselines) and the 10
evaluation seeds are each embarrassingly parallel -- every point/seed loads
its own instances and scores independently -- so both run across a process
pool (``--workers``, default: all cores) instead of one at a time.

Usage::

    python -m pccv_bench.run_e3_matched_budget \\
        --config pccv_bench/configs/medium_v1_2_fitted.yaml \\
        --seed-sweep-dir /tmp/pccv_seed_sweep \\
        --eval-seeds 0 1 2 3 4 5 6 7 8 9 \\
        --out-dir results/agreement_v2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from pccv_bench.core.matched_budget import THRESHOLD_GRID, _baseline_kwargs, pick_matched_threshold
from pccv_bench.core.harness import ArmConfig, run_arm
from pccv_bench.core.statistics import DEFAULT_N_BOOT, DEFAULT_SEED, paired_difference
from pccv_bench.ablate import build_arms
from pccv_bench.cameras import fit_synthetic_camera
from pccv_bench.cli import load_frozen_config
from pccv_bench.instance_io import load_instances

FROZEN_ARM_NAME = "+all+twoway+offset_aware"
BASELINES = ("symmetric", "coherent")

#: The 6 metrics E3 reports, as (label, keep(verdict)->bool, value(verdict)->bool).
POPULATIONS: list[tuple[str, Callable[[Any], bool], Callable[[Any], bool]]] = [
    ("clean_flag_rate", lambda v: v.injection_class == "clean", lambda v: v.flagged),
    ("missing_part_recall", lambda v: v.injection_class == "missing_part", lambda v: v.flagged),
    ("ghost_part_recall", lambda v: v.injection_class == "ghost_part", lambda v: v.flagged),
    ("pooled_detection", lambda v: v.expected_action == "alarm", lambda v: v.flagged),
    ("in_bound_flag_rate", lambda v: v.injection_class == "position_shift_x_inbound", lambda v: v.flagged),
    ("pooled_false_error", lambda v: v.expected_action == "pass", lambda v: v.flagged),
]


def _git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(_ROOT), text=True)
        return f"{rev}-dirty" if dirty else rev
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _plain_rate(verdicts: list, keep, value) -> float | None:
    pool = [v for v in verdicts if keep(v)]
    return None if not pool else sum(1 for v in pool if value(v)) / len(pool)


def _mappers_for(instances, gen_meta: dict) -> dict:
    """Refit each camera's mapper exactly as ``cli.py``'s ``cmd_ablate`` does:
    same seed the instances were generated with (not an arbitrary one --
    ``fit_synthetic_camera`` draws from the rng, so a different seed gives a
    different, inconsistent calibration for the same nominal camera, silently
    corrupting every arm that re-projects world-space parts)."""
    cfg = gen_meta.get("config", {})
    seed = gen_meta.get("seed", 42)
    cam_rng = np.random.default_rng(seed)
    cameras = cfg.get("cameras") or sorted({i.camera for i in instances})
    return {c: fit_synthetic_camera(cam_rng, c) for c in cameras}


def _load_with_mappers(instances_path: str) -> tuple[list, dict]:
    instances, gen_meta = load_instances(Path(instances_path))
    return instances, _mappers_for(instances, gen_meta)


def generate_only(config: str, seed: int, out_dir: Path) -> Path:
    """Just the instances file -- E3's tuning set (and each eval seed, scored
    with E3's own custom ArmConfigs here) needs only raw instances, not a
    full 15-arm ladder ablate."""
    stem = Path(config).stem
    instances_path = out_dir / f"instances_{stem}_seed{seed}.json"
    if not instances_path.is_file():
        subprocess.run(
            [
                sys.executable, "-m", "pccv_bench.cli", "generate",
                "--config", config, "--seed", str(seed), "--out", str(instances_path),
            ],
            cwd=str(_ROOT), check=True,
        )
    else:
        print(f"  [skip generate] {instances_path} exists")
    return instances_path


# ── parallel worker functions (module level: must be picklable) ────────────


def _sweep_point_worker(args: tuple[str, str, str, float]) -> dict:
    """One (baseline, threshold) point of the tuning sweep -- runs in its own
    process, reloading the tuning instances itself (cheap: one JSON parse)
    rather than pickling ~1,680 Instance objects into every worker."""
    tuning_instances_path, metric, baseline, threshold = args
    instances, mappers = _load_with_mappers(tuning_instances_path)
    arm = ArmConfig(name=f"sweep_{baseline}_{threshold}", metric=metric, threshold=threshold, baseline=baseline, **_baseline_kwargs(baseline))
    vs = run_arm(instances, arm, mappers=mappers)
    return {
        "baseline": baseline,
        "threshold": threshold,
        "clean_false_error": _plain_rate(vs, POPULATIONS[0][1], POPULATIONS[0][2]),
        "detection_rate_pooled": _plain_rate(vs, POPULATIONS[3][1], POPULATIONS[3][2]),
        "missing_part_recall": _plain_rate(vs, POPULATIONS[1][1], POPULATIONS[1][2]),
    }


def _eval_seed_worker(args: tuple[str, dict, dict]) -> dict:
    """One evaluation seed: verifier + both matched baselines, run in its own
    process. Returns per-metric rates plus the raw (instance-ordered)
    flagged/kept booleans for every metric x method, so the parent process
    can pool paired differences across seeds without re-scoring."""
    instances_path, verifier_kwargs, baseline_thresholds = args
    instances, mappers = _load_with_mappers(instances_path)

    verifier_arm = ArmConfig(**verifier_kwargs)
    arms = {"verifier": verifier_arm}
    for b, t in baseline_thresholds.items():
        arms[b] = ArmConfig(name=f"baseline_{b}_matched", metric=verifier_kwargs["metric"], threshold=t, baseline=b, **_baseline_kwargs(b))

    verdicts = {name: run_arm(instances, arm, mappers=mappers) for name, arm in arms.items()}
    for name, vs in verdicts.items():
        assert [v.instance_id for v in vs] == [v.instance_id for v in verdicts["verifier"]], (
            f"{name} scored a different/reordered instance set than verifier"
        )

    rates = {name: {label: _plain_rate(vs, keep, value) for label, keep, value in POPULATIONS} for name, vs in verdicts.items()}
    flags = {
        name: {label: [bool(keep(v)) and bool(value(v)) if keep(v) else None for v in vs] for label, keep, value in POPULATIONS}
        for name, vs in verdicts.items()
    }
    keep_masks = {label: [bool(keep(v)) for v in verdicts["verifier"]] for label, keep, value in POPULATIONS}
    return {"rates": rates, "flags": flags, "keep_masks": keep_masks, "n_instances": len(instances)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(_ROOT / "pccv_bench" / "configs" / "medium_v1_2_fitted.yaml"))
    ap.add_argument("--seed-sweep-dir", required=True, help="dir with seed-swept instances (run_seed_sweep.py)")
    ap.add_argument("--tuning-seed", type=int, default=1000)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    frozen = load_frozen_config()
    sweep_dir = Path(args.seed_sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # ── tuning set (seed 1000) ────────────────────────────────────────────
    print(f"Building tuning set at seed {args.tuning_seed}…")
    tuning_instances_path = generate_only(args.config, args.tuning_seed, sweep_dir)
    tuning_instances, tuning_meta = load_instances(tuning_instances_path)
    tuning_mappers = _mappers_for(tuning_instances, tuning_meta)
    print(f"  {len(tuning_instances)} tuning instances")

    verifier_arm = next(a for a in build_arms(frozen=frozen) if a.name == FROZEN_ARM_NAME)
    verifier_vs = run_arm(tuning_instances, verifier_arm, mappers=tuning_mappers)
    budget = _plain_rate(verifier_vs, POPULATIONS[0][1], POPULATIONS[0][2])
    print(f"Budget (frozen verifier's own tuning clean flag rate): {budget:.4f}")

    # ── baseline threshold sweeps on tuning, in parallel ──────────────────
    print(f"Sweeping {len(BASELINES)} baselines x {len(THRESHOLD_GRID)} thresholds on tuning ({args.workers} workers)…")
    sweep_jobs = [
        (str(tuning_instances_path), frozen["metric"], baseline, t)
        for baseline in BASELINES
        for t in THRESHOLD_GRID
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        sweep_results = list(pool.map(_sweep_point_worker, sweep_jobs))

    sweep_rows: dict[str, list[dict]] = {b: [] for b in BASELINES}
    for row in sweep_results:
        sweep_rows[row["baseline"]].append(row)
    for b in BASELINES:
        sweep_rows[b].sort(key=lambda r: THRESHOLD_GRID.index(r["threshold"]))

    selected: dict[str, dict] = {}
    for baseline in BASELINES:
        pick = pick_matched_threshold(sweep_rows[baseline], budget)
        if pick is None:
            raise SystemExit(f"no tuning threshold for baseline_{baseline} reaches budget {budget:.4f}")
        selected[baseline] = pick
        print(f"  {baseline}: selected threshold={pick['threshold']:g} (tuning clean_fer={pick['clean_false_error']:.4f}, det_pooled={pick['detection_rate_pooled']:.4f})")

    # ── evaluate on seeds 0-9, in parallel ─────────────────────────────────
    print(f"Evaluating {len(args.eval_seeds)} seeds ({args.workers} workers)…")
    verifier_kwargs = {
        "name": verifier_arm.name, "metric": verifier_arm.metric, "threshold": verifier_arm.threshold,
        "correction_offset_mm": verifier_arm.correction_offset_mm, "use_temporal": verifier_arm.use_temporal,
        "twoway": verifier_arm.twoway, "twoway_threshold": verifier_arm.twoway_threshold,
        "twoway_offset_aware": verifier_arm.twoway_offset_aware,
    }
    baseline_thresholds = {b: selected[b]["threshold"] for b in BASELINES}
    eval_jobs = []
    for seed in args.eval_seeds:
        inst_path = generate_only(args.config, seed, sweep_dir)
        eval_jobs.append((str(inst_path), verifier_kwargs, baseline_thresholds))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        eval_results = list(pool.map(_eval_seed_worker, eval_jobs))

    per_seed: dict[str, dict[int, dict]] = {"verifier": {}, "symmetric": {}, "coherent": {}}
    paired_pool: dict[str, dict[str, dict[str, list]]] = {
        b: {label: {"verifier": [], "baseline": [], "clusters": []} for label, _, _ in POPULATIONS} for b in BASELINES
    }
    for seed, result in zip(args.eval_seeds, eval_results):
        for method in ("verifier", *BASELINES):
            per_seed[method][seed] = result["rates"][method]
        n = result["n_instances"]
        for b in BASELINES:
            for label, _, _ in POPULATIONS:
                mask = result["keep_masks"][label]
                fv = result["flags"]["verifier"][label]
                fb = result["flags"][b][label]
                for i in range(n):
                    if mask[i]:
                        paired_pool[b][label]["verifier"].append(bool(fv[i]))
                        paired_pool[b][label]["baseline"].append(bool(fb[i]))
                        paired_pool[b][label]["clusters"].append(f"seed{seed}:{i}")

    def _mean_range(vals: list[float]) -> dict:
        arr = [v for v in vals if v is not None]
        return {"mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)), "per_seed": vals}

    summary = {}
    for method in ("verifier", *BASELINES):
        summary[method] = {}
        for label, _, _ in POPULATIONS:
            vals = [per_seed[method][s][label] for s in args.eval_seeds]
            summary[method][label] = _mean_range(vals)

    paired_diffs: dict[str, dict[str, dict]] = {b: {} for b in BASELINES}
    for b in BASELINES:
        for label, _, _ in POPULATIONS:
            p = paired_pool[b][label]
            paired_diffs[b][label] = paired_difference(
                p["verifier"], p["baseline"], p["clusters"],
                n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED,
            )

    payload: dict[str, Any] = {
        "git_revision": _git_rev(),
        "tuning_seed": args.tuning_seed,
        "eval_seeds": args.eval_seeds,
        "frozen_config": frozen,
        "budget": budget,
        "budget_source": "frozen verifier's (+all+twoway+offset_aware) own tuning clean flag rate",
        "selected_thresholds": {b: selected[b]["threshold"] for b in BASELINES},
        "tuning_selection": selected,
        "sweep_rows": sweep_rows,
        "per_seed_and_mean": summary,
        "paired_differences_verifier_minus_baseline": paired_diffs,
        "note": (
            "Paired differences are pooled across all 10 evaluation seeds, position-paired "
            "within each seed's instance list (not by instance_id, which is only unique "
            "within one seed) and instance-bootstrapped (each synthetic instance is its own "
            "cluster, per PREREGISTRATION.md §2)."
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "e3_matched_budget.json"
    out_md = out_dir / "e3_matched_budget.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


def render_markdown(payload: dict) -> str:
    lines = [
        "# E3 — synthetic matched-budget protocol",
        "",
        f"- git revision: `{payload['git_revision']}`",
        f"- tuning seed: {payload['tuning_seed']}; evaluation seeds: {payload['eval_seeds']}",
        f"- budget: **{payload['budget']:.4f}** ({payload['budget_source']})",
        f"- selected thresholds: symmetric={payload['selected_thresholds']['symmetric']:g}, "
        f"coherent={payload['selected_thresholds']['coherent']:g}",
        "",
        "## Per-seed and mean (verifier / symmetric / coherent)",
        "",
        "| metric | method | " + " | ".join(f"seed {s}" for s in payload["eval_seeds"]) + " | mean | range |",
        "|---|---|" + "---|" * len(payload["eval_seeds"]) + "---|---|",
    ]
    for label, _, _ in POPULATIONS:
        for method in ("verifier", "symmetric", "coherent"):
            e = payload["per_seed_and_mean"][method][label]
            row = " | ".join(f"{v:.4f}" if v is not None else "—" for v in e["per_seed"])
            lines.append(f"| `{label}` | {method} | {row} | {e['mean']:.4f} | [{e['min']:.4f}, {e['max']:.4f}] |")
    lines += [
        "",
        "## Paired differences (verifier minus baseline), pooled across seeds, instance-bootstrap 95% CI",
        "",
        "| baseline | metric | difference | 95% CI | significant |",
        "|---|---|---|---|---|",
    ]
    for b in ("symmetric", "coherent"):
        for label, _, _ in POPULATIONS:
            d = payload["paired_differences_verifier_minus_baseline"][b][label]
            ci = d["ci95"]
            lines.append(
                f"| {b} | `{label}` | {d['difference']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
                f"{'**yes**' if d['significant'] else 'no'} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
