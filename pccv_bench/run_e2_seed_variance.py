"""PCCV-Bench agreement study, E2 — seed variance (PREREGISTRATION.md §6).

Aggregates the aligned medium-tier ladder across evaluation seeds 0-9
(``run_seed_sweep.py``'s output): for every arm, the mean and range of
pooled detection, clean flag rate, and each of the seven per-class flag
rates.

Usage::

    python -m pccv_bench.run_e2_seed_variance \\
        --seed-sweep-dir /tmp/pccv_seed_sweep \\
        --config-stem medium_v1_2_fitted \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --out-dir results/agreement_v2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from pccv_bench.core.inject import INJECTION_CLASSES

METRICS = ["pooled_detection", "clean_flag_rate"] + [f"class:{c}" for c in INJECTION_CLASSES if c != "clean"]


def _git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(_ROOT), text=True)
        return f"{rev}-dirty" if dirty else rev
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _rate(headline_and_per_class: dict, metric: str) -> float | None:
    if metric == "pooled_detection":
        return headline_and_per_class["headline"]["detection_rate_pooled"]["rate"]
    if metric == "clean_flag_rate":
        return headline_and_per_class["headline"]["false_error_rate_clean_only"]["rate"]
    cls = metric.split(":", 1)[1]
    entry = headline_and_per_class["per_class"].get(cls)
    return entry["flag_rate"]["rate"] if entry else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed-sweep-dir", required=True)
    ap.add_argument("--config-stem", default="medium_v1_2_fitted")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sweep_dir = Path(args.seed_sweep_dir)
    per_seed_payloads = {}
    for seed in args.seeds:
        path = sweep_dir / f"results_{args.config_stem}_seed{seed}_aligned.json"
        if not path.is_file():
            raise SystemExit(f"missing {path} -- run run_seed_sweep.py first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("frozen_operating_point") is not None, f"{path} is not aligned"
        per_seed_payloads[seed] = payload

    arm_names = [s["arm"] for s in per_seed_payloads[args.seeds[0]]["summaries"]]
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in arm_names:
        per_arm[arm] = {}
        for metric in METRICS:
            vals = []
            for seed in args.seeds:
                by_arm = {s["arm"]: s for s in per_seed_payloads[seed]["summaries"]}
                vals.append(_rate(by_arm[arm], metric))
            defined = [v for v in vals if v is not None]
            per_arm[arm][metric] = {
                "per_seed": vals,
                "mean": float(np.mean(defined)) if defined else None,
                "min": float(np.min(defined)) if defined else None,
                "max": float(np.max(defined)) if defined else None,
                "range": float(np.max(defined) - np.min(defined)) if defined else None,
            }

    payload_out = {
        "git_revision": _git_rev(),
        "seeds": args.seeds,
        "seed_git_revisions": {s: per_seed_payloads[s].get("git_revision") for s in args.seeds},
        "metrics": METRICS,
        "per_arm": per_arm,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "e2_seed_variance.json"
    out_md = out_dir / "e2_seed_variance.md"
    out_json.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload_out), encoding="utf-8")
    for arm in ("+all+twoway+offset_aware",):
        for metric in ("pooled_detection", "clean_flag_rate"):
            e = per_arm[arm][metric]
            print(f"{arm} {metric}: mean={e['mean']:.4f} range=[{e['min']:.4f}, {e['max']:.4f}]")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


def render_markdown(payload: dict) -> str:
    lines = [
        "# E2 — seed variance (medium, aligned, seeds 0-9)",
        "",
        f"- git revision: `{payload['git_revision']}`",
        f"- seeds: {payload['seeds']}",
        "",
        "| arm | metric | mean | min | max | range |",
        "|---|---|---|---|---|---|",
    ]
    for arm, metrics in payload["per_arm"].items():
        for metric, e in metrics.items():
            if e["mean"] is None:
                lines.append(f"| `{arm}` | `{metric}` | — | — | — | — |")
                continue
            lines.append(
                f"| `{arm}` | `{metric}` | {e['mean']:.4f} | {e['min']:.4f} | {e['max']:.4f} | {e['range']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
