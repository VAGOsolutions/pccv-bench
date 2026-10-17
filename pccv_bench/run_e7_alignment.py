"""PCCV-Bench agreement study, E7 — operating-point alignment.

Recomputes the 5-arm ladder's Spearman rho and the two-arm clean
false-error comparison (``+all+twoway``, ``+all+twoway+offset_aware``)
against the real study (``injection_summary.json``), once at the
pre-existing (misaligned) operating point and once with the seed-42 medium
v1.2 run rescored at the real study's frozen point
(``pccv_bench.cli ablate --frozen-operating-point``), and reports both side
by side. See ``docs/PREREGISTRATION.md`` §1.

Usage::

    python -m pccv_bench.run_e7_alignment \\
        --old results/results_medium_v1_2_fitted.json \\
        --new /tmp/e7_results_medium_v1_2_fitted_aligned.json \\
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

from pccv_bench.release_validation import rank_correlation

LADDER_5 = ("geometry", "+offset", "+temporal", "+offset+temporal", "+all+twoway")
CLEAN_FER_ARMS = ("+all+twoway", "+all+twoway+offset_aware")


def _git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(_ROOT), text=True)
        return f"{rev}-dirty" if dirty else rev
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _by_arm(payload: dict) -> dict[str, dict]:
    return {s["arm"]: s["headline"] for s in payload["summaries"]}


def _rate(headline: dict, key: str) -> float:
    return headline[key]["rate"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", required=True, help="pre-E7 (misaligned) results JSON")
    ap.add_argument("--new", required=True, help="post-E7 (frozen-aligned) results JSON")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--real-summary", required=True, help="injection_summary.json of the real-data study (not released)")
    args = ap.parse_args()
    real_summary_path = Path(args.real_summary)

    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    real = json.loads(real_summary_path.read_text(encoding="utf-8"))

    old_by_arm, new_by_arm, real_by_arm = _by_arm(old), _by_arm(new), _by_arm(real)

    assert new["frozen_operating_point"] is not None, "--new must be an aligned (--frozen-operating-point) run"
    assert old.get("frozen_operating_point") is None, "--old must be the pre-E7, unaligned run"

    real_detect = {a: _rate(real_by_arm[a], "detection_rate_pooled") for a in LADDER_5}
    old_detect = {a: _rate(old_by_arm[a], "detection_rate_pooled") for a in LADDER_5}
    new_detect = {a: _rate(new_by_arm[a], "detection_rate_pooled") for a in LADDER_5}

    old_rho = rank_correlation(real_detect, old_detect)
    new_rho = rank_correlation(real_detect, new_detect)

    clean_fer_rows = []
    for arm in CLEAN_FER_ARMS:
        clean_fer_rows.append(
            {
                "arm": arm,
                "real": _rate(real_by_arm[arm], "false_error_rate_clean_only"),
                "synthetic_old_misaligned": _rate(old_by_arm[arm], "false_error_rate_clean_only"),
                "synthetic_new_aligned": _rate(new_by_arm[arm], "false_error_rate_clean_only"),
            }
        )

    detect_rows = []
    for arm in LADDER_5:
        detect_rows.append(
            {
                "arm": arm,
                "real": real_detect[arm],
                "synthetic_old_misaligned": old_detect[arm],
                "synthetic_new_aligned": new_detect[arm],
            }
        )

    payload: dict[str, Any] = {
        "git_revision": _git_rev(),
        "real_reference": {
            "path": real_summary_path.name,
            "git_revision": real.get("git_revision"),
            "n_instances": real.get("n_instances"),
        },
        "old_run": {
            "path": args.old,
            "git_revision": old.get("git_revision"),
            "frozen_operating_point": old.get("frozen_operating_point"),
        },
        "new_run": {
            "path": args.new,
            "git_revision": new.get("git_revision"),
            "frozen_operating_point": new.get("frozen_operating_point"),
        },
        "ladder_5_arms": list(LADDER_5),
        "spearman_rho": {
            "old_misaligned": old_rho.spearman_rho,
            "new_aligned": new_rho.spearman_rho,
        },
        "pooled_detection_rate": detect_rows,
        "clean_false_error_rate": clean_fer_rows,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "e7_operating_point_alignment.json"
    out_md = out_dir / "e7_operating_point_alignment.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"old (misaligned) 5-arm rho: {old_rho.spearman_rho:.4f}")
    print(f"new (aligned)    5-arm rho: {new_rho.spearman_rho:.4f}")
    for row in clean_fer_rows:
        print(
            f"{row['arm']:26s} real={row['real']:.4f}  "
            f"old_misaligned={row['synthetic_old_misaligned']:.4f}  "
            f"new_aligned={row['synthetic_new_aligned']:.4f}"
        )
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


def render_markdown(payload: dict) -> str:
    lines = [
        "# E7 — operating-point alignment: old vs new",
        "",
        f"- git revision: `{payload['git_revision']}`",
        f"- real reference: `{payload['real_reference']['path']}` "
        f"(git_revision `{payload['real_reference']['git_revision']}`, "
        f"{payload['real_reference']['n_instances']} instances)",
        f"- old (misaligned) run git revision: `{payload['old_run']['git_revision']}`",
        f"- new (aligned) run git revision: `{payload['new_run']['git_revision']}`, "
        f"frozen point: {payload['new_run']['frozen_operating_point']}",
        "",
        "## 5-arm ladder Spearman rho (geometry, +offset, +temporal, +offset+temporal, +all+twoway)",
        "",
        f"| | rho |",
        f"|---|---|",
        f"| old (misaligned) | {payload['spearman_rho']['old_misaligned']:.4f} |",
        f"| new (aligned) | {payload['spearman_rho']['new_aligned']:.4f} |",
        "",
        "## Pooled detection rate, per arm",
        "",
        "| arm | real | synthetic (old, misaligned) | synthetic (new, aligned) |",
        "|---|---|---|---|",
    ]
    for row in payload["pooled_detection_rate"]:
        lines.append(
            f"| `{row['arm']}` | {row['real']:.4f} | {row['synthetic_old_misaligned']:.4f} | "
            f"{row['synthetic_new_aligned']:.4f} |"
        )
    lines += [
        "",
        "## Clean false-error rate (presumed-clean disagreement), per arm",
        "",
        "| arm | real | synthetic (old, misaligned) | synthetic (new, aligned) |",
        "|---|---|---|---|",
    ]
    for row in payload["clean_false_error_rate"]:
        lines.append(
            f"| `{row['arm']}` | {row['real']:.4f} | {row['synthetic_old_misaligned']:.4f} | "
            f"{row['synthetic_new_aligned']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
