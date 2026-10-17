"""PCCV-Bench agreement study, E1 — tiers.

Reports the full aligned ladder across easy/medium/hard (seed 42), and the
pre-registered tier-expectation check (PREREGISTRATION.md §6): for the
frozen verifier (``+all+twoway+offset_aware``), pooled detection must be
non-increasing and clean flag rate non-decreasing from easy to hard. Checked,
not tuned -- a failure is reported, not silently fixed.

Usage::

    python -m pccv_bench.run_e1_tiers \\
        --easy /tmp/e1_results_easy_v1_2_aligned.json \\
        --medium results/results_medium_v1_2_fitted_aligned.json \\
        --hard /tmp/e1_results_hard_v1_2_aligned.json \\
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

FROZEN_ARM = "+all+twoway+offset_aware"
TIER_ORDER = ("easy", "medium", "hard")


def _git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(_ROOT), text=True)
        return f"{rev}-dirty" if dirty else rev
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _by_arm(payload: dict) -> dict[str, dict]:
    return {s["arm"]: s["headline"] for s in payload["summaries"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--easy", required=True)
    ap.add_argument("--medium", required=True)
    ap.add_argument("--hard", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    payloads = {
        "easy": json.loads(Path(args.easy).read_text(encoding="utf-8")),
        "medium": json.loads(Path(args.medium).read_text(encoding="utf-8")),
        "hard": json.loads(Path(args.hard).read_text(encoding="utf-8")),
    }
    for tier, p in payloads.items():
        assert p.get("frozen_operating_point") is not None, f"{tier} run must be --frozen-operating-point aligned"

    by_arm = {tier: _by_arm(p) for tier, p in payloads.items()}
    arm_names = list(by_arm["medium"].keys())

    ladder: dict[str, dict[str, dict]] = {}
    for arm in arm_names:
        ladder[arm] = {}
        for tier in TIER_ORDER:
            h = by_arm[tier][arm]
            ladder[arm][tier] = {
                "detection_rate_pooled": h["detection_rate_pooled"]["rate"],
                "false_error_rate_clean_only": h["false_error_rate_clean_only"]["rate"],
            }

    frozen_detect = [ladder[FROZEN_ARM][t]["detection_rate_pooled"] for t in TIER_ORDER]
    frozen_clean = [ladder[FROZEN_ARM][t]["false_error_rate_clean_only"] for t in TIER_ORDER]
    detect_non_increasing = all(frozen_detect[i] >= frozen_detect[i + 1] for i in range(len(frozen_detect) - 1))
    clean_non_decreasing = all(frozen_clean[i] <= frozen_clean[i + 1] for i in range(len(frozen_clean) - 1))

    payload: dict[str, Any] = {
        "git_revision": _git_rev(),
        "frozen_arm": FROZEN_ARM,
        "tier_order": list(TIER_ORDER),
        "tier_git_revisions": {t: payloads[t].get("git_revision") for t in TIER_ORDER},
        "tier_instance_git_revisions": {t: payloads[t]["generation_meta"].get("git_revision") for t in TIER_ORDER},
        "full_ladder": ladder,
        "tier_expectation_check": {
            "pooled_detection": frozen_detect,
            "pooled_detection_non_increasing_easy_to_hard": detect_non_increasing,
            "clean_flag_rate": frozen_clean,
            "clean_flag_rate_non_decreasing_easy_to_hard": clean_non_decreasing,
            "both_hold": detect_non_increasing and clean_non_decreasing,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "e1_tiers.json"
    out_md = out_dir / "e1_tiers.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"frozen verifier ({FROZEN_ARM}) pooled detection, easy->medium->hard: {frozen_detect}")
    print(f"  non-increasing: {detect_non_increasing}")
    print(f"frozen verifier clean flag rate, easy->medium->hard: {frozen_clean}")
    print(f"  non-decreasing: {clean_non_decreasing}")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


def render_markdown(payload: dict) -> str:
    chk = payload["tier_expectation_check"]
    lines = [
        "# E1 — tiers (easy / medium / hard), aligned to the frozen operating point",
        "",
        f"- git revision: `{payload['git_revision']}`",
        f"- tier git revisions: {payload['tier_git_revisions']}",
        "",
        f"## Tier-expectation check (frozen verifier `{payload['frozen_arm']}`)",
        "",
        f"- pooled detection (easy, medium, hard): {chk['pooled_detection']} "
        f"-- non-increasing: **{chk['pooled_detection_non_increasing_easy_to_hard']}**",
        f"- clean flag rate (easy, medium, hard): {chk['clean_flag_rate']} "
        f"-- non-decreasing: **{chk['clean_flag_rate_non_decreasing_easy_to_hard']}**",
        f"- both hold: **{chk['both_hold']}**",
        "",
        "## Full ladder",
        "",
        "| arm | easy detect | medium detect | hard detect | easy clean | medium clean | hard clean |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, tiers in payload["full_ladder"].items():
        row = [arm]
        for metric in ("detection_rate_pooled",):
            for t in ("easy", "medium", "hard"):
                row.append(f"{tiers[t][metric]:.4f}")
        for t in ("easy", "medium", "hard"):
            row.append(f"{tiers[t]['false_error_rate_clean_only']:.4f}")
        lines.append("| " + " | ".join(f"`{row[0]}`" if i == 0 else row[i] for i in range(len(row))) + " |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
