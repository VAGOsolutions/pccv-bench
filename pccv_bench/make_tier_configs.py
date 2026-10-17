"""Derive easy_v1_2.yaml / hard_v1_2.yaml from medium_v1_2_fitted.yaml — E1.

PREREGISTRATION.md §6: "Tiers are defined relative to the fitted noise rather
than the hand-set v0.1: easy: jitter, miss, FP, and merge rates x0.5; medium:
v1.2-fitted as is; hard: x1.5, with rates capped at 1. Occlusion rate 0.05 /
0.10 / 0.20."

Scaled fields: ``jitter_px_sigma``, ``miss_rate``, ``fp_rate``, ``merge_rate``
— in both the pooled ``noise:`` block and every camera's ``per_camera_noise``
entry. The cap at 1 applies only to the three *rate* fields (probabilities);
``jitter_px_sigma`` is a pixel sigma, not a rate, so it is scaled unclipped.
``split_rate``, ``fp_size_frac``, ``score_mean``/``score_sigma``,
``noise_version``, ``n_instances_per_class``, ``seed``, ``twoway_threshold``,
and ``cameras`` are copied unchanged — the preregistration's field list
excludes them.

Run once; the outputs are committed under ``pccv_bench/configs/``, not
regenerated at every invocation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SOURCE = _ROOT / "pccv_bench" / "configs" / "medium_v1_2_fitted.yaml"
SCALED_RATE_FIELDS = ("miss_rate", "fp_rate", "merge_rate")
SCALED_FIELDS = SCALED_RATE_FIELDS + ("jitter_px_sigma",)

TIERS = {
    "easy": {"factor": 0.5, "occlusion_rate": 0.05, "suffix": "easy_v1_2"},
    "hard": {"factor": 1.5, "occlusion_rate": 0.20, "suffix": "hard_v1_2"},
}

HEADER = """# PCCV-Bench {tier} tier, v1.2-fitted-derived (E1, PREREGISTRATION.md §6).
# Derived from medium_v1_2_fitted.yaml by pccv_bench/make_tier_configs.py:
# jitter_px_sigma, miss_rate, fp_rate, and merge_rate (pooled block and every
# per_camera_noise entry) scaled x{factor}{cap_note}. occlusion_rate set to
# {occlusion_rate}. Every other field (split_rate, fp_size_frac, score_mean,
# score_sigma, noise_version, n_instances_per_class, seed, twoway_threshold,
# cameras) is unchanged from medium_v1_2_fitted.yaml.
"""


def _scale_block(block: dict, factor: float) -> dict:
    out = dict(block)
    for field in SCALED_FIELDS:
        if field not in block:
            continue
        scaled = block[field] * factor
        if field in SCALED_RATE_FIELDS:
            scaled = min(scaled, 1.0)
        out[field] = scaled
    return out


def make_tier_config(source: dict, factor: float, occlusion_rate: float) -> dict:
    cfg = dict(source)
    cfg["noise"] = _scale_block(source["noise"], factor)
    if "per_camera_noise" in source:
        cfg["per_camera_noise"] = {
            cam: _scale_block(cam_cfg, factor) for cam, cam_cfg in source["per_camera_noise"].items()
        }
    cfg["occlusion_rate"] = occlusion_rate
    return cfg


def main() -> None:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    for tier, spec in TIERS.items():
        cfg = make_tier_config(source, spec["factor"], spec["occlusion_rate"])
        cap_note = ", capped at 1" if spec["factor"] > 1 else ""
        header = HEADER.format(
            tier=tier, factor=spec["factor"], cap_note=cap_note, occlusion_rate=spec["occlusion_rate"]
        )
        body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        out_path = _ROOT / "pccv_bench" / "configs" / f"{spec['suffix']}.yaml"
        out_path.write_text(header + "\n" + body, encoding="utf-8")
        print(f"wrote {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
