"""PCCV-Bench CLI.

Anyone should be able to run::

    python -m pccv_bench.cli generate --config pccv_bench/configs/medium_v1_2_fitted.yaml --seed 0 --out instances.json
    python -m pccv_bench.cli ablate   --instances instances.json --out results.json
    python -m pccv_bench.cli report   --results results.json

and land within CI tolerance of the published numbers, from the README alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pccv_bench.core.evaluate import (
    format_per_class_table,
    format_summary_table,
    summarise_arm,
)
from pccv_bench.core.inject import INJECTION_CLASSES
from pccv_bench.ablate import run_all_arms
from pccv_bench.core.manifest import (
    GENERATOR_VERSION,
    manifest_checksum,
    save_manifest,
)
from pccv_bench.instance_io import _json_default, load_instances, save_instances
from pccv_bench.noise import NOISE_VERSION, NoiseParams
from pccv_bench.synthetic_provider import SyntheticProvider

#: E7 (operating-point alignment, PREREGISTRATION.md §1): the companion
#: real-data study's frozen operating point. Fixed path, not a runtime
#: argument -- there is exactly one real study to align to.
FROZEN_CONFIG_PATH = Path(__file__).resolve().parent / "frozen_operating_point.json"


def load_frozen_config(path: Path = FROZEN_CONFIG_PATH) -> dict:
    """The real study's ``{"metric", "threshold", "twoway_threshold", ...}``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["frozen"]


def _git_rev() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True
        ).strip()
        dirty_paths = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(_ROOT), text=True
        )
        return f"{revision}-dirty" if dirty_paths else revision
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _config_hash(config: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def noise_version_for_config(config: dict) -> str:
    """Return the explicit version attached to a benchmark configuration."""
    configured_version = config.get("noise_version", NOISE_VERSION)
    if not isinstance(configured_version, str) or not configured_version:
        raise ValueError("noise_version must be a non-empty string")
    return configured_version


# ── generate ─────────────────────────────────────────────────────────────────


def _warn_if_untested_opencv() -> None:
    """OpenCV 4.14 and later change the fitted camera model, so their instances differ from the pinned set."""
    import cv2

    major, minor = (int(part) for part in cv2.__version__.split(".")[:2])
    if (major, minor) >= (4, 14):
        print(
            f"WARNING: OpenCV {cv2.__version__} is outside the supported range (4.5 to 4.13). "
            "Generated instances will not match the pinned set and results will differ.",
            file=sys.stderr,
        )



def cmd_generate(args: argparse.Namespace) -> None:
    _warn_if_untested_opencv()
    config = load_config(Path(args.config))
    if args.seed is not None:
        config["seed"] = args.seed
    seed = config.get("seed", 42)
    n = config.get("n_instances_per_class", 120)
    cameras = tuple(config.get("cameras") or [])
    noise_cfg = config.get("noise", {})
    pooled_noise = NoiseParams(
        version=noise_version_for_config(config),
        miss_rate=noise_cfg.get("miss_rate", 0.03),
        fp_rate=noise_cfg.get("fp_rate", 0.03),
        fp_size_frac=tuple(noise_cfg.get("fp_size_frac", [0.01, 0.05])),
        jitter_px_sigma=noise_cfg.get("jitter_px_sigma", 4.0),
        split_rate=noise_cfg.get("split_rate", 0.05),
        merge_rate=noise_cfg.get("merge_rate", 0.05),
        score_mean=noise_cfg.get("score_mean", 0.75),
        score_sigma=noise_cfg.get("score_sigma", 0.12),
    )
    # Optional per-camera overrides (pccv_bench/noise.py's dict convention):
    # a config may set `per_camera_noise: {camera: {field: value, ...}, ...}`
    # alongside the pooled `noise:` block above. Any field a camera's entry
    # doesn't set falls back to that same pooled block (not NoiseParams'
    # hand-set v0.1 dataclass defaults), since a config author overriding
    # only e.g. jitter_px_sigma per camera should not silently lose their
    # own pooled miss_rate/fp_rate for the fields they didn't override.
    per_camera_cfg = config.get("per_camera_noise")
    noise: NoiseParams | dict[str, NoiseParams]
    if per_camera_cfg:
        noise = {
            camera: NoiseParams(
                version=noise_version_for_config(config),
                miss_rate=cam_cfg.get("miss_rate", pooled_noise.miss_rate),
                fp_rate=cam_cfg.get("fp_rate", pooled_noise.fp_rate),
                fp_size_frac=tuple(
                    cam_cfg.get("fp_size_frac", list(pooled_noise.fp_size_frac))
                ),
                jitter_px_sigma=cam_cfg.get(
                    "jitter_px_sigma", pooled_noise.jitter_px_sigma
                ),
                split_rate=cam_cfg.get("split_rate", pooled_noise.split_rate),
                merge_rate=cam_cfg.get("merge_rate", pooled_noise.merge_rate),
                score_mean=cam_cfg.get("score_mean", pooled_noise.score_mean),
                score_sigma=cam_cfg.get("score_sigma", pooled_noise.score_sigma),
            )
            for camera, cam_cfg in per_camera_cfg.items()
        }
    else:
        noise = pooled_noise
    occlusion_rate = config.get("occlusion_rate", 0.10)

    print(
        f"Generating from {args.config} (seed={seed}, n/class={n}, cameras={cameras})"
    )
    mappers: dict = {}
    all_instances = []
    rejections: dict[str, int] = {}
    for cls in INJECTION_CLASSES:
        provider = SyntheticProvider(
            cls,
            n_instances=n,
            seed=seed,
            cameras=cameras or None,
            noise=noise,
            occlusion_rate=occlusion_rate,
            mappers=mappers if mappers else None,
        )
        if not mappers:
            mappers = provider.mappers
        instances = list(provider)
        all_instances.extend(instances)
        rejections[cls] = len(provider.rejected)
        print(
            f"  {cls:28s} generated={len(instances):4d}  rejected={len(provider.rejected):4d}"
        )

    meta = {
        "git_revision": _git_rev(),
        "config_path": str(args.config),
        "config_hash": _config_hash(config),
        "config": config,
        "noise_version": noise_version_for_config(config),
        "seed": seed,
        "rejections": rejections,
        "n_instances": len(all_instances),
        # Pins the generated set itself, not just the config that asked for it:
        # a released benchmark has to be checkable against the instances the
        # published numbers were computed on.
        "instance_checksum": manifest_checksum(all_instances),
        "generator_version": GENERATOR_VERSION,
    }
    out_path = Path(args.out)
    save_instances(out_path, all_instances, meta=meta)
    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    save_manifest(manifest_path, all_instances, extra={"generated_by": meta})
    print(f"\nSaved {len(all_instances)} instances -> {out_path}")
    print(f"Saved manifest ({meta['instance_checksum'][:12]}…) -> {manifest_path}")


# ── ablate ───────────────────────────────────────────────────────────────────


def cmd_ablate(args: argparse.Namespace) -> None:
    _warn_if_untested_opencv()
    instances, gen_meta = load_instances(Path(args.instances))
    print(
        f"Loaded {len(instances)} instances (config_hash={gen_meta.get('config_hash')})"
    )
    # Verify before scoring: an instance file that no longer matches the
    # checksum recorded at generation is not the set any published rate
    # belongs to.  Older files predate the checksum and are only reported on.
    stored_checksum = gen_meta.get("instance_checksum")
    if stored_checksum is None:
        print("  [warn] instance file predates checksums — cannot verify the set")
    else:
        actual = manifest_checksum(instances)
        if actual != stored_checksum:
            raise SystemExit(
                f"instance checksum mismatch: file records {stored_checksum}, "
                f"loaded instances hash to {actual}"
            )
        print(f"  ✓ instance checksum verified ({actual[:12]}…)")

    # The offset-correction sweep needs each camera's fitted mapper; instances
    # don't carry it (it isn't instance data), so regenerate mappers from the
    # same seed/config the instances were generated with — deterministic, so
    # this reproduces the exact mappers used at generation time.
    import numpy as np

    from pccv_bench.cameras import fit_synthetic_camera

    cfg = gen_meta.get("config", {})
    seed = gen_meta.get("seed", 42)
    cam_rng = np.random.default_rng(seed)
    cameras = cfg.get("cameras") or sorted({i.camera for i in instances})
    mappers = {c: fit_synthetic_camera(cam_rng, c) for c in cameras}

    twoway_threshold = cfg.get("twoway_threshold", 0.3)

    # E7 (PREREGISTRATION.md §1): score every arm -- baselines included -- at
    # the real study's frozen operating point instead of its own native one.
    # Two independent ways to turn this on: the CLI flag (forces it on
    # regardless of the config) or the generating config's own
    # `align_to_frozen_config: true` key (so an instance file generated with
    # that key set replays aligned by default, without the flag). Neither
    # changes anything for a config/CLI invocation that doesn't ask for it --
    # existing (pre-E7) instance files and result files reproduce unchanged.
    align = args.frozen_operating_point or bool(cfg.get("align_to_frozen_config", False))
    frozen = load_frozen_config() if align else None
    if frozen is not None:
        print(
            f"Aligned to the real study's frozen config (E7): metric={frozen['metric']!r}, "
            f"threshold={frozen['threshold']}, twoway_threshold={frozen['twoway_threshold']} "
            f"(overrides this config's own twoway_threshold={twoway_threshold})"
        )
    else:
        print(f"Running ablation ladder (twoway_threshold={twoway_threshold})…")
    verdicts_by_arm = run_all_arms(
        instances, mappers, twoway_threshold=twoway_threshold, frozen=frozen
    )

    summaries = []
    for arm_name, verdicts in verdicts_by_arm.items():
        s = summarise_arm(arm_name, verdicts)
        summaries.append(s)
        d = s["headline"]["detection_rate_pooled"]["rate"]
        f = s["headline"]["false_error_rate_clean_only"]["rate"]
        print(
            f"  {arm_name:22s} detect={'—' if d is None else format(d,'.3f')}  "
            f"false_err(clean)={'—' if f is None else format(f,'.3f')}"
        )

    payload = {
        "git_revision": _git_rev(),
        "generation_meta": gen_meta,
        "n_instances": len(instances),
        "frozen_operating_point": frozen,
        "summaries": summaries,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"\nSaved -> {out_path}")


# ── report ───────────────────────────────────────────────────────────────────


def cmd_report(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    summaries = payload["summaries"]
    print(f"git revision:   {payload.get('git_revision')}")
    gm = payload.get("generation_meta", {})
    print(f"config:         {gm.get('config_path')} (hash {gm.get('config_hash')})")
    print(f"noise version:  {gm.get('noise_version')}")
    print(f"instances:      {payload.get('n_instances')}")
    print(format_summary_table(summaries))
    for s in summaries:
        if s["arm"] in ("flag_none", "flag_all"):
            continue
        print(format_per_class_table(s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate labelled synthetic instances")
    g.add_argument("--config", required=True)
    g.add_argument("--seed", type=int, default=None, help="override the config's seed")
    g.add_argument("--out", required=True, help="output instances JSON path")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser(
        "ablate", help="run the ablation ladder over generated instances"
    )
    a.add_argument("--instances", required=True)
    a.add_argument("--out", required=True, help="output results JSON path")
    a.add_argument(
        "--frozen-operating-point", action="store_true",
        help=(
            "score every arm (baselines included) at the real study's frozen "
            f"operating point ({FROZEN_CONFIG_PATH.relative_to(_ROOT)}) instead of "
            "its own native metric/threshold -- PCCV-Bench agreement study E7. "
            "A generating config's own `align_to_frozen_config: true` key turns "
            "this on too, without the flag; this flag always wins if passed."
        ),
    )
    a.set_defaults(func=cmd_ablate)

    r = sub.add_parser("report", help="print tables from an ablate results file")
    r.add_argument("--results", required=True)
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
