"""Generate + ablate medium v1.2 across many seeds, aligned — shared driver
for E2 (seed variance), E3 (evaluation seeds), and E4 (stability check).

Shells out to the tested CLI path (``python -m pccv_bench.cli generate`` /
``ablate --frozen-operating-point``) rather than reimplementing it, so every
seed's instances/results carry the same provenance stamping (git revision,
config hash, instance checksum) a standalone CLI run would. Idempotent: a
seed already present under ``--out-dir`` (both instances and results files
exist) is skipped, so an interrupted sweep resumes cheaply.

Usage::

    python -m pccv_bench.run_seed_sweep \\
        --config pccv_bench/configs/medium_v1_2_fitted.yaml \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --out-dir /tmp/pccv_seed_sweep
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def run_one(config: str, seed: int, out_dir: Path, align: bool) -> tuple[Path, Path]:
    stem = Path(config).stem
    instances_path = out_dir / f"instances_{stem}_seed{seed}.json"
    results_path = out_dir / f"results_{stem}_seed{seed}{'_aligned' if align else ''}.json"
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
    if not results_path.is_file():
        cmd = [
            sys.executable, "-m", "pccv_bench.cli", "ablate",
            "--instances", str(instances_path), "--out", str(results_path),
        ]
        if align:
            cmd.append("--frozen-operating-point")
        subprocess.run(cmd, cwd=str(_ROOT), check=True)
    else:
        print(f"  [skip ablate] {results_path} exists")
    return instances_path, results_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-align", action="store_true", help="skip --frozen-operating-point (default: aligned)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"seed {seed}:")
        run_one(args.config, seed, out_dir, align=not args.no_align)
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
