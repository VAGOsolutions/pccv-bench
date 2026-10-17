#!/usr/bin/env bash
# Regenerates the seed-42 reference results of all three tiers (about 15 minutes
# per tier on one core; the tiers run in parallel). Run from the repository root.
#
# Outputs are written to a scratch directory and copied into results/ and
# instances/ at the end, so the working tree stays clean while the runs record
# their git revision.
set -euo pipefail

PYTHON="${PYTHON:-python}"
WORK="$(mktemp -d)"

run_tier () {
  local tier="$1"
  "$PYTHON" -m pccv_bench.cli generate --config "pccv_bench/configs/${tier}.yaml" --seed 42 --out "${WORK}/${tier}.json"
  "$PYTHON" -m pccv_bench.cli ablate --instances "${WORK}/${tier}.json" --out "${WORK}/results_${tier}_aligned.json" --frozen-operating-point
  "$PYTHON" -m pccv_bench.cli report --results "${WORK}/results_${tier}_aligned.json" > "${WORK}/report_${tier}_aligned.txt"
}

for tier in easy_v1_2 medium_v1_2_fitted hard_v1_2; do
  run_tier "$tier" &
done
wait

mkdir -p results instances
cp "${WORK}"/results_*_aligned.json "${WORK}"/report_*_aligned.txt results/
# The pinned instance set is the medium tier at seed 42.
cp "${WORK}/medium_v1_2_fitted.json" instances/medium_v1_2_fitted_seed42.json
cp "${WORK}/medium_v1_2_fitted_manifest.json" instances/medium_v1_2_fitted_seed42_manifest.json
echo "done; scratch directory: ${WORK}"
