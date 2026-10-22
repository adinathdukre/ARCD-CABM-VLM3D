#!/bin/bash
#SBATCH --job-name=ct-cff-calibrate
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
cd "$PROJECT_DIR"
mkdir -p logs calibration/cross_foundation

PREDICTIONS=(
  calibration/predictions/adq_mr.csv
  calibration/predictions/adq_cdra.csv
  calibration/adq_crs_asl_stage2.csv
  calibration/predictions/ct_clip_lipro_v2.csv
)
NAMES=(adq_mr adq_cdra adq_crs_asl ct_clip_lipro)

for path in "${PREDICTIONS[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing prediction file: $path" >&2
    exit 1
  fi
done

# Repeated patient-grouped splits expose fragile weights or thresholds. The
# same models are evaluated at each seed; no result is silently selected here.
for seed in 2026 2027 2028 2029 2030; do
  "$PYTHON" ct_rate_calibrate_ensemble.py calibrate \
    --predictions "${PREDICTIONS[@]}" \
    --names "${NAMES[@]}" \
    --output-dir "calibration/cross_foundation/seed_${seed}" \
    --calibration-fraction 0.5 \
    --seed "$seed" \
    --weight-resolution 10 \
    --crg-fraction 0.45
done

"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd

rows = []
for path in sorted(Path("calibration/cross_foundation").glob("seed_*/summary.csv")):
    frame = pd.read_csv(path)
    frame.insert(0, "seed", int(path.parent.name.split("_")[-1]))
    rows.append(frame)
all_results = pd.concat(rows, ignore_index=True)
all_results.to_csv("calibration/cross_foundation/repeated_summary.csv", index=False)
aggregate = (
    all_results.groupby(["model", "mode"])[
        ["crg", "macro_auroc", "macro_f1", "macro_accuracy"]
    ]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)
aggregate.to_csv("calibration/cross_foundation/repeated_aggregate.csv", index=False)
print(aggregate.to_string(index=False))
PY

echo "Completed repeated patient-safe cross-foundation calibration"
