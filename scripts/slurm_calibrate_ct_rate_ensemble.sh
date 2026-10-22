#!/bin/bash
#SBATCH --job-name=ct-cal-fit
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
cd "$PROJECT_DIR"
mkdir -p logs calibration/adq_cdra_crs_patient_safe

"$PYTHON" ct_rate_calibrate_ensemble.py calibrate \
  --predictions \
    calibration/predictions/adq_mr.csv \
    calibration/predictions/adq_cdra.csv \
    calibration/predictions/adq_crs_asl.csv \
  --names adq_mr adq_cdra adq_crs_asl \
  --output-dir calibration/adq_cdra_crs_patient_safe \
  --calibration-fraction 0.5 \
  --seed 2026 \
  --weight-resolution 10 \
  --crg-fraction 0.45
