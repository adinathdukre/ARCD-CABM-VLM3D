#!/bin/bash
#SBATCH --job-name=ct-arcd-submit
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT=${REPO_ROOT}
SUBMISSION="$PROJECT/submission/ct_abnormality"
PYTHON=python
FORITHMUS=${HOME}/.forithmus-cli-py311/bin/forithmus
PREDICTIONS="$PROJECT/calibration/predictions/arcd_v1.csv"
IMAGE=${SCRATCH_ROOT}/Med-SenseNova-Vision/submission_artifacts/vlm3d-ct-abnormality-arcd-amd64.tar.gz
WEIGHTS="$SUBMISSION/artifacts/anymc3d-arcd-weights.zip"
MARKER="$SUBMISSION/artifacts/arcd_scored_submission_completed.txt"
SEEDS=(2026 2027 2028 3026 4026)

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ANYMC3D_INFERENCE_ONLY=1

cd "$PROJECT"
mkdir -p logs

if [[ -f "$MARKER" ]]; then
  echo "Refusing a duplicate scored submission; marker exists: $MARKER" >&2
  exit 1
fi
if [[ ! -s "$PREDICTIONS" ]]; then
  echo "Missing ARCD predictions: $PREDICTIONS" >&2
  exit 1
fi
sha256sum -c "$IMAGE.sha256"

SUMMARY_ARGS=()
PARAMETER_ARGS=()
for seed in "${SEEDS[@]}"; do
  output_dir="$PROJECT/calibration/arcd_patient_safe_seed${seed}"
  "$PYTHON" ct_rate_calibrate_ensemble.py calibrate \
    --predictions "$PREDICTIONS" \
    --names arcd \
    --output-dir "$output_dir" \
    --calibration-fraction 0.5 \
    --seed "$seed" \
    --weight-resolution 10 \
    --crg-fraction 0.45
  SUMMARY_ARGS+=("$output_dir/summary.csv")
  PARAMETER_ARGS+=(
    --parameters "$output_dir/single_model_thresholds.csv" --seed "$seed"
  )
done

"$PYTHON" "$SUBMISSION/check_arcd_calibration.py" "${SUMMARY_ARGS[@]}" \
  --model arcd --minimum-auroc 0.86 --minimum-f1 0.59 --minimum-crg 0.54

"$PYTHON" "$SUBMISSION/finalize_arcd_manifest.py" \
  --manifest "$SUBMISSION/manifest_arcd.json" \
  --model arcd \
  "${PARAMETER_ARGS[@]}"

cd "$SUBMISSION"
./package_arcd_weights.sh
FORITHMUS_TEST_MANIFEST=manifest_arcd.json "$PYTHON" test_contract.py
sha256sum -c "$WEIGHTS.sha256"
"$FORITHMUS" whoami
"$FORITHMUS" phases ct-abnormality-classification

"$FORITHMUS" submit "$IMAGE" \
  --phase main-2026 \
  --tier gpu-a100-80 \
  --time-budget 90 \
  --weights "$WEIGHTS" \
  --desc "AnyMC3D ARCD with acquisition-routed consensus distillation and robust patient-grouped calibration"

date -u '+submitted_at=%Y-%m-%dT%H:%M:%SZ' > "$MARKER"
echo "Scored ARCD submission created; marker: $MARKER"
