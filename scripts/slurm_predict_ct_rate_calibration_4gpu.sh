#!/bin/bash
#SBATCH --job-name=ct-cal-predict
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: sbatch $0 NAME CONFIG CHECKPOINT OUTPUT.csv" >&2
  exit 2
fi

NAME=$1
CONFIG=$2
CHECKPOINT=$3
OUTPUT=$4

module load cuda/12.6

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
cd "$PROJECT_DIR"
mkdir -p logs "$(dirname "$OUTPUT")"

export TORCH_HOME=${HOME}/.cache/torch
export HF_HOME=${HOME}/.cache/huggingface
export DINOV2_REPO=${HOME}/.cache/torch/hub/facebookresearch_dinov2_main
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

echo "Prediction model: $NAME"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"

srun --label --gpu-bind=single:1 scripts/run_with_local_caches.sh \
  "$PYTHON" ct_rate_calibrate_ensemble.py predict \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch-size 1 \
  --num-workers 4

OUTPUT_STEM=${OUTPUT%.csv}
TASK_COUNT=${SLURM_NTASKS:-4}
printf -v TASK_COUNT_PADDED '%05d' "$TASK_COUNT"
if [[ $TASK_COUNT -eq 1 ]]; then
  PARTS=("$OUTPUT")
else
  PARTS=("${OUTPUT_STEM}".part-*-of-"${TASK_COUNT_PADDED}".csv)
fi
if [[ ${#PARTS[@]} -ne $TASK_COUNT ]]; then
  echo "Expected $TASK_COUNT prediction shards, found ${#PARTS[@]}" >&2
  exit 1
fi

if [[ $TASK_COUNT -gt 1 ]]; then
  "$PYTHON" ct_rate_calibrate_ensemble.py merge \
    --parts "${PARTS[@]}" \
    --output "$OUTPUT"
fi

echo "Completed prediction export for $NAME"
