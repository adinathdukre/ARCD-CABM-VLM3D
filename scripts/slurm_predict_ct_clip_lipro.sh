#!/bin/bash
#SBATCH --job-name=ct-lipro-predict
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

LIMIT=${1:-}
OUTPUT=${2:-calibration/ct_clip_lipro_v2.csv}
PROJECT_DIR=${REPO_ROOT}
PYTHON=python
DATASET=${CT_RATE_ROOT}
CTCLIP=${HOME}/3dvlm/CT-CLIP
CHECKPOINT=${CT_RATE_PARENT}/models/CT-CLIP-Related/CT_LiPro_v2.pt

module load cuda/12.6
cd "$PROJECT_DIR"
mkdir -p logs "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=12

ARGS=(
  predict
  --data-root "$DATASET/valid_fixed"
  --metadata "$DATASET/metadata/validation_metadata.csv"
  --labels "$DATASET/multi_abnormality_labels/valid_predicted_labels.csv"
  --checkpoint "$CHECKPOINT"
  --ctvit-source "$CTCLIP/transformer_maskgit"
  --output "$OUTPUT"
  --intensity-mode hu
  --amp none
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

srun --label --gpu-bind=single:1 scripts/run_with_local_caches.sh \
  "$PYTHON" ct_clip_lipro_predict.py "${ARGS[@]}"

OUTPUT_STEM=${OUTPUT%.csv}
TASK_COUNT=${SLURM_NTASKS:-4}
printf -v TASK_COUNT_PADDED '%05d' "$TASK_COUNT"
if [[ $TASK_COUNT -eq 1 ]]; then
  if [[ ! -s "$OUTPUT" ]]; then
    echo "Expected non-empty prediction output: $OUTPUT" >&2
    exit 1
  fi
  echo "Completed CT-CLIP LiPro export: $OUTPUT"
  exit 0
fi
PARTS=("${OUTPUT_STEM}".part-*-of-"${TASK_COUNT_PADDED}".csv)
if [[ ${#PARTS[@]} -ne $TASK_COUNT ]]; then
  echo "Expected $TASK_COUNT prediction shards, found ${#PARTS[@]}" >&2
  exit 1
fi
"$PYTHON" ct_clip_lipro_predict.py merge --parts "${PARTS[@]}" --output "$OUTPUT"

echo "Completed CT-CLIP LiPro export: $OUTPUT"
