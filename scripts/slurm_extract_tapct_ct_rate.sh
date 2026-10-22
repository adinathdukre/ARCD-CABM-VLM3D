#!/bin/bash
#SBATCH --job-name=tapct_extract
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
module load cuda/12.6

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
OUTPUT_DIR=${SCRATCH_ROOT}/Med-SenseNova-Vision/artifacts/tapct-classification
NUM_TRAIN_SHARDS=16
NUM_VALID_SHARDS=16
cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR/features"
export HF_HOME=${HOME}/.cache/huggingface
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

srun --label "$PYTHON" tapct_ct_rate.py extract \
  --split train \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_TRAIN_SHARDS" \
  --output-dir "$OUTPUT_DIR" \
  --chunk-batch-size 1

srun --label "$PYTHON" tapct_ct_rate.py extract \
  --split valid \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_VALID_SHARDS" \
  --output-dir "$OUTPUT_DIR" \
  --chunk-batch-size 1
