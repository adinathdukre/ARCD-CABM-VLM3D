#!/bin/bash
#SBATCH --job-name=tapct_head
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load cuda/12.6

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
OUTPUT_DIR=${SCRATCH_ROOT}/Med-SenseNova-Vision/artifacts/tapct-classification
cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

srun --label "$PYTHON" tapct_ct_rate.py train \
  --output-dir "$OUTPUT_DIR" \
  --seed 2026 \
  --epochs 60 \
  --patience 10 \
  --batch-size 256
