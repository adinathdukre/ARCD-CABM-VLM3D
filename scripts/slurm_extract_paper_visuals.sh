#!/bin/bash
#SBATCH --job-name=ct-paper-vis
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load cuda/12.6

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
cd "$PROJECT_DIR"
mkdir -p logs

export TORCH_HOME=${HOME}/.cache/torch
export HF_HOME=${HOME}/.cache/huggingface
export DINOV2_REPO=${HOME}/.cache/torch/hub/facebookresearch_dinov2_main
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export ANYMC3D_SLICE_CHUNK=16
export OMP_NUM_THREADS=8

PER_CLASS=${PER_CLASS:-20}
GATE_PAIRS=${GATE_PAIRS:-24}
OUTPUT_PATH=${OUTPUT_PATH:-${SCRATCH_ROOT}/paper_manuscript_stage/data/model_visuals.npz}

srun --gpu-bind=single:1 scripts/run_with_local_caches.sh \
  "$PYTHON" extract_paper_visuals.py \
  --per-class "$PER_CLASS" \
  --gate-pairs "$GATE_PAIRS" \
  --output "$OUTPUT_PATH"
