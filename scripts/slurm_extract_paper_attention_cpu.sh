#!/bin/bash
#SBATCH --job-name=ct-vis-cpu
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
cd "$PROJECT_DIR"
mkdir -p logs

export TORCH_HOME=${HOME}/.cache/torch
export HF_HOME=${HOME}/.cache/huggingface
export DINOV2_REPO=${HOME}/.cache/torch/hub/facebookresearch_dinov2_main
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export ANYMC3D_SLICE_CHUNK=16
export OMP_NUM_THREADS=64
export NUMEXPR_MAX_THREADS=64

srun scripts/run_with_local_caches.sh \
  "$PYTHON" extract_paper_visuals.py \
  --device cpu \
  --models CDRA "Practical ACER" \
  --per-class 1 \
  --gate-pairs 4 \
  --output ${SCRATCH_ROOT}/paper_manuscript_stage/data/model_visuals_attention.npz
