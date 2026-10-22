#!/bin/bash
#SBATCH --job-name=anymc3d-smoke
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
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
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

srun --label "$PYTHON" scripts/smoke_test_ct_rate.py
