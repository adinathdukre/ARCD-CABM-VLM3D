#!/bin/bash
#SBATCH --job-name=anymc3d-ct-rate
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
# workq_qos caps jobs at 24 hours.
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

export WANDB_MODE=offline
export WANDB_DIR="$PROJECT_DIR/wandb"
export TORCH_HOME=${HOME}/.cache/torch
export HF_HOME=${HOME}/.cache/huggingface
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

srun --label "$PYTHON" train.py \
  data=ct_rate_full \
  model=anymc3d_ct_rate_vitb
