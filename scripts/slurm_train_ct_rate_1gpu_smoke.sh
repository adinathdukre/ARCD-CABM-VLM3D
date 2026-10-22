#!/bin/bash
#SBATCH --job-name=anymc3d-ct-1g-smoke
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load cuda/12.6

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

export WANDB_MODE=offline
export WANDB_DIR="$PROJECT_DIR/wandb"
export TORCH_HOME=${HOME}/.cache/torch
export HF_HOME=${HOME}/.cache/huggingface
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

# This is a real Lightning fit/validation cycle on two train and two validation
# CT-RATE scans. Eight sampled slices keep the preflight short; the full job
# retains its production setting of 96 slices.
srun --label "$PYTHON" train.py \
  data=ct_rate_full \
  model=anymc3d_ct_rate_vitb \
  model.devices=1 \
  model.num_nodes=1 \
  model.strategy=auto \
  model.max_epochs=1 \
  model.save_top_k=1 \
  model.early_stopping_patience=1 \
  model.log_every_n_steps=1 \
  model.run_name=anymc3d_ct_rate_1gpu_smoke \
  data.module.num_slices=8 \
  data.module.num_workers=0 \
  data.module.train_limit=2 \
  data.module.val_limit=2
