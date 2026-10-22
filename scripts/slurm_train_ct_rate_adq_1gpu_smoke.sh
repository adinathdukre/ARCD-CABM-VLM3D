#!/bin/bash
#SBATCH --job-name=ct-adq-1g-smoke
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
# The successful reference smoke peaked at 2.4 GiB RSS.  Set this explicitly
# per GPU so the partition's 115 GB/GPU default does not unnecessarily exclude
# partially occupied nodes.
#SBATCH --mem-per-gpu=16G
# The complete four-volume train/validation smoke normally finishes in under
# two minutes; a short request is much easier for Slurm to backfill.
#SBATCH --time=00:10:00
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

# End-to-end ADQ-MR preflight: paired reconstructions, independent MONAI
# augmentation, 18 query heads, consistency loss, backward, and validation.
srun --label "$PYTHON" train.py \
  data=ct_rate_adq \
  model=anymc3d_ct_rate_adq \
  model.devices=1 \
  model.num_nodes=1 \
  model.strategy=auto \
  model.max_epochs=1 \
  model.save_top_k=1 \
  model.early_stopping_patience=1 \
  model.log_every_n_steps=1 \
  model.run_name=ct_rate_adq_1gpu_smoke \
  data.module.num_slices=8 \
  data.module.num_workers=0 \
  data.module.train_limit=4 \
  data.module.val_limit=4
