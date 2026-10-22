#!/bin/bash
#SBATCH --job-name=ct-tam-smoke
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load cuda/12.6

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
INIT_CHECKPOINT=${SCRATCH_ROOT}/Med-SenseNova-Vision/artifacts/anymc3d-classification/ct_rate_arcd_dinov2_vitb_96s_v1_fold0/epoch=01-val_auroc=0.8628.ckpt
cd "$PROJECT_DIR"
mkdir -p logs

export WANDB_MODE=offline
export TORCH_HOME=${HOME}/.cache/torch
export HF_HOME=${HOME}/.cache/huggingface
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

srun --label "$PYTHON" train.py \
  data=ct_rate_acer \
  model=anymc3d_ct_rate_arcd_tam \
  init_from_checkpoint="$INIT_CHECKPOINT" \
  model.devices=1 \
  model.num_nodes=1 \
  model.strategy=auto \
  model.max_epochs=1 \
  model.tail_drw_start_fraction=0.0 \
  model.run_name=ct_rate_arcd_tam_smoke_v1 \
  model.log_every_n_steps=1 \
  data.module.num_slices=8 \
  data.module.num_workers=0 \
  data.module.train_limit=16 \
  data.module.val_limit=16
