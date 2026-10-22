#!/bin/bash
#SBATCH --job-name=ct-adq-crsasl-8n
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
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
export NCCL_DEBUG=WARN
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn0
export TORCH_DISABLE_NATIVE_JIT=1

srun --label scripts/run_with_local_caches.sh "$PYTHON" train.py \
  data=ct_rate_adq \
  model=anymc3d_ct_rate_adq_crs_asl \
  init_from_checkpoint=${REPO_ROOT}/checkpoints/ct_rate_adq_mr_dinov2_vitb_96s_stage1_v2_fold0/last.ckpt \
  model.devices=1 \
  model.num_nodes=8 \
  model.max_epochs=1 \
  model.run_name=ct_rate_adq_crs_asl_dinov2_vitb_96s_stage2_v1 \
  data.module.num_workers=0
