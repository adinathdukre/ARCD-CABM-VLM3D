#!/bin/bash
#SBATCH --job-name=ct-adq-8g-smoke
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000

srun --label scripts/run_with_local_caches.sh "$PYTHON" train.py \
  data=ct_rate_adq \
  model=anymc3d_ct_rate_adq \
  model.devices=1 \
  model.num_nodes=8 \
  model.max_epochs=1 \
  model.save_top_k=1 \
  model.early_stopping_patience=1 \
  model.log_every_n_steps=1 \
  model.run_name=ct_rate_adq_8gpu_smoke_v3 \
  data.module.num_slices=8 \
  data.module.num_workers=0 \
  data.module.train_limit=32 \
  data.module.val_limit=16
