#!/bin/bash
#SBATCH --job-name=anymc3d-ct-8g-smoke
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Multi-node NCCL initialization and first-time kernel compilation are slow on
# this system even with node-local caches.
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
# The cluster OFI plugin is built against NCCL 2.26.6 while this PyTorch
# wheel bundles NCCL 2.29.3.  Use the bundled socket transport on HSN to avoid
# a plugin-ABI collective hang at the first DDP backward pass.
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn0
# PyTorch 2.13's eager native-DSL override JIT-compiles bmm_outer_product
# independently on every node.  On this cluster that compile can stall one
# rank long enough to block the whole DDP group; use the stable ATen fallback.
export TORCH_DISABLE_NATIVE_JIT=1

srun --label scripts/run_with_local_caches.sh "$PYTHON" train.py \
  data=ct_rate_full \
  model=anymc3d_ct_rate_vitb \
  model.devices=1 \
  model.num_nodes=8 \
  model.max_epochs=1 \
  model.save_top_k=1 \
  model.early_stopping_patience=1 \
  model.log_every_n_steps=1 \
  model.run_name=anymc3d_ct_rate_8gpu_smoke \
  data.module.num_slices=8 \
  data.module.num_workers=0 \
  data.module.train_limit=8 \
  data.module.val_limit=8
