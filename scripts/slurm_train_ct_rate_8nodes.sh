#!/bin/bash
#SBATCH --job-name=anymc3d-ct-rate-8n
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
# Match PyTorch's bundled NCCL instead of loading the cluster OFI plugin,
# which was compiled against an older NCCL ABI.
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn0
# Avoid per-node Triton compilation of PyTorch native-DSL eager kernels.  A
# single slow compiler rank otherwise blocks every process at the DDP step.
export TORCH_DISABLE_NATIVE_JIT=1

srun --label scripts/run_with_local_caches.sh "$PYTHON" train.py \
  data=ct_rate_full \
  model=anymc3d_ct_rate_vitb \
  model.devices=1 \
  model.num_nodes=8 \
  data.module.num_workers=4
