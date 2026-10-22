#!/bin/bash
#SBATCH --job-name=nccl-2n8g-smoke
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load cuda/12.6

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
cd "$PROJECT_DIR"
mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29686
export NCCL_DEBUG=INFO
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn0

srun --label "$PYTHON" scripts/ddp_allreduce_smoke.py
