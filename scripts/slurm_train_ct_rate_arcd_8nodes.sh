#!/bin/bash
#SBATCH --job-name=ct-arcd-8n
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load cuda/12.6

PROJECT_DIR=${REPO_ROOT}
PYTHON=python
CHECKPOINT_ROOT=/lus/lfs1aip2${SCRATCH_ROOT}/Med-SenseNova-Vision/artifacts/anymc3d-classification
RUN_NAME=ct_rate_arcd_dinov2_vitb_96s_v1
INIT_CHECKPOINT="$CHECKPOINT_ROOT/ct_rate_acer_crs_dinov2_vitb_96s_v1_fold0/last.ckpt"
RESUME_CHECKPOINT="$CHECKPOINT_ROOT/${RUN_NAME}_fold0/resume/last.ckpt"

cd "$PROJECT_DIR"
mkdir -p logs "$CHECKPOINT_ROOT"
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

training_pid=""
requeue_before_timeout() {
  echo "Received Slurm pre-timeout signal; requeueing job ${SLURM_JOB_ID}."
  if [[ -n "$training_pid" ]]; then
    kill -TERM "$training_pid" 2>/dev/null || true
    wait "$training_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_before_timeout USR1

train_args=(
  train.py
  data=ct_rate_acer
  model=anymc3d_ct_rate_arcd
  model.devices=1
  model.num_nodes=8
  model.max_epochs=2
  model.run_name="$RUN_NAME"
  data.module.num_workers=0
  checkpoint_root="$CHECKPOINT_ROOT"
  checkpoint_every_n_train_steps=500
)

if [[ -s "$RESUME_CHECKPOINT" ]]; then
  echo "Resuming ARCD from: $RESUME_CHECKPOINT"
  train_args+=(resume_from_checkpoint="$RESUME_CHECKPOINT")
else
  echo "Warm-starting ARCD from: $INIT_CHECKPOINT"
  train_args+=(init_from_checkpoint="$INIT_CHECKPOINT")
fi

srun --label scripts/run_with_local_caches.sh "$PYTHON" "${train_args[@]}" &
training_pid=$!
wait "$training_pid"
