#!/bin/bash
set -euo pipefail

# Triton writes a compiled-kernel metadata file and then reads it back. Eight
# ranks using the same cache on the shared filesystem can observe a stale NFS
# handle. Give every Slurm rank independent node-local compiler caches.
CACHE_ROOT="/tmp/anymc3d-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

exec "$@"
