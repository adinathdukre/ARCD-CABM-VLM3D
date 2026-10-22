"""Minimal multi-node CUDA/NCCL collective check launched by Slurm."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NTASKS"])
local_rank = int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(local_rank)

dist.init_process_group(
    backend="nccl",
    rank=rank,
    world_size=world_size,
    timeout=timedelta(minutes=2),
)

value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
dist.all_reduce(value)
expected = world_size * (world_size + 1) / 2
if value.item() != expected:
    raise RuntimeError(f"rank {rank}: all_reduce={value.item()}, expected={expected}")

print(f"rank {rank}: all_reduce OK ({value.item():.0f})", flush=True)
dist.barrier()
dist.destroy_process_group()
