import os
import torch
import torch.distributed as dist
from loguru import logger


def init_distributed_torch():
    """Initializes one PyTorch distributed process per MESH host."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = os.environ.get(
        "TORCH_DISTRIBUTED_BACKEND",
        "nccl" if torch.cuda.is_available() else "gloo",
    )

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    if rank == 0:
        logger.info(
            "PyTorch distributed initialized with %d processes (backend=%s, device=%s).",
            world_size,
            backend,
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        )


init_distributed_torch()
