from __future__ import annotations

import os
import random

import numpy as np
import torch


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def describe_runtime(device: torch.device) -> str:
    return (
        f"torch={torch.__version__} | device={device} | "
        f"machine={os.uname().machine} | mps={torch.backends.mps.is_available()} | "
        f"cuda={torch.cuda.is_available()}"
    )
