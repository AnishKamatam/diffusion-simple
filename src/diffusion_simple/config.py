from dataclasses import dataclass, field
from pathlib import Path

import torch

from .data import REPO_ROOT
from .schedules import ScheduleKind

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures"

Parameterization = str  # "eps" | "x0"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class DiTConfig:
    image_size: int = 28
    in_channels: int = 1
    patch_size: int = 4
    dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0

    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} not divisible by patch_size {self.patch_size}"
            )
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size**2


@dataclass(frozen=True)
class TrainConfig:
    name: str = "eps_linear"
    parameterization: Parameterization = "eps"
    schedule: ScheduleKind = "linear"
    num_timesteps: int = 1000
    epochs: int = 25
    batch_size: int = 128
    lr: float = 3e-4
    seed: int = 0
    log_every: int = 100
    model: DiTConfig = field(default_factory=DiTConfig)

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / f"{self.name}.pt"
