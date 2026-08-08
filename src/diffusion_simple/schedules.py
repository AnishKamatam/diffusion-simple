import math
from dataclasses import dataclass, fields, replace
from typing import Literal

import torch

ScheduleKind = Literal["linear", "cosine"]


@dataclass(frozen=True)
class NoiseSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor

    @classmethod
    def from_betas(cls, betas: torch.Tensor) -> "NoiseSchedule":
        betas = betas.double()
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([alphas.new_ones(1), alpha_bars[:-1]])
        return cls(
            betas=betas.float(),
            alphas=alphas.float(),
            alpha_bars=alpha_bars.float(),
            alpha_bars_prev=alpha_bars_prev.float(),
            sqrt_alpha_bars=alpha_bars.sqrt().float(),
            sqrt_one_minus_alpha_bars=(1.0 - alpha_bars).sqrt().float(),
        )

    def __len__(self) -> int:
        return len(self.betas)

    def to(self, device: torch.device | str) -> "NoiseSchedule":
        moved = {f.name: getattr(self, f.name).to(device) for f in fields(self)}
        return replace(self, **moved)


def linear_betas(
    num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def cosine_betas(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
    f = torch.cos((t / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    return betas.clamp(max=0.999)


def make_schedule(kind: ScheduleKind, num_timesteps: int = 1000) -> NoiseSchedule:
    if kind == "linear":
        betas = linear_betas(num_timesteps)
    elif kind == "cosine":
        betas = cosine_betas(num_timesteps)
    else:
        raise ValueError(f"unknown schedule kind {kind!r}; expected 'linear' or 'cosine'")
    return NoiseSchedule.from_betas(betas)
