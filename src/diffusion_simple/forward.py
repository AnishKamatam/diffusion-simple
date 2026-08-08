import torch

from .schedules import NoiseSchedule


def extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.to(t.device).gather(0, t)
    return out.view(t.shape[0], *((1,) * (len(shape) - 1)))


def sample_timesteps(batch_size: int, num_timesteps: int, device) -> torch.Tensor:
    return torch.randint(0, num_timesteps, (batch_size,), device=device, dtype=torch.long)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x0)
    signal = extract(schedule.sqrt_alpha_bars, t, x0.shape)
    scale = extract(schedule.sqrt_one_minus_alpha_bars, t, x0.shape)
    return signal * x0 + scale * noise, noise
