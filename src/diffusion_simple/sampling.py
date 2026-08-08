import time
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from .dit import DiT
from .forward import extract
from .schedules import NoiseSchedule


@dataclass
class SampleResult:
    images: torch.Tensor
    trajectory: list[torch.Tensor] = field(default_factory=list)
    timesteps: list[int] = field(default_factory=list)
    seconds: float = 0.0
    num_model_calls: int = 0


@torch.no_grad()
def predict_x0_and_eps(
    model: DiT,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    clip_denoised: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(x_t, t)
    sqrt_ab = extract(schedule.sqrt_alpha_bars, t, x_t.shape)
    sqrt_1mab = extract(schedule.sqrt_one_minus_alpha_bars, t, x_t.shape)

    if model.parameterization == "eps":
        x0 = (x_t - sqrt_1mab * output) / sqrt_ab
    else:
        x0 = output

    if clip_denoised:
        x0 = x0.clamp(-1.0, 1.0)
    eps = (x_t - sqrt_ab * x0) / sqrt_1mab
    return x0, eps


@torch.no_grad()
def ddpm_sample(
    model: DiT,
    schedule: NoiseSchedule,
    num_images: int = 16,
    *,
    record_every: int | None = None,
    clip_denoised: bool = True,
    generator: torch.Generator | None = None,
    progress: bool = False,
) -> SampleResult:
    device = next(model.parameters()).device
    cfg = model.config
    shape = (num_images, cfg.in_channels, cfg.image_size, cfg.image_size)

    x = torch.randn(shape, device=device, generator=generator)
    result = SampleResult(images=x)
    start = time.perf_counter()

    for step in tqdm(range(len(schedule) - 1, -1, -1), disable=not progress, leave=False):
        if record_every and step % record_every == 0:
            result.trajectory.append(x.clone().cpu())
            result.timesteps.append(step)

        t = torch.full((num_images,), step, device=device, dtype=torch.long)
        _, eps = predict_x0_and_eps(model, x, t, schedule, clip_denoised)
        result.num_model_calls += 1

        beta_t = schedule.betas[step]
        alpha_t = schedule.alphas[step]
        alpha_bar_t = schedule.alpha_bars[step]
        alpha_bar_prev = schedule.alpha_bars_prev[step]

        # x_{t-1} = 1/sqrt(a_t) * (x_t - b_t/sqrt(1-ab_t) * eps) + sigma_t * z
        #   b_t/sqrt(1-ab_t) * eps  this step's share of the total noise in x_t
        noise_to_remove = (beta_t / (1.0 - alpha_bar_t).sqrt()) * eps
        #   1/sqrt(a_t)             undo the variance-preserving shrink
        mean = (x - noise_to_remove) / alpha_t.sqrt()
        #   sigma_t^2               posterior variance; exactly 0 at t=0
        variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        #   + sigma_t * z           sample, don't average, or every run collapses
        x = mean + variance.sqrt() * torch.randn(shape, device=device, generator=generator)

    result.images = x
    result.seconds = time.perf_counter() - start
    if record_every:
        result.trajectory.append(x.clone().cpu())
        result.timesteps.append(-1)
    return result


def ddim_timesteps(num_timesteps: int, steps: int) -> list[int]:
    if steps > num_timesteps:
        raise ValueError(f"steps {steps} exceeds num_timesteps {num_timesteps}")
    spaced = torch.linspace(0, num_timesteps - 1, steps).round().long()
    return sorted(set(spaced.tolist()), reverse=True)


@torch.no_grad()
def ddim_sample(
    model: DiT,
    schedule: NoiseSchedule,
    num_images: int = 16,
    *,
    steps: int = 50,
    eta: float = 0.0,
    record_every: int | None = None,
    clip_denoised: bool = True,
    generator: torch.Generator | None = None,
    progress: bool = False,
) -> SampleResult:
    device = next(model.parameters()).device
    cfg = model.config
    shape = (num_images, cfg.in_channels, cfg.image_size, cfg.image_size)

    x = torch.randn(shape, device=device, generator=generator)
    times = ddim_timesteps(len(schedule), steps)
    result = SampleResult(images=x)
    start = time.perf_counter()

    for i, step in enumerate(tqdm(times, disable=not progress, leave=False)):
        if record_every and i % record_every == 0:
            result.trajectory.append(x.clone().cpu())
            result.timesteps.append(step)

        t = torch.full((num_images,), step, device=device, dtype=torch.long)
        x0, eps = predict_x0_and_eps(model, x, t, schedule, clip_denoised)
        result.num_model_calls += 1

        alpha_bar_t = schedule.alpha_bars[step]
        prev_step = times[i + 1] if i + 1 < len(times) else -1
        alpha_bar_prev = (
            schedule.alpha_bars[prev_step]
            if prev_step >= 0
            else torch.ones((), device=device)
        )

        # x_prev = sqrt(ab_prev) * x0 + sqrt(1 - ab_prev - sigma^2) * eps + sigma * z
        #   sigma            eta=0 -> deterministic; eta=1 -> the DDPM posterior std
        sigma = (
            eta
            * ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)).sqrt()
            * (1.0 - alpha_bar_t / alpha_bar_prev).sqrt()
        )
        #   sqrt(ab_prev)*x0 re-noise the predicted clean image to the prev level
        predicted_signal = alpha_bar_prev.sqrt() * x0
        #   sqrt(...)*eps    reuse the *same* eps, which is what makes steps skippable
        direction = (1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0).sqrt() * eps

        x = predicted_signal + direction
        if eta > 0 and prev_step >= 0:
            x = x + sigma * torch.randn(shape, device=device, generator=generator)

    result.images = x
    result.seconds = time.perf_counter() - start
    if record_every:
        result.trajectory.append(x.clone().cpu())
        result.timesteps.append(-1)
    return result
