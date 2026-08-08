"""Lesson 4 -- Run the chain backwards: noise in, digits out.

Run:  uv run experiments/04_reverse_sampling.py [--name eps_linear]
"""

import argparse

import matplotlib.pyplot as plt
import torch

from diffusion_simple.config import get_device
from diffusion_simple.sampling import ddpm_sample, predict_x0_and_eps
from diffusion_simple.train import load_model
from diffusion_simple.viz import save_figure, save_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="eps_linear")
    p.add_argument("--num-images", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model, schedule, _ = load_model(args.name, device)
    torch.manual_seed(0)

    print(f"sampling {args.num_images} images with {len(schedule)}-step DDPM...")
    result = ddpm_sample(model, schedule, args.num_images, record_every=50, progress=True)
    print(f"{result.seconds:.1f}s, {result.num_model_calls} model calls")

    save_grid(
        result.images,
        f"04_samples_{args.name}.png",
        ncols=8,
        title=f"{args.name}: {len(schedule)}-step DDPM samples from pure noise",
    )

    trajectory = torch.stack(result.trajectory)  # (snapshots, N, 1, 28, 28)
    shown = min(6, args.num_images)
    rows = [trajectory[:, i] for i in range(shown)]
    save_grid(
        torch.cat(rows),
        f"04_evolution_{args.name}.png",
        ncols=trajectory.shape[0],
        col_titles=[f"t={s}" if s >= 0 else "t=0" for s in result.timesteps],
        title="Denoising trajectory: every 50th step",
        cell=0.85,
    )

    print("\nre-running to capture x0 predictions...")
    torch.manual_seed(0)
    x = torch.randn(shown, 1, 28, 28, device=device)
    checkpoints = [999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 0]
    x0_rows: list[torch.Tensor] = []
    captured: dict[int, torch.Tensor] = {}

    for step in range(len(schedule) - 1, -1, -1):
        t = torch.full((shown,), step, device=device, dtype=torch.long)
        x0_hat, eps = predict_x0_and_eps(model, x, t, schedule)
        if step in checkpoints:
            captured[step] = x0_hat.cpu().clone()

        beta_t = schedule.betas[step]
        alpha_t = schedule.alphas[step]
        alpha_bar_t = schedule.alpha_bars[step]
        alpha_bar_prev = schedule.alpha_bars_prev[step]
        mean = (x - (beta_t / (1 - alpha_bar_t).sqrt()) * eps) / alpha_t.sqrt()
        variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        x = mean + variance.sqrt() * torch.randn_like(x)

    for i in range(shown):
        for step in checkpoints:
            x0_rows.append(captured[step][i : i + 1])

    save_grid(
        torch.cat(x0_rows),
        f"04_x0_predictions_{args.name}.png",
        ncols=len(checkpoints),
        col_titles=[f"t={s}" for s in checkpoints],
        title=r"The model's running guess at $x_0$ (blurry average $\to$ a specific digit)",
        cell=0.85,
    )

    deltas = [
        (trajectory[i + 1] - trajectory[i]).flatten().abs().mean().item()
        for i in range(trajectory.shape[0] - 1)
    ]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(result.timesteps[:-1], deltas, marker="o", ms=3)
    ax.invert_xaxis()
    ax.set(
        xlabel="timestep t (sampling runs right to left)",
        ylabel="mean |change| per 50 steps",
        title="Where the denoising work actually happens",
    )
    ax.grid(alpha=0.3)
    save_figure(fig, f"04_change_per_step_{args.name}.png")

    final = result.images
    print(f"\nfinal samples: range [{final.min():+.2f}, {final.max():+.2f}], std {final.std():.3f}")
    print("(real MNIST in [-1,1] has std 0.58)")


if __name__ == "__main__":
    main()
