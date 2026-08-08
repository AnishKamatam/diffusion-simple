"""Lesson 8 (experiment C) -- How few DDIM steps can you get away with?

Run:  uv run experiments/08_ddim_step_sweep.py [--name eps_linear]
"""

import argparse

import matplotlib.pyplot as plt
import torch

from diffusion_simple.config import get_device
from diffusion_simple.sampling import ddim_sample
from diffusion_simple.train import load_model
from diffusion_simple.viz import save_figure, save_grid

STEP_COUNTS = [2, 5, 10, 20, 50, 100, 250, 500]
N = 8
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="eps_linear")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model, schedule, _ = load_model(args.name, device)
    T = len(schedule)

    print(f"reference: {T}-step DDIM (the model's own deterministic limit)...")
    reference = ddim_sample(
        model, schedule, N, steps=T,
        generator=torch.Generator(device).manual_seed(SEED), progress=True,
    )
    print(f"  {reference.seconds:.1f}s\n")

    rows, times, deviations = [], [], []
    print(" steps |   time  | per-image ms | mean |dev| from limit")
    for steps in STEP_COUNTS:
        result = ddim_sample(
            model, schedule, N, steps=steps,
            generator=torch.Generator(device).manual_seed(SEED),
        )
        deviation = (result.images - reference.images).abs().mean().item()
        rows.append(result.images.cpu())
        times.append(result.seconds)
        deviations.append(deviation)
        print(f"{steps:6d} | {result.seconds:6.2f}s | {1000 * result.seconds / N:8.1f} ms  "
              f"|      {deviation:.4f}")

    rows.append(reference.images.cpu())
    save_grid(
        torch.cat(rows),
        f"08_step_sweep_{args.name}.png",
        ncols=N,
        row_labels=[f"{s} steps" for s in STEP_COUNTS] + [f"{T} (limit)"],
        title="Same starting noise, increasing DDIM step count",
        cell=1.15,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(STEP_COUNTS, deviations, marker="o")
    ax1.set(xlabel="DDIM steps", ylabel=f"mean |x - x_{T}steps|",
            title="Discretization error vs step count")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3, which="both")

    ax2.plot(times, deviations, marker="o")
    for steps, t, d in zip(STEP_COUNTS, times, deviations):
        ax2.annotate(f"{steps}", (t, d), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set(xlabel="wall-clock seconds", ylabel="deviation from limit",
            title="The actual trade-off: time bought vs error paid")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle(f"{args.name}: DDIM step-count sweep ({N} images, {device})")
    save_figure(fig, f"08_tradeoff_{args.name}.png")

    print()
    for steps, t, d in zip(STEP_COUNTS, times, deviations):
        relative = d / deviations[0]
        print(f"{steps:4d} steps: {100 * relative:5.1f}% of the 2-step error, "
              f"{reference.seconds / t:5.1f}x faster than the {T}-step limit")


if __name__ == "__main__":
    main()
