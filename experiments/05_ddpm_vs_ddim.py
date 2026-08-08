"""Lesson 5 -- 1000-step DDPM against 50-step DDIM, same model.

Run:  uv run experiments/05_ddpm_vs_ddim.py [--name eps_linear]
"""

import argparse

import matplotlib.pyplot as plt
import torch

from diffusion_simple.config import get_device
from diffusion_simple.sampling import ddim_sample, ddpm_sample
from diffusion_simple.train import load_model
from diffusion_simple.viz import save_figure, save_grid

N = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="eps_linear")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model, schedule, _ = load_model(args.name, device)

    seed = 1234
    print(f"DDPM, {len(schedule)} steps...")
    ddpm = ddpm_sample(
        model, schedule, N, generator=torch.Generator(device).manual_seed(seed), progress=True
    )
    print(f"  {ddpm.seconds:.2f}s, {ddpm.num_model_calls} model calls")

    print("DDIM, 50 steps...")
    ddim = ddim_sample(
        model, schedule, N, steps=50,
        generator=torch.Generator(device).manual_seed(seed), progress=True,
    )
    print(f"  {ddim.seconds:.2f}s, {ddim.num_model_calls} model calls")

    speedup = ddpm.seconds / ddim.seconds
    print(f"\nDDIM is {speedup:.1f}x faster ({ddpm.num_model_calls // ddim.num_model_calls}x fewer model calls)")

    save_grid(
        torch.cat([ddpm.images.cpu(), ddim.images.cpu()]),
        f"05_ddpm_vs_ddim_{args.name}.png",
        ncols=N,
        row_labels=[
            f"DDPM {ddpm.num_model_calls} steps\n{ddpm.seconds:.1f}s",
            f"DDIM {ddim.num_model_calls} steps\n{ddim.seconds:.1f}s",
        ],
        title="Same model, same starting noise, two samplers",
        cell=1.3,
    )

    configs = [(5, "ddim"), (10, "ddim"), (20, "ddim"), (50, "ddim"), (100, "ddim"), (250, "ddim")]
    times, calls = [], []
    for steps, _ in configs:
        r = ddim_sample(model, schedule, N, steps=steps,
                        generator=torch.Generator(device).manual_seed(seed))
        times.append(r.seconds)
        calls.append(r.num_model_calls)
        print(f"DDIM {steps:4d} steps: {r.seconds:6.2f}s")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(calls, times, marker="o", label="DDIM")
    ax.scatter([ddpm.num_model_calls], [ddpm.seconds], color="crimson", zorder=5, label="DDPM")
    ax.set(
        xlabel="model calls (= sampling steps)",
        ylabel="wall-clock seconds",
        title=f"Sampling cost is linear in steps ({N} images, {device})",
    )
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, f"05_timing_{args.name}.png")

    print("\neta sweep at 50 steps (0 = deterministic, 1 = DDPM-like):")
    rows = []
    etas = [0.0, 0.25, 0.5, 1.0]
    for eta in etas:
        r = ddim_sample(model, schedule, N, steps=50, eta=eta,
                        generator=torch.Generator(device).manual_seed(seed))
        rows.append(r.images.cpu())
        print(f"  eta={eta:.2f}: std {r.images.std():.3f}")

    save_grid(
        torch.cat(rows),
        f"05_eta_sweep_{args.name}.png",
        ncols=N,
        row_labels=[f"eta={e}" for e in etas],
        title="DDIM eta: 0 is deterministic, 1 matches the DDPM posterior noise",
        cell=1.2,
    )

    a = ddim_sample(model, schedule, 4, steps=50,
                    generator=torch.Generator(device).manual_seed(99))
    b = ddim_sample(model, schedule, 4, steps=50,
                    generator=torch.Generator(device).manual_seed(99))
    print(f"\nDDIM eta=0 reproducible from the same seed: {torch.equal(a.images, b.images)}")

    c = ddpm_sample(model, schedule, 4, generator=torch.Generator(device).manual_seed(99))
    d = ddpm_sample(model, schedule, 4, generator=torch.Generator(device).manual_seed(99))
    print(f"DDPM reproducible from the same seed:      {torch.equal(c.images, d.images)}")
    print("(both reproduce, but DDPM consumes fresh randomness at every one of its steps)")


if __name__ == "__main__":
    main()
