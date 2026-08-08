"""Lesson 2 -- What the beta schedule actually controls.

Run:  uv run experiments/02_schedules.py
"""

import matplotlib.pyplot as plt
import torch

from diffusion_simple.data import make_dataloader
from diffusion_simple.forward import q_sample
from diffusion_simple.schedules import make_schedule
from diffusion_simple.viz import save_figure, save_grid

T = 1000
KINDS = ("linear", "cosine")
COLORS = {"linear": "#1f77b4", "cosine": "#d62728"}


def main() -> None:
    torch.manual_seed(0)
    schedules = {kind: make_schedule(kind, T) for kind in KINDS}
    steps = torch.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    for kind, s in schedules.items():
        axes[0].plot(steps, s.betas, label=kind, color=COLORS[kind])
        axes[1].plot(steps, s.alphas, label=kind, color=COLORS[kind])
        axes[2].plot(steps, s.alpha_bars, label=kind, color=COLORS[kind])

    axes[0].set(xlabel="t", ylabel=r"$\beta_t$", title=r"$\beta_t$: noise added per step")
    axes[0].set_yscale("log")
    axes[1].set(xlabel="t", ylabel=r"$\alpha_t$", title=r"$\alpha_t = 1-\beta_t$")
    axes[2].set(
        xlabel="t", ylabel=r"$\bar\alpha_t$", title=r"$\bar\alpha_t$: fraction of $x_0$ surviving"
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Beta schedules and everything derived from them")
    save_figure(fig, "02_schedules.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for kind, s in schedules.items():
        ax.plot(steps, s.sqrt_alpha_bars, color=COLORS[kind], label=kind)
        half = (s.sqrt_alpha_bars < 0.5).nonzero()[0].item()
        ax.axvline(half, color=COLORS[kind], ls="--", lw=1)
        ax.annotate(
            f"{kind}: 50% at t={half}",
            (half, 0.5),
            textcoords="offset points",
            xytext=(8, 12),
            color=COLORS[kind],
            fontsize=9,
        )
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.set(
        xlabel="timestep t",
        ylabel=r"$\sqrt{\bar\alpha_t}$  (signal remaining)",
        title="Linear destroys the image far earlier than cosine",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "02_signal_remaining.png")

    print("\n  t   |  linear a_bar   signal% |  cosine a_bar   signal%")
    for step in (0, 50, 100, 200, 300, 500, 700, 900, 999):
        lin, cos = schedules["linear"], schedules["cosine"]
        print(
            f"{step:5d} |    {lin.alpha_bars[step]:.4f}      {100 * lin.sqrt_alpha_bars[step]:5.1f}% "
            f"|    {cos.alpha_bars[step]:.4f}      {100 * cos.sqrt_alpha_bars[step]:5.1f}%"
        )

    for kind, s in schedules.items():
        half = (s.sqrt_alpha_bars < 0.5).nonzero()[0].item()
        gone = (s.sqrt_alpha_bars < 0.1).nonzero()[0].item()
        print(f"\n{kind:7s} 50% signal at t={half}, under 10% from t={gone} "
              f"({100 * (T - gone) / T:.0f}% of the chain is near-pure noise)")

    images, _ = next(iter(make_dataloader(batch_size=1, train=False)))
    noise = torch.randn_like(images)
    shown = [0, 100, 200, 400, 600, 800, 999]

    rows = []
    for kind in KINDS:
        for step in shown:
            x_t, _ = q_sample(images, torch.tensor([step]), schedules[kind], noise=noise)
            rows.append(x_t)

    save_grid(
        torch.cat(rows),
        "02_schedule_comparison.png",
        ncols=len(shown),
        col_titles=[f"t={s}" for s in shown],
        row_labels=list(KINDS),
        title="Same image, same noise, different schedule",
    )


if __name__ == "__main__":
    main()
