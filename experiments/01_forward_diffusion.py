"""Lesson 1 -- Watch an image dissolve into noise.

Run:  uv run experiments/01_forward_diffusion.py
"""

import matplotlib.pyplot as plt
import torch

from diffusion_simple.data import make_dataloader
from diffusion_simple.forward import q_sample
from diffusion_simple.schedules import make_schedule
from diffusion_simple.viz import save_figure, save_grid, to_image

TIMESTEPS = [0, 10, 50, 100, 200, 400, 600, 800, 999]


def main() -> None:
    torch.manual_seed(0)
    schedule = make_schedule("linear", 1000)

    images, labels = next(iter(make_dataloader(batch_size=64, train=False)))

    rows = []
    for i in range(8):
        x0 = images[i : i + 1]
        noise = torch.randn_like(x0)
        for step in TIMESTEPS:
            t = torch.tensor([step])
            x_t, _ = q_sample(x0, t, schedule, noise=noise)
            rows.append(x_t)

    save_grid(
        torch.cat(rows),
        "01_forward_diffusion.png",
        ncols=len(TIMESTEPS),
        col_titles=[f"t={s}" for s in TIMESTEPS],
        row_labels=[str(int(v)) for v in labels[:8]],
        title="Forward diffusion: the same image at increasing noise levels",
    )

    print("\n  t   | sqrt(a_bar)  noise scale | pixel std | corr(x_t, x_0)")
    x0 = images[:64]
    for step in TIMESTEPS:
        t = torch.full((64,), step)
        x_t, _ = q_sample(x0, t, schedule)
        corr = torch.corrcoef(torch.stack([x_t.flatten(), x0.flatten()]))[0, 1]
        print(
            f"{step:5d} |   {schedule.sqrt_alpha_bars[step]:.4f}      "
            f"{schedule.sqrt_one_minus_alpha_bars[step]:.4f}   |   {x_t.std():.3f}   |   {corr:+.4f}"
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    steps = torch.arange(1000)
    ax.plot(steps, schedule.sqrt_alpha_bars, label=r"$\sqrt{\bar\alpha_t}$  (signal kept)")
    ax.plot(
        steps,
        schedule.sqrt_one_minus_alpha_bars,
        label=r"$\sqrt{1-\bar\alpha_t}$  (noise added)",
    )
    for step in TIMESTEPS:
        ax.axvline(step, color="0.85", lw=0.8, zorder=0)
    ax.set(xlabel="timestep t", ylabel="coefficient", title="Signal and noise trade off exactly")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "01_signal_vs_noise.png")

    print(f"\nstd of a real batch at t=0:   {images.std():.3f}")
    x_T, _ = q_sample(images, torch.full((images.shape[0],), 999), schedule)
    print(f"std of the same batch at t=999: {x_T.std():.3f}  (target 1.0)")


if __name__ == "__main__":
    main()
