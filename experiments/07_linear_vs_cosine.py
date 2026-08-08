"""Lesson 7 (experiment B) -- Linear versus cosine schedule, both trained.

Run:  uv run experiments/07_linear_vs_cosine.py
"""

import matplotlib.pyplot as plt
import torch

from diffusion_simple.config import get_device
from diffusion_simple.data import make_dataloader
from diffusion_simple.forward import q_sample
from diffusion_simple.sampling import ddim_sample, ddpm_sample, predict_x0_and_eps
from diffusion_simple.train import load_model
from diffusion_simple.viz import save_figure, save_grid

NAMES = ("eps_linear", "eps_cosine")
COLORS = {"eps_linear": "#1f77b4", "eps_cosine": "#d62728"}
N = 8


def main() -> None:
    device = get_device()
    models = {name: load_model(name, device) for name in NAMES}

    for name, (_, schedule, blob) in models.items():
        signal = schedule.sqrt_alpha_bars
        half = (signal < 0.5).nonzero()[0].item()
        wasted = (signal < 0.1).nonzero()[0].item()
        print(f"{name}: final loss {blob['epoch_losses'][-1]:.5f} | "
              f"50% signal at t={half} | under 10% from t={wasted} "
              f"({100 * (len(schedule) - wasted) / len(schedule):.0f}% of chain)")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in NAMES:
        losses = models[name][2]["epoch_losses"]
        ax.plot(range(1, len(losses) + 1), losses, marker="o", ms=3,
                color=COLORS[name], label=name)
    ax.set(xlabel="epoch", ylabel="mean MSE on eps",
           title="Training loss (comparable: both predict eps)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "07_loss_curves.png")

    batch, _ = next(iter(make_dataloader(batch_size=512, train=False)))
    batch = batch.to(device)
    steps = list(range(10, 1000, 40))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for name in NAMES:
        model, schedule, _ = models[name]
        eps_mse, x0_mse = [], []
        with torch.no_grad():
            for step in steps:
                t = torch.full((512,), step, device=device, dtype=torch.long)
                x_t, noise = q_sample(batch, t, schedule)
                x0_hat, eps_hat = predict_x0_and_eps(
                    model, x_t, t, schedule, clip_denoised=False
                )
                eps_mse.append(torch.nn.functional.mse_loss(eps_hat, noise).item())
                x0_mse.append(torch.nn.functional.mse_loss(x0_hat, batch).item())
        ax1.plot(steps, eps_mse, color=COLORS[name], label=name)
        ax2.plot(steps, x0_mse, color=COLORS[name], label=name)

    ax1.set(xlabel="t", ylabel="MSE on eps", title="Noise-prediction error along the chain")
    ax2.set(xlabel="t", ylabel="MSE on $x_0$", title="Implied clean-image error")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Each model is evaluated on its own schedule -- same t means different noise levels")
    save_figure(fig, "07_per_timestep_error.png")

    rows, labels = [], []
    for sampler_name, sampler in (("DDPM 1000", ddpm_sample), ("DDIM 50", ddim_sample)):
        for name in NAMES:
            model, schedule, _ = models[name]
            kwargs = {"steps": 50} if sampler is ddim_sample else {}
            result = sampler(model, schedule, N,
                             generator=torch.Generator(device).manual_seed(3), **kwargs)
            rows.append(result.images.cpu())
            labels.append(f"{name}\n{sampler_name}")
            print(f"{name:11s} {sampler_name:9s}: std {result.images.std():.3f}")

    save_grid(
        torch.cat(rows),
        "07_samples_linear_vs_cosine.png",
        ncols=N,
        row_labels=labels,
        title="Linear vs cosine schedule, same seed",
        cell=1.3,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in NAMES:
        schedule = models[name][1]
        ax.plot(schedule.sqrt_alpha_bars.cpu(), color=COLORS[name], label=name)
    ax.axhline(0.1, color="0.6", lw=0.8, ls="--")
    ax.annotate("10% signal", (20, 0.12), fontsize=9, color="0.4")
    ax.set(xlabel="t", ylabel=r"$\sqrt{\bar\alpha_t}$", title="What each model was trained on")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "07_signal_curves.png")

    print("\nreal MNIST reference: std 0.578")


if __name__ == "__main__":
    main()
