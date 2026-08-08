"""Lesson 6 (experiment A) -- Predicting eps versus predicting x_0.

Run:  uv run experiments/06_x0_vs_eps.py
"""

import matplotlib.pyplot as plt
import torch

from diffusion_simple.config import get_device
from diffusion_simple.data import make_dataloader
from diffusion_simple.forward import q_sample
from diffusion_simple.sampling import ddim_sample, ddpm_sample, predict_x0_and_eps
from diffusion_simple.train import load_model
from diffusion_simple.viz import save_figure, save_grid

NAMES = ("eps_linear", "x0_linear")
COLORS = {"eps_linear": "#1f77b4", "x0_linear": "#d62728"}
N = 8


@torch.no_grad()
def per_timestep_errors(model, schedule, batch, steps):
    """Measure BOTH errors for a model, whatever it was trained to output."""
    eps_mse, x0_mse = [], []
    for step in steps:
        t = torch.full((batch.shape[0],), step, device=batch.device, dtype=torch.long)
        x_t, noise = q_sample(batch, t, schedule)
        x0_hat, eps_hat = predict_x0_and_eps(model, x_t, t, schedule, clip_denoised=False)
        eps_mse.append(torch.nn.functional.mse_loss(eps_hat, noise).item())
        x0_mse.append(torch.nn.functional.mse_loss(x0_hat, batch).item())
    return eps_mse, x0_mse


def main() -> None:
    device = get_device()
    models = {}
    for name in NAMES:
        model, schedule, blob = load_model(name, device)
        models[name] = (model, schedule, blob)
        print(f"{name}: {blob['parameterization']}-prediction, "
              f"final train loss {blob['epoch_losses'][-1]:.5f} "
              f"({blob['seconds'] / 60:.1f} min)")
    print("\n(those two losses measure different targets and are not comparable)")

    batch, _ = next(iter(make_dataloader(batch_size=512, train=False)))
    batch = batch.to(device)
    steps = list(range(10, 1000, 40))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for name in NAMES:
        model, schedule, _ = models[name]
        eps_mse, x0_mse = per_timestep_errors(model, schedule, batch, steps)
        ax1.plot(steps, eps_mse, label=name, color=COLORS[name])
        ax2.plot(steps, x0_mse, label=name, color=COLORS[name])
        print(f"\n{name}")
        print("   t  |  eps MSE  |  x0 MSE")
        for i, step in enumerate(steps[::5]):
            j = steps.index(step)
            print(f"{step:5d} |  {eps_mse[j]:.4f}   |  {x0_mse[j]:.4f}")

    ax1.set(xlabel="timestep t", ylabel="MSE on eps", title="Error at predicting the noise")
    ax2.set(xlabel="timestep t", ylabel="MSE on $x_0$", title="Error at predicting the clean image")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Same two models, both metrics: each parameterization wins where it was trained")
    save_figure(fig, "06_per_timestep_error.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, name in zip(axes, NAMES):
        losses = models[name][2]["epoch_losses"]
        ax.plot(range(1, len(losses) + 1), losses, marker="o", ms=3, color=COLORS[name])
        ax.set(xlabel="epoch", ylabel="MSE", title=f"{name} training loss")
        ax.grid(alpha=0.3)
    save_figure(fig, "06_loss_curves.png")

    rows, labels = [], []
    for sampler_name, sampler in (("DDPM 1000", ddpm_sample), ("DDIM 50", ddim_sample)):
        for name in NAMES:
            model, schedule, _ = models[name]
            kwargs = {"steps": 50} if sampler is ddim_sample else {}
            result = sampler(
                model, schedule, N,
                generator=torch.Generator(device).manual_seed(7), **kwargs,
            )
            rows.append(result.images.cpu())
            labels.append(f"{name}\n{sampler_name}")
            print(f"{name:11s} {sampler_name:9s}: {result.seconds:5.1f}s  "
                  f"std {result.images.std():.3f}")

    save_grid(
        torch.cat(rows),
        "06_samples_x0_vs_eps.png",
        ncols=N,
        row_labels=labels,
        title=r"$\epsilon$-prediction vs $x_0$-prediction, same seed",
        cell=1.3,
    )

    print("\nreal MNIST reference: std 0.578")


if __name__ == "__main__":
    main()
