"""Lesson 3 -- Train the DiT to predict the noise, and look at what it predicts.

Run:  uv run experiments/03_train_dit.py [--epochs N] [--name NAME]
"""

import argparse

import matplotlib.pyplot as plt
import torch

from diffusion_simple.config import TrainConfig, get_device
from diffusion_simple.data import make_dataloader
from diffusion_simple.forward import q_sample
from diffusion_simple.train import load_model, train
from diffusion_simple.viz import save_figure, save_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="eps_linear")
    p.add_argument("--parameterization", default="eps", choices=["eps", "x0"])
    p.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--figures", action="store_true", help="only redraw figures")
    return p.parse_args()


def plot_loss(result, name: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))

    steps = torch.tensor(result.step_losses)
    smooth = steps.unfold(0, 100, 100).mean(1) if len(steps) > 100 else steps
    ax1.plot(torch.arange(len(smooth)) * 100, smooth, lw=1.2)
    ax1.set(xlabel="step", ylabel="MSE loss", title="training loss (100-step mean)")
    ax1.set_yscale("log")

    ax2.plot(range(1, len(result.epoch_losses) + 1), result.epoch_losses, marker="o", ms=3)
    ax2.set(xlabel="epoch", ylabel="mean MSE loss", title="per-epoch loss")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)

    fig.suptitle(f"{name}: {result.seconds / 60:.1f} min on {get_device()}")
    save_figure(fig, f"03_loss_{name}.png")


def plot_true_vs_predicted(name: str) -> None:
    """The lesson's real payoff: is the predicted noise actually the noise?"""
    device = get_device()
    model, schedule, _ = load_model(name, device)

    x0, _ = next(iter(make_dataloader(batch_size=6, train=False)))
    x0 = x0.to(device)
    timesteps = [50, 250, 500, 750, 999]

    rows, labels = [], []
    for step in timesteps:
        t = torch.full((6,), step, device=device, dtype=torch.long)
        x_t, noise = q_sample(x0, t, schedule)
        with torch.no_grad():
            predicted = model(x_t, t)
        if model.parameterization == "x0":
            noise, predicted = x0, predicted
        rows += [x_t.cpu(), noise.cpu(), predicted.cpu()]
        target = "x_0" if model.parameterization == "x0" else "noise"
        labels += [f"x_t  (t={step})", f"true {target}", f"predicted {target}"]

    save_grid(
        torch.cat(rows),
        f"03_true_vs_predicted_{name}.png",
        ncols=6,
        row_labels=labels,
        title=f"{name}: input, target, and prediction across timesteps",
        cell=1.0,
    )

    print("\n  t    | corr(true, predicted) | MSE")
    for step in timesteps:
        t = torch.full((256,), step, device=device, dtype=torch.long)
        x0_batch, _ = next(iter(make_dataloader(batch_size=256, train=False)))
        x0_batch = x0_batch.to(device)
        x_t, noise = q_sample(x0_batch, t, schedule)
        with torch.no_grad():
            predicted = model(x_t, t)
        target = x0_batch if model.parameterization == "x0" else noise
        corr = torch.corrcoef(torch.stack([target.flatten(), predicted.flatten()]))[0, 1]
        mse = torch.nn.functional.mse_loss(predicted, target)
        print(f"{step:5d}  |        {corr:+.4f}        | {mse:.4f}")


def main() -> None:
    args = parse_args()

    if not args.figures:
        cfg = TrainConfig(
            name=args.name,
            parameterization=args.parameterization,
            schedule=args.schedule,
            epochs=args.epochs,
        )
        print(f"training {cfg.name}: {cfg.parameterization}-prediction, {cfg.schedule} schedule")
        result = train(cfg, loader=make_dataloader(batch_size=cfg.batch_size))
        print(f"\ndone in {result.seconds / 60:.1f} min -> {result.checkpoint}")
        plot_loss(result, cfg.name)

    plot_true_vs_predicted(args.name)


if __name__ == "__main__":
    main()
