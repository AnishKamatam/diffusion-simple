import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CHECKPOINT_DIR, DiTConfig, TrainConfig, get_device
from .data import make_dataloader
from .dit import DiT
from .forward import q_sample, sample_timesteps
from .schedules import NoiseSchedule, make_schedule


@dataclass
class TrainResult:
    config: TrainConfig
    step_losses: list[float] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)
    seconds: float = 0.0
    checkpoint: Path | None = None


def training_target(
    x0: torch.Tensor, noise: torch.Tensor, parameterization: str
) -> torch.Tensor:
    return noise if parameterization == "eps" else x0


def train(
    cfg: TrainConfig,
    device: torch.device | None = None,
    loader: DataLoader | None = None,
    progress: bool | None = None,
) -> TrainResult:
    if progress is None:
        progress = sys.stderr.isatty()
    device = device or get_device()
    torch.manual_seed(cfg.seed)

    loader = loader or make_dataloader(batch_size=cfg.batch_size, train=True)
    schedule = make_schedule(cfg.schedule, cfg.num_timesteps).to(device)
    model = DiT(cfg.model, cfg.parameterization).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)

    result = TrainResult(config=cfg)
    start = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        running, batches = 0.0, 0
        bar = tqdm(
            loader,
            desc=f"{cfg.name} epoch {epoch + 1}/{cfg.epochs}",
            disable=not progress,
            leave=False,
        )
        for x0, _ in bar:
            x0 = x0.to(device, non_blocking=True)
            t = sample_timesteps(x0.shape[0], cfg.num_timesteps, device)
            x_t, noise = q_sample(x0, t, schedule)

            prediction = model(x_t, t)
            loss = torch.nn.functional.mse_loss(
                prediction, training_target(x0, noise, cfg.parameterization)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            value = loss.item()
            result.step_losses.append(value)
            running += value
            batches += 1
            bar.set_postfix(loss=f"{value:.4f}")

        result.epoch_losses.append(running / batches)
        print(
            f"{cfg.name} epoch {epoch + 1}/{cfg.epochs}  loss {running / batches:.5f}",
            flush=True,
        )

    result.seconds = time.perf_counter() - start
    result.checkpoint = save_checkpoint(model, cfg, result)
    return result


def save_checkpoint(model: DiT, cfg: TrainConfig, result: TrainResult) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.checkpoint_path
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": asdict(cfg.model),
            "parameterization": cfg.parameterization,
            "schedule": cfg.schedule,
            "num_timesteps": cfg.num_timesteps,
            "epoch_losses": result.epoch_losses,
            "step_losses": result.step_losses,
            "seconds": result.seconds,
            "train_config": asdict(replace(cfg, model=DiTConfig())) | {"model": asdict(cfg.model)},
        },
        path,
    )
    return path


def load_model(
    name_or_path: str | Path, device: torch.device | None = None
) -> tuple[DiT, NoiseSchedule, dict]:
    device = device or get_device()
    path = Path(name_or_path)
    if not path.suffix:
        path = CHECKPOINT_DIR / f"{path.name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}. Train it first: "
            f"uv run experiments/03_train_dit.py"
        )

    blob = torch.load(path, map_location=device, weights_only=False)
    model = DiT(DiTConfig(**blob["model_config"]), blob["parameterization"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    schedule = make_schedule(blob["schedule"], blob["num_timesteps"]).to(device)
    return model, schedule, blob
