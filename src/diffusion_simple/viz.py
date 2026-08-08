from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from .config import FIGURE_DIR  # noqa: E402


def to_image(x: torch.Tensor) -> torch.Tensor:
    return ((x.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = FIGURE_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    out = resolve(path)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def save_grid(
    images: torch.Tensor,
    path: str | Path,
    *,
    ncols: int = 8,
    col_titles: list[str] | None = None,
    row_labels: list[str] | None = None,
    title: str | None = None,
    cell: float = 1.1,
) -> Path:
    images = to_image(images)
    n = images.shape[0]
    nrows = (n + ncols - 1) // ncols

    fig, _ = plt.subplots(
        nrows, ncols, figsize=(ncols * cell, nrows * cell + (0.5 if title else 0.0))
    )

    for i, ax in enumerate(fig.axes):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i < n:
            ax.imshow(images[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.axis("off")
        if col_titles and i < ncols and i < len(col_titles):
            ax.set_title(col_titles[i], fontsize=9)
        if row_labels and i % ncols == 0 and i // ncols < len(row_labels):
            ax.set_ylabel(row_labels[i // ncols], fontsize=9, rotation=0, ha="right", va="center")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, path)
