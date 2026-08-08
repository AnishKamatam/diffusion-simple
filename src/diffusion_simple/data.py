import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = REPO_ROOT / "data" / "mnist-dataset.zip"
DEFAULT_RAW = REPO_ROOT / "data" / "raw"

IDX_DTYPE_UNSIGNED_BYTE = 0x08

IDX_FILENAMES = (
    "train-images.idx3-ubyte",
    "train-labels.idx1-ubyte",
    "t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte",
)


def read_idx(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    dtype_code, ndim = raw[2], raw[3]
    if dtype_code != IDX_DTYPE_UNSIGNED_BYTE:
        raise ValueError(
            f"{path} is not an unsigned-byte IDX file: expected dtype code "
            f"0x{IDX_DTYPE_UNSIGNED_BYTE:02x}, found 0x{dtype_code:02x}"
        )
    shape = np.frombuffer(raw, dtype=">u4", count=ndim, offset=4)

    header_size = 4 + 4 * ndim
    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape)


def ensure_extracted(zip_path: Path, dest: Path) -> Path:
    if all((dest / name).exists() for name in IDX_FILENAMES):
        return dest

    if not zip_path.exists():
        raise FileNotFoundError(f"MNIST archive not found at {zip_path}")

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for name in IDX_FILENAMES:
            with archive.open(name) as src, (dest / name).open("wb") as out:
                shutil.copyfileobj(src, out)
    return dest


class MNISTDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        zip_path: Path = DEFAULT_ZIP,
        raw_dir: Path = DEFAULT_RAW,
    ) -> None:
        ensure_extracted(zip_path, raw_dir)
        split = "train" if train else "t10k"

        images = read_idx(raw_dir / f"{split}-images.idx3-ubyte")
        labels = read_idx(raw_dir / f"{split}-labels.idx1-ubyte")

        pixels = torch.from_numpy(images.copy()).float().div_(255.0)
        self.images = pixels.mul_(2.0).sub_(1.0).unsqueeze(1)
        self.labels = torch.from_numpy(labels.copy()).long()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def make_dataloader(
    batch_size: int = 128,
    train: bool = True,
    zip_path: Path = DEFAULT_ZIP,
    raw_dir: Path = DEFAULT_RAW,
) -> DataLoader:
    dataset = MNISTDataset(train=train, zip_path=zip_path, raw_dir=raw_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=0,
    )
