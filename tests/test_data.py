import numpy as np
import pytest
import torch

from diffusion_simple.data import DEFAULT_RAW, MNISTDataset, make_dataloader, read_idx


def test_read_idx_parses_images_and_labels():
    images = read_idx(DEFAULT_RAW / "train-images.idx3-ubyte")
    labels = read_idx(DEFAULT_RAW / "train-labels.idx1-ubyte")
    assert images.shape == (60000, 28, 28)
    assert labels.shape == (60000,)
    assert images.dtype == np.uint8 and images.max() == 255
    assert labels[:10].tolist() == [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]


def test_read_idx_rejects_a_non_idx_file(tmp_path):
    bogus = tmp_path / "bogus.ubyte"
    bogus.write_bytes(b"\x00\x00\xff\x03" + b"\x00" * 32)
    with pytest.raises(ValueError, match="not an unsigned-byte IDX file"):
        read_idx(bogus)


def test_dataset_is_scaled_to_minus_one_to_one():
    dataset = MNISTDataset(train=True)
    x, y = dataset[0]
    assert x.shape == (1, 28, 28)
    assert x.dtype == torch.float32
    assert y.item() == 5
    assert dataset.images.min() == -1.0
    assert dataset.images.max() == 1.0


def test_test_split_is_smaller():
    assert len(MNISTDataset(train=False)) == 10000
    assert len(MNISTDataset(train=True)) == 60000


def test_dataloader_yields_batched_images():
    loader = make_dataloader(batch_size=64)
    x, y = next(iter(loader))
    assert x.shape == (64, 1, 28, 28)
    assert y.shape == (64,)
    assert x.min() >= -1.0 and x.max() <= 1.0
