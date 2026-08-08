import pytest
import torch

from diffusion_simple.config import DiTConfig
from diffusion_simple.dit import DiT, DiTBlock, modulate
from diffusion_simple.embeddings import timestep_embedding

TINY = DiTConfig(image_size=8, patch_size=2, dim=32, depth=2, heads=4)


def test_forward_preserves_image_shape():
    model = DiT(TINY)
    x = torch.randn(3, 1, 8, 8)
    assert model(x, torch.tensor([0, 5, 999])).shape == x.shape


def test_block_is_the_identity_at_init():
    block = DiTBlock(32, 4, 4.0)
    tokens = torch.randn(2, 16, 32)
    conditioning = torch.randn(2, 32)
    assert torch.equal(block(tokens, conditioning), tokens)


def test_model_output_is_exactly_zero_at_init():
    model = DiT(TINY)
    out = model(torch.randn(4, 1, 8, 8), torch.arange(4))
    assert torch.equal(out, torch.zeros_like(out))


def test_zero_init_survives_the_generic_weight_init():
    model = DiT(TINY)
    for block in model.blocks:
        assert torch.equal(block.adaln[-1].weight, torch.zeros_like(block.adaln[-1].weight))


def test_unpatchify_inverts_patchify():
    model = DiT(TINY)
    img = torch.randn(3, 1, 8, 8)
    grid, patch = TINY.grid_size, TINY.patch_size
    patches = (
        img.reshape(3, 1, grid, patch, grid, patch)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(3, grid * grid, patch * patch)
    )
    assert torch.equal(model.unpatchify(patches), img)


def test_modulate_is_identity_when_scale_and_shift_are_zero():
    x = torch.randn(2, 9, 32)
    zeros = torch.zeros(2, 32)
    assert torch.equal(modulate(x, zeros, zeros), x)


def test_timestep_embedding_is_distinct_per_timestep():
    emb = timestep_embedding(torch.arange(1000), 256)
    assert emb.shape == (1000, 256)
    assert emb.isfinite().all()
    near = torch.cosine_similarity(emb[500], emb[501], dim=0)
    far = torch.cosine_similarity(emb[0], emb[999], dim=0)
    assert near > far


def test_parameterization_is_validated_and_stored():
    assert DiT(TINY, "x0").parameterization == "x0"
    with pytest.raises(ValueError, match="parameterization"):
        DiT(TINY, "velocity")


def test_indivisible_patch_size_raises():
    with pytest.raises(ValueError, match="not divisible"):
        _ = DiTConfig(image_size=28, patch_size=5).grid_size
