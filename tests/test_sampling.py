import torch

from diffusion_simple.config import DiTConfig
from diffusion_simple.dit import DiT
from diffusion_simple.forward import q_sample
from diffusion_simple.sampling import (
    ddim_sample,
    ddim_timesteps,
    ddpm_sample,
    predict_x0_and_eps,
)
from diffusion_simple.schedules import make_schedule

TINY = DiTConfig(image_size=8, patch_size=2, dim=32, depth=2, heads=4)


def test_predicted_x0_and_eps_satisfy_the_forward_equation():
    torch.manual_seed(0)
    schedule = make_schedule("linear", 100)
    model = DiT(TINY, "eps")
    x_t = torch.randn(4, 1, 8, 8)
    t = torch.tensor([1, 25, 60, 99])

    x0, eps = predict_x0_and_eps(model, x_t, t, schedule, clip_denoised=False)
    rebuilt, _ = q_sample(x0, t, schedule, noise=eps)
    assert torch.allclose(rebuilt, x_t, atol=1e-4)


def test_both_parameterizations_use_the_same_code_path():
    schedule = make_schedule("linear", 100)
    x_t = torch.randn(2, 1, 8, 8)
    t = torch.tensor([10, 90])
    for parameterization in ("eps", "x0"):
        model = DiT(TINY, parameterization)
        x0, eps = predict_x0_and_eps(model, x_t, t, schedule, clip_denoised=False)
        rebuilt, _ = q_sample(x0, t, schedule, noise=eps)
        assert torch.allclose(rebuilt, x_t, atol=1e-4)


def test_clipping_bounds_x0_and_keeps_the_pair_consistent():
    schedule = make_schedule("linear", 100)
    model = DiT(TINY, "x0")
    x_t = torch.randn(4, 1, 8, 8) * 5.0
    t = torch.tensor([5, 20, 50, 95])

    x0, eps = predict_x0_and_eps(model, x_t, t, schedule, clip_denoised=True)
    assert x0.min() >= -1.0 and x0.max() <= 1.0
    rebuilt, _ = q_sample(x0, t, schedule, noise=eps)
    assert torch.allclose(rebuilt, x_t, atol=1e-4)


def test_ddim_eta_one_matches_the_ddpm_posterior_std():
    s = make_schedule("linear", 1000)
    posterior_std = (s.betas * (1 - s.alpha_bars_prev) / (1 - s.alpha_bars)).sqrt()

    ab_t, ab_prev = s.alpha_bars, s.alpha_bars_prev
    ddim_sigma = ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
    assert torch.allclose(ddim_sigma[1:], posterior_std[1:], atol=1e-5)


def test_ddim_timesteps_are_descending_unique_and_reach_zero():
    times = ddim_timesteps(1000, 50)
    assert len(times) == 50
    assert times[0] == 999 and times[-1] == 0
    assert times == sorted(times, reverse=True)


def test_samplers_return_correct_shapes_and_finite_values():
    torch.manual_seed(0)
    schedule = make_schedule("linear", 20)
    model = DiT(TINY, "eps")

    ddpm = ddpm_sample(model, schedule, num_images=2)
    assert ddpm.images.shape == (2, 1, 8, 8)
    assert ddpm.images.isfinite().all()
    assert ddpm.num_model_calls == 20

    ddim = ddim_sample(model, schedule, num_images=2, steps=5)
    assert ddim.images.shape == (2, 1, 8, 8)
    assert ddim.images.isfinite().all()
    assert ddim.num_model_calls == 5


def test_ddim_with_eta_zero_is_deterministic():
    schedule = make_schedule("linear", 20)
    model = DiT(TINY, "eps")
    a = ddim_sample(
        model, schedule, num_images=2, steps=5,
        generator=torch.Generator().manual_seed(7),
    )
    b = ddim_sample(
        model, schedule, num_images=2, steps=5,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(a.images, b.images)


def test_trajectory_recording_captures_start_and_end():
    schedule = make_schedule("linear", 20)
    model = DiT(TINY, "eps")
    result = ddpm_sample(model, schedule, num_images=1, record_every=5)
    assert result.timesteps[0] == 15
    assert result.timesteps[-1] == -1
    assert len(result.trajectory) == len(result.timesteps)
    assert torch.equal(result.trajectory[-1], result.images.cpu())
