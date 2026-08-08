import torch

from diffusion_simple.forward import extract, q_sample, sample_timesteps
from diffusion_simple.schedules import make_schedule


def test_extract_reshapes_for_broadcasting():
    s = make_schedule("linear", 100)
    t = torch.tensor([0, 5, 99])
    out = extract(s.alpha_bars, t, torch.Size([3, 1, 28, 28]))
    assert out.shape == (3, 1, 1, 1)
    assert torch.equal(out.flatten(), s.alpha_bars[t])


def test_each_image_uses_its_own_timestep():
    s = make_schedule("linear", 100)
    x0 = torch.ones(3, 1, 28, 28)
    t = torch.tensor([0, 50, 99])
    x_t, _ = q_sample(x0, t, s, noise=torch.zeros_like(x0))
    for i in range(3):
        assert torch.allclose(x_t[i], s.sqrt_alpha_bars[t[i]] * x0[i], atol=1e-6)


def test_q_sample_at_t0_is_almost_the_original():
    s = make_schedule("linear", 1000)
    x0 = torch.randn(4, 1, 28, 28)
    x_t, _ = q_sample(x0, torch.zeros(4, dtype=torch.long), s)
    assert (x_t - x0).abs().max() < 0.06


def test_q_sample_at_final_step_is_essentially_pure_noise():
    s = make_schedule("linear", 1000)
    x0 = torch.randn(64, 1, 28, 28)
    t = torch.full((64,), 999)
    x_t, noise = q_sample(x0, t, s)
    correlation = torch.corrcoef(torch.stack([x_t.flatten(), noise.flatten()]))[0, 1]
    assert correlation > 0.999
    assert abs(x_t.std().item() - 1.0) < 0.02


def test_closed_form_matches_running_the_chain():
    torch.manual_seed(0)
    s = make_schedule("linear", 1000)
    n = 100_000
    x = x0 = torch.full((n,), 0.7)
    for step in range(300):
        x = s.alphas[step].sqrt() * x + (1 - s.alphas[step]).sqrt() * torch.randn(n)

    expected_mean = (s.sqrt_alpha_bars[299] * 0.7).item()
    expected_std = s.sqrt_one_minus_alpha_bars[299].item()
    assert abs(x.mean().item() - expected_mean) < 0.01
    assert abs(x.std().item() - expected_std) < 0.01
    assert x0.shape == x.shape


def test_supplying_noise_makes_q_sample_deterministic():
    s = make_schedule("cosine", 100)
    x0 = torch.randn(2, 1, 28, 28)
    noise = torch.randn_like(x0)
    t = torch.tensor([10, 20])
    a, na = q_sample(x0, t, s, noise=noise)
    b, nb = q_sample(x0, t, s, noise=noise)
    assert torch.equal(a, b)
    assert torch.equal(na, noise) and torch.equal(nb, noise)


def test_sample_timesteps_stays_in_range():
    t = sample_timesteps(1000, 250, torch.device("cpu"))
    assert t.shape == (1000,)
    assert t.dtype == torch.long
    assert t.min() >= 0 and t.max() < 250
