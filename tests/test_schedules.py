import pytest
import torch

from diffusion_simple.schedules import make_schedule

KINDS = ("linear", "cosine")


@pytest.mark.parametrize("kind", KINDS)
def test_alpha_bar_decreases_from_one_to_zero(kind):
    s = make_schedule(kind, 1000)
    assert torch.all(s.alpha_bars[1:] < s.alpha_bars[:-1])
    assert s.alpha_bars.max() <= 1.0
    assert s.alpha_bars.min() > 0.0
    assert s.alpha_bars[0] > 0.999
    assert s.alpha_bars[-1] < 1e-3


@pytest.mark.parametrize("kind", KINDS)
def test_betas_are_valid_probabilities(kind):
    s = make_schedule(kind, 1000)
    assert torch.all(s.betas > 0.0)
    assert torch.all(s.betas <= 0.999)
    assert torch.allclose(s.alphas, 1.0 - s.betas)


@pytest.mark.parametrize("kind", KINDS)
def test_alpha_bars_prev_is_shifted_with_leading_one(kind):
    s = make_schedule(kind, 100)
    assert s.alpha_bars_prev[0] == 1.0
    assert torch.equal(s.alpha_bars_prev[1:], s.alpha_bars[:-1])


@pytest.mark.parametrize("kind", KINDS)
def test_sqrt_tables_match_their_definitions(kind):
    s = make_schedule(kind, 500)
    assert torch.allclose(s.sqrt_alpha_bars, s.alpha_bars.sqrt(), atol=1e-6)
    assert torch.allclose(s.sqrt_one_minus_alpha_bars, (1 - s.alpha_bars).sqrt(), atol=1e-6)


def test_posterior_variance_is_zero_at_t0():
    s = make_schedule("linear", 200)
    posterior = s.betas * (1 - s.alpha_bars_prev) / (1 - s.alpha_bars)
    assert posterior[0] == 0.0
    assert torch.all(posterior[1:] > 0.0)
    assert bool(posterior.isfinite().all())


def test_cosine_keeps_more_signal_through_the_bulk_of_the_chain():
    lin = make_schedule("linear", 1000).alpha_bars
    cos = make_schedule("cosine", 1000).alpha_bars
    assert torch.all(cos[1:990] > lin[1:990])
    assert (cos.sqrt() < 0.5).nonzero()[0] > (lin.sqrt() < 0.5).nonzero()[0] + 200


def test_cosine_destroys_more_completely_at_the_very_end():
    lin = make_schedule("linear", 1000).alpha_bars
    cos = make_schedule("cosine", 1000).alpha_bars
    assert cos[-1] < lin[-1]


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown schedule kind"):
        make_schedule("quadratic", 100)  # type: ignore[arg-type]


def test_to_moves_every_field():
    s = make_schedule("linear", 50).to("cpu")
    assert s.alpha_bars_prev.device.type == "cpu"
    assert len(s) == 50
