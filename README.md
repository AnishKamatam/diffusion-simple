# diffusion-simple

A diffusion model built from scratch on MNIST, one concept at a time — using a
**DiT** (diffusion transformer) rather than the usual U-Net.

Everything is written to be read. No `diffusers`, no `torchvision`: the IDX binaries
are parsed by hand, the noise schedules are derived from `beta_t`, and every term of
the reverse-step equation is explained in a comment next to the line that computes it.

## Setup

```bash
uv sync
uv run pytest          # 42 tests covering the schedule math and model invariants
```

MNIST auto-extracts from `data/mnist-dataset.zip` on first use. Training runs on MPS,
CUDA, or CPU — whichever is available.

## The curriculum

Run them in order. Each writes figures to `outputs/figures/`.

| # | Command | What you learn |
|---|---|---|
| 1 | `uv run experiments/01_forward_diffusion.py` | The forward process is one closed-form equation with zero learned parameters |
| 2 | `uv run experiments/02_schedules.py` | `beta_t` → `alpha_t` → `alpha_bar_t`, and why the schedule choice matters |
| 3 | `uv run experiments/03_train_dit.py` | Train the DiT; compare true noise against predicted noise |
| 4 | `uv run experiments/04_reverse_sampling.py` | Run the chain backwards; watch noise become a digit |
| 5 | `uv run experiments/05_ddpm_vs_ddim.py` | 1000-step DDPM vs 50-step DDIM on the same weights |
| 6 | `uv run experiments/06_x0_vs_eps.py` | **Experiment A:** predicting `x_0` instead of `eps` |
| 7 | `uv run experiments/07_linear_vs_cosine.py` | **Experiment B:** linear vs cosine schedule |
| 8 | `uv run experiments/08_ddim_step_sweep.py` | **Experiment C:** how few DDIM steps you can get away with |

Lessons 1 and 2 need no model. Lesson 3 produces `checkpoints/eps_linear.pt`, which
4, 5 and 8 consume. Lessons 6 and 7 train one extra model each:

```bash
uv run experiments/03_train_dit.py --name x0_linear  --parameterization x0
uv run experiments/03_train_dit.py --name eps_cosine --schedule cosine
```

## The library

```
src/diffusion_simple/
├── data.py         IDX parser -> Dataset -> DataLoader, rescaled to [-1, 1]
├── schedules.py    linear + cosine betas, and every table derived from them
├── forward.py      q_sample: the closed-form jump to any timestep
├── embeddings.py   sinusoidal timestep embedding
├── dit.py          patchify, adaLN-Zero transformer blocks, DiT
├── train.py        training loop and checkpointing
├── sampling.py     DDPM and DDIM, heavily commented
├── viz.py          figures
└── config.py       DiTConfig, TrainConfig, device selection
```

## Three ideas worth carrying away

**1. Forward diffusion needs no simulation.** Chaining `t` noising steps collapses
into a single equation:

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
```

because the sum of independent Gaussians is Gaussian. Training samples a random `t`
per image and jumps straight there. Without this, diffusion would be untrainable.
`tests/test_forward.py` verifies the shortcut against 300 simulated steps.

**2. eps-prediction and x0-prediction are the same model wearing different clothes.**
The equation above is invertible, so either output determines the other:

```
eps-model  ->  x0_hat  = (x_t - sqrt(1-alpha_bar) * eps_hat) / sqrt(alpha_bar)
x0-model   ->  eps_hat = (x_t - sqrt(alpha_bar)   * x0_hat)  / sqrt(1-alpha_bar)
```

`predict_x0_and_eps` normalizes both into the same pair, which is why the samplers
contain no branching on parameterization and lesson 6 required no new sampling code.
What actually differs is how squared error gets weighted across timesteps.

**3. adaLN-Zero makes a deep transformer start as the identity function.** Each DiT
block regresses its LayerNorm scale/shift *and a residual gate* from the timestep. The
gate is zero-initialized, so at step 0 every block passes its input straight through
and the network outputs exactly zero. Training begins from a well-behaved identity map
instead of random noise. `tests/test_dit.py` asserts both properties.

## Architecture

A faithful, small DiT (Peebles & Xie, 2023):

```
x_t (1,28,28) --patchify 4x4--> 49 tokens x 192
                                     + learned position embedding
t --sinusoidal--> MLP --> c (192)
                                     |
              6 x DiTBlock(adaLN-Zero, conditioned on c)
                                     |
              adaLN final layer -> unpatchify -> eps_hat (1,28,28)
```

~4.2M parameters. `patch_size=4` gives 49 tokens; `patch_size=2` gives 196 and
sharper detail, but attention is quadratic in token count and it trains ~4.6× slower.
Change it in `DiTConfig`.
