import torch
import torch.nn.functional as F
from torch import nn

from .config import DiTConfig, Parameterization
from .embeddings import TimestepEmbedder


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.zero_init_conditioning()

    def zero_init_conditioning(self) -> None:
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln(
            c
        ).chunk(6, dim=1)
        x = x + gate_attn.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_attn, scale_attn)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.zero_init_conditioning()

    def zero_init_conditioning(self) -> None:
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    def __init__(
        self, config: DiTConfig | None = None, parameterization: Parameterization = "eps"
    ) -> None:
        super().__init__()
        if parameterization not in ("eps", "x0"):
            raise ValueError(f"parameterization must be 'eps' or 'x0', got {parameterization!r}")
        self.config = config or DiTConfig()
        self.parameterization = parameterization

        cfg = self.config
        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.dim))
        self.t_embedder = TimestepEmbedder(cfg.dim)
        self.blocks = nn.ModuleList(
            DiTBlock(cfg.dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)
        )
        self.final_layer = FinalLayer(cfg.dim, cfg.patch_size, cfg.in_channels)
        self.init_weights()

    def init_weights(self) -> None:
        def basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight.view(self.config.dim, -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            block.zero_init_conditioning()
        self.final_layer.zero_init_conditioning()

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch = x.shape[0]
        grid, patch, channels = cfg.grid_size, cfg.patch_size, cfg.in_channels
        x = x.reshape(batch, grid, grid, patch, patch, channels)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(batch, channels, grid * patch, grid * patch)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embedder(t)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
