from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spira_ar.spira.utils.raster_order import as_3tuple, raster_coords_tensor


class ScalarTokenEmbedder(nn.Module):
    def __init__(
        self, levels: Sequence[int], hidden_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.levels = [int(v) for v in levels]
        self.embeds = nn.ModuleList(
            [nn.Embedding(level, hidden_dim) for level in self.levels]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @property
    def scalar_dim(self) -> int:
        return len(self.levels)

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        if scalars.shape[-1] != self.scalar_dim:
            raise ValueError(
                f"Expected scalar dim {self.scalar_dim}, got {tuple(scalars.shape)}"
            )
        out = None
        for j, emb in enumerate(self.embeds):
            current = emb(scalars[..., j].long())
            out = current if out is None else out + current
        return self.dropout(self.norm(out))


class SourceBlockEncoder(nn.Module):
    """Contextual dense source encoder for spatially aligned 3D FSQ blocks.

    The encoder has two levels:

    1. A block-local bidirectional Transformer models the ``V`` source tokens
       inside every block.
    2. A four-layer bidirectional Transformer models all source blocks after
       adding their normalized 3D coordinates.

    It returns both contextual block memory ``[B,N,C]`` for source
    cross-attention and contextual per-token source features ``[B,N,V,C]`` for
    the co-located dense source term used by the next-block predictor.
    """

    def __init__(
        self,
        levels: Sequence[int],
        hidden_dim: int = 768,
        tokens_per_block: int = 64,
        *,
        block_grid_dhw: Sequence[int] | None = None,
        local_num_layers: int = 1,
        local_num_heads: int = 8,
        global_num_layers: int = 4,
        global_num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % int(local_num_heads):
            raise ValueError("hidden_dim must be divisible by local_num_heads")
        if hidden_dim % int(global_num_heads):
            raise ValueError("hidden_dim must be divisible by global_num_heads")
        if int(local_num_layers) < 1:
            raise ValueError("local_num_layers must be at least 1")
        if int(global_num_layers) < 1:
            raise ValueError("global_num_layers must be at least 1")

        self.hidden_dim = int(hidden_dim)
        self.tokens_per_block = int(tokens_per_block)
        self.block_grid_dhw = (
            None
            if block_grid_dhw is None
            else as_3tuple(block_grid_dhw, "block_grid_dhw")
        )

        default_coords = None
        if self.block_grid_dhw is not None:
            default_coords = raster_coords_tensor(self.block_grid_dhw)
        self.register_buffer("default_block_coords", default_coords, persistent=False)

        self.token_embed = ScalarTokenEmbedder(levels, hidden_dim, dropout=dropout)
        self.intra_pos = nn.Parameter(torch.zeros(tokens_per_block, hidden_dim))

        local_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(local_num_heads),
            dim_feedforward=max(int(hidden_dim * mlp_ratio), hidden_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(
            local_layer, num_layers=int(local_num_layers)
        )
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1)

        self.coord_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        global_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(global_num_heads),
            dim_feedforward=max(int(hidden_dim * mlp_ratio), hidden_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            global_layer, num_layers=int(global_num_layers)
        )
        self.global_norm = nn.LayerNorm(hidden_dim)

        self.block_to_token = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.token_out_norm = nn.LayerNorm(hidden_dim)

        nn.init.normal_(self.intra_pos, std=0.02)

    def _resolve_coords(
        self, block_coords: torch.Tensor | None, num_blocks: int, device
    ) -> torch.Tensor:
        coords = self.default_block_coords if block_coords is None else block_coords
        if coords is None:
            raise ValueError(
                "block_coords must be supplied when block_grid_dhw was not provided at construction"
            )
        if coords.ndim != 2 or coords.shape != (num_blocks, 3):
            raise ValueError(
                f"Expected block_coords [{num_blocks},3], got {tuple(coords.shape)}"
            )
        return coords.to(device=device)

    @staticmethod
    def _normalize_coords(coords: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        coords_f = coords.to(dtype=dtype)
        coord_max = coords_f.max(dim=0).values
        denom = coord_max.clamp_min(1.0)
        normalized = coords_f / denom
        normalized = normalized * 2.0 - 1.0
        return torch.where(
            coord_max.view(1, 3) > 0, normalized, torch.zeros_like(normalized)
        )

    def forward(
        self,
        source_blocks: torch.Tensor,
        block_coords: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if source_blocks.ndim != 4:
            raise ValueError(
                f"Expected source_blocks [B,N,V,d], got {tuple(source_blocks.shape)}"
            )
        b, n, v, _ = source_blocks.shape
        if v != self.tokens_per_block:
            raise ValueError(
                f"Expected {self.tokens_per_block} tokens per block, got {v}"
            )

        coords = self._resolve_coords(block_coords, n, source_blocks.device)

        x = self.token_embed(source_blocks)
        x = x + self.intra_pos.view(1, 1, self.tokens_per_block, self.hidden_dim)
        x = self.local_encoder(x.reshape(b * n, v, self.hidden_dim))
        local_tokens = self.local_norm(x).reshape(b, n, v, self.hidden_dim)

        pool_weights = self.pool_score(local_tokens).softmax(dim=2)
        block_tokens = (local_tokens * pool_weights).sum(dim=2)
        coord_pos = self.coord_embed(
            self._normalize_coords(coords, block_tokens.dtype)
        ).view(1, n, self.hidden_dim)
        memory = self.global_norm(self.global_encoder(block_tokens + coord_pos))

        # Broadcast globally contextualized block features back to every dense
        # source token, while preserving the block-local token representation.
        contextual_tokens = self.token_out_norm(
            local_tokens + self.block_to_token(memory)[:, :, None, :]
        )

        if return_tokens:
            return memory, contextual_tokens
        return memory
