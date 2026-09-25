from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class SourceConditionedBOVBlock(nn.Module):
    """Construct the virtual target block that predicts the first ordered block.

    The block follows

    ``Q_bov[v] = E_bov[v] + W_s S[i0,v] + W_g mean(M_s) + E_coord[p0,v] + E_type``.

    It intentionally returns token embeddings rather than FSQ contexts.  The
    caller sends these tokens through the same intra-block encoder,
    target-history backbone, next-block adapter, and FSQ head used by real
    target blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        tokens_per_block: int,
        latent_shape_dhw: Sequence[int],
    ) -> None:
        super().__init__()
        if int(tokens_per_block) < 1:
            raise ValueError("tokens_per_block must be positive")
        if len(latent_shape_dhw) != 3 or min(int(v) for v in latent_shape_dhw) < 1:
            raise ValueError(f"Invalid latent_shape_dhw {latent_shape_dhw!r}")

        self.hidden_dim = int(hidden_dim)
        self.tokens_per_block = int(tokens_per_block)
        self.register_buffer(
            "coord_denom",
            torch.tensor(
                [max(int(v) - 1, 1) for v in latent_shape_dhw], dtype=torch.float32
            ),
            persistent=False,
        )

        self.bov_position = nn.Parameter(
            torch.empty(self.tokens_per_block, self.hidden_dim)
        )
        self.bov_type = nn.Parameter(torch.empty(self.hidden_dim))
        self.source_token_norm = nn.LayerNorm(self.hidden_dim)
        self.source_token_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.global_source_norm = nn.LayerNorm(self.hidden_dim)
        self.global_source_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.coord_embed = nn.Sequential(
            nn.Linear(3, self.hidden_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)

        nn.init.normal_(self.bov_position, std=0.02)
        nn.init.normal_(self.bov_type, std=0.02)

    def forward(
        self,
        source_memory: torch.Tensor,
        source_tokens: torch.Tensor,
        b0_token_coords: torch.Tensor,
        source_block_index: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        if source_memory.ndim != 3:
            raise ValueError(
                f"source_memory must be [B,N,C], got {tuple(source_memory.shape)}"
            )
        if source_tokens.ndim != 4:
            raise ValueError(
                f"source_tokens must be [B,N,V,C], got {tuple(source_tokens.shape)}"
            )
        b, n, c = source_memory.shape
        if source_tokens.shape != (b, n, self.tokens_per_block, c):
            raise ValueError(
                "source_tokens must align with source_memory and tokens_per_block: "
                f"expected {(b, n, self.tokens_per_block, c)}, got {tuple(source_tokens.shape)}"
            )
        if c != self.hidden_dim:
            raise ValueError(f"Expected hidden dim {self.hidden_dim}, got {c}")
        if b0_token_coords.shape != (self.tokens_per_block, 3):
            raise ValueError(
                f"b0_token_coords must be {(self.tokens_per_block, 3)}, got {tuple(b0_token_coords.shape)}"
            )

        block_index = int(torch.as_tensor(source_block_index).item())
        if not 0 <= block_index < n:
            raise IndexError(
                f"source_block_index must be in [0,{n}), got {block_index}"
            )
        dense_source = self.source_token_proj(
            self.source_token_norm(source_tokens[:, block_index])
        )
        global_source = self.global_source_proj(
            self.global_source_norm(source_memory.mean(dim=1))
        )[:, None, :]
        coords = b0_token_coords.to(
            device=source_memory.device, dtype=source_memory.dtype
        )
        coords = coords / self.coord_denom.to(device=coords.device, dtype=coords.dtype)
        coords = coords * 2.0 - 1.0
        coord = self.coord_embed(coords).view(1, self.tokens_per_block, self.hidden_dim)
        learned = self.bov_position.view(1, self.tokens_per_block, self.hidden_dim)
        bov_type = self.bov_type.view(1, 1, self.hidden_dim)
        return self.out_norm(learned + dense_source + global_source + coord + bov_type)
