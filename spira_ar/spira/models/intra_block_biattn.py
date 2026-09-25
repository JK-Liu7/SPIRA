from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spira_ar.spira.models.source_block_encoder import ScalarTokenEmbedder


class TargetBlockEmbedder(nn.Module):
    def __init__(
        self,
        levels: Sequence[int],
        hidden_dim: int = 768,
        tokens_per_block: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.scalar_embed = ScalarTokenEmbedder(levels, hidden_dim, dropout=dropout)
        self.intra_pos = nn.Parameter(torch.zeros(tokens_per_block, hidden_dim))
        nn.init.normal_(self.intra_pos, std=0.02)

    def forward(self, target_blocks: torch.Tensor) -> torch.Tensor:
        x = self.scalar_embed(target_blocks)
        return x + self.intra_pos.view(1, 1, self.intra_pos.shape[0], -1)


class IntraBlockBiAttn(nn.Module):
    """Bidirectional attention over the 64 visible tokens inside each known block."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _flatten_token_mask(
        token_valid_mask: torch.Tensor,
        *,
        leading_shape: tuple[int, ...],
        tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        expected = (*leading_shape, tokens)
        if tuple(token_valid_mask.shape) != expected:
            raise ValueError(
                f"token_valid_mask must be {expected}, "
                f"got {tuple(token_valid_mask.shape)}"
            )
        return token_valid_mask.to(device=device, dtype=torch.bool).reshape(-1, tokens)

    def forward(
        self,
        block_tokens: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if block_tokens.ndim not in {3, 4}:
            raise ValueError(
                f"Expected [B,N,V,C] or [B,V,C], got {tuple(block_tokens.shape)}"
            )
        leading_shape = tuple(int(value) for value in block_tokens.shape[:-2])
        tokens, channels = block_tokens.shape[-2:]
        flattened = block_tokens.reshape(-1, tokens, channels)
        flattened_mask = None
        if token_valid_mask is not None:
            flattened_mask = self._flatten_token_mask(
                token_valid_mask,
                leading_shape=leading_shape,
                tokens=tokens,
                device=block_tokens.device,
            )
            # Invalid scalar tokens must not enter either attention or the FFN.
            flattened = flattened * flattened_mask.to(dtype=flattened.dtype).unsqueeze(
                -1
            )
            padding_mask = ~flattened_mask
            # PyTorch attention cannot consume a row with every key masked.
            # Expose one zero placeholder during the kernel call, then zero the
            # complete row below so an all-invalid block remains exactly null.
            all_invalid = ~flattened_mask.any(dim=-1)
            padding_mask = padding_mask.clone()
            padding_mask[:, 0] &= ~all_invalid
        else:
            padding_mask = None
        out = self.encoder(
            flattened,
            src_key_padding_mask=padding_mask,
        )
        if flattened_mask is not None:
            out = out * flattened_mask.to(dtype=out.dtype).unsqueeze(-1)
        return out.reshape(*leading_shape, tokens, channels)

    def pool(
        self,
        hidden: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.pool_score(self.norm(hidden)).squeeze(-1)
        if token_valid_mask is None:
            weights = scores.softmax(dim=-1)
        else:
            expected = tuple(hidden.shape[:-1])
            if tuple(token_valid_mask.shape) != expected:
                raise ValueError(
                    f"token_valid_mask must be {expected}, "
                    f"got {tuple(token_valid_mask.shape)}"
                )
            mask = token_valid_mask.to(device=hidden.device, dtype=torch.bool)
            safe_mask = mask.clone()
            all_invalid = ~mask.any(dim=-1)
            safe_mask[..., 0] |= all_invalid
            scores = scores.masked_fill(
                ~safe_mask,
                torch.finfo(scores.dtype).min,
            )
            weights = scores.softmax(dim=-1)
            weights = weights * mask.to(dtype=weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).eps
            )
        return (hidden * weights.unsqueeze(-1)).sum(dim=-2)
