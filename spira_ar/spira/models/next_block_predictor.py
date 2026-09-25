from __future__ import annotations
import math
import torch
import torch.nn as nn


class NextBlockPredictor(nn.Module):
    """Build token-correspondent contexts for the next target block.

    For current block ``g`` and next block ``g+1``, the predictor implements

    ``q[g+1,v] = alpha W_h h[g,v] + W_c c[g] + W_m m[g,v]``
    ``              + W_s s[g+1,v] + e_v + e_delta``

    where ``h[g,v]`` is the current target block's bidirectional token hidden,
    ``c[g]`` is the causal inter-block context, ``s[g+1,v]`` is the co-located
    contextual source token, ``m[g,v]`` is the optional dense-recent / compressed-
    old target-history context, and ``e_delta`` encodes the 3D transition from
    the current block to the next block. A small block-local bidirectional
    decoder then lets the ``V`` next-block queries interact before the FSQ head.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        tokens_per_block: int = 64,
        *,
        local_num_layers: int = 1,
        local_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        current_token_gate_enabled: bool = True,
        current_token_gate_learnable: bool = False,
        current_token_gate_init: float = 1.0,
        current_token_gate_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if int(local_num_layers) < 0:
            raise ValueError("local_num_layers must be non-negative")
        if int(local_num_layers) > 0 and hidden_dim % int(local_num_heads):
            raise ValueError("hidden_dim must be divisible by local_num_heads")
        self.hidden_dim = int(hidden_dim)
        self.tokens_per_block = int(tokens_per_block)
        self.current_token_gate_enabled = bool(current_token_gate_enabled)
        self.current_token_gate_learnable = (
            bool(current_token_gate_learnable) and self.current_token_gate_enabled
        )
        self.current_token_gate_activation = str(current_token_gate_activation).lower()
        if self.current_token_gate_activation != "sigmoid":
            raise ValueError(
                "current_token_gate.activation currently supports only 'sigmoid'"
            )
        gate_init = float(current_token_gate_init)
        if not 0.0 <= gate_init <= 1.0:
            raise ValueError("current_token_gate.init must be in [0, 1]")
        self.current_token_gate_fixed = (
            gate_init if self.current_token_gate_enabled else 0.0
        )
        if self.current_token_gate_learnable:
            if not 0.0 < gate_init < 1.0:
                raise ValueError(
                    "A learnable sigmoid current-token gate requires init strictly between 0 and 1"
                )
            gate_logit = math.log(gate_init / (1.0 - gate_init))
            self.current_token_gate_logit = nn.Parameter(
                torch.tensor(gate_logit, dtype=torch.float32)
            )
        else:
            self.register_parameter("current_token_gate_logit", None)
        self.current_token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        self.block_context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        self.source_token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        self.source_block_fallback = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        self.transition_embed = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.intra_pos = nn.Parameter(torch.zeros(tokens_per_block, hidden_dim))
        if int(local_num_layers) > 0:
            local_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(local_num_heads),
                dim_feedforward=max(int(hidden_dim * mlp_ratio), hidden_dim),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.local_decoder: nn.Module = nn.TransformerEncoder(
                local_layer, num_layers=int(local_num_layers)
            )
        else:
            self.local_decoder = nn.Identity()
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        nn.init.normal_(self.intra_pos, std=0.02)

    def current_token_gate_value(self) -> torch.Tensor:
        if self.current_token_gate_logit is not None:
            return self.current_token_gate_logit.sigmoid()
        return self.intra_pos.new_tensor(self.current_token_gate_fixed)

    def forward(
        self,
        block_context: torch.Tensor,
        next_coords: torch.Tensor,
        *,
        current_block_hidden: torch.Tensor,
        current_coords: torch.Tensor,
        source_memory: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
        next_block_indices: torch.Tensor | None = None,
        coord_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if block_context.ndim == 2:
            block_context = block_context[:, None, :]
        if current_block_hidden.ndim == 3:
            current_block_hidden = current_block_hidden[:, None, :, :]
        if block_context.ndim != 3:
            raise ValueError(
                f"Expected block_context [B,N,C] or [B,C], got {tuple(block_context.shape)}"
            )
        if current_block_hidden.ndim != 4:
            raise ValueError(
                f"Expected current_block_hidden [B,N,V,C] or [B,V,C], got {tuple(current_block_hidden.shape)}"
            )
        b, n, c = block_context.shape
        if c != self.hidden_dim:
            raise ValueError(f"Expected hidden dim {self.hidden_dim}, got {c}")
        if current_block_hidden.shape != (b, n, self.tokens_per_block, c):
            raise ValueError(
                f"current_block_hidden must align with block_context and tokens_per_block: expected {(b, n, self.tokens_per_block, c)}, got {tuple(current_block_hidden.shape)}"
            )
        if next_coords.ndim != 2 or next_coords.shape != (n, 3):
            raise ValueError(
                f"Expected next_coords [{n},3], got {tuple(next_coords.shape)}"
            )
        if current_coords.ndim != 2 or current_coords.shape != (n, 3):
            raise ValueError(
                f"Expected current_coords [{n},3], got {tuple(current_coords.shape)}"
            )
        next_coords_f = next_coords.to(block_context.device, block_context.dtype)
        current_coords_f = current_coords.to(block_context.device, block_context.dtype)
        if coord_max is None:
            coord_max_f = torch.maximum(
                next_coords_f.abs().max(dim=0).values,
                current_coords_f.abs().max(dim=0).values,
            )
        else:
            coord_max_f = coord_max.to(block_context.device, block_context.dtype)
        transition = (next_coords_f - current_coords_f) / coord_max_f.clamp_min(1.0)
        transition = self.transition_embed(transition).view(1, n, 1, c)
        x = self.current_token_gate_value().to(
            dtype=current_block_hidden.dtype
        ) * self.current_token_proj(current_block_hidden)
        x = x + self.block_context_proj(block_context)[:, :, None, :]
        x = x + self.intra_pos.view(1, 1, self.tokens_per_block, c)
        x = x + transition
        if source_tokens is not None:
            if next_block_indices is None:
                raise ValueError(
                    "next_block_indices is required when source_tokens is provided"
                )
            vox = source_tokens.index_select(
                1, next_block_indices.to(source_tokens.device).long()
            )
            if vox.shape != (b, n, self.tokens_per_block, c):
                raise ValueError(
                    f"Selected source tokens must have shape {(b, n, self.tokens_per_block, c)}, got {tuple(vox.shape)}"
                )
            x = x + self.source_token_proj(vox)
        elif source_memory is not None:
            if next_block_indices is None:
                raise ValueError(
                    "next_block_indices is required when source_memory is provided"
                )
            local = source_memory.index_select(
                1, next_block_indices.to(source_memory.device).long()
            )
            x = x + self.source_block_fallback(local)[:, :, None, :]
        x = self.local_decoder(x.reshape(b * n, self.tokens_per_block, c))
        x = x.reshape(b, n, self.tokens_per_block, c)
        return self.out(x)
