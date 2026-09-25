from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from spira_ar.spira.models.kv_cache import AttentionKVCache, init_layer_kv_caches
from spira_ar.spira.models.rope3d import apply_rope_3d
from spira_ar.spira.utils.raster_order import raster_coords_tensor


def find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(
            torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + self.eps
        )
        return out.to(dtype=x.dtype) * self.weight


@dataclass
class InterBlockConfig:
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.05
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    multiple_of: int = 256
    block_grid_dhw: tuple[int, int, int] = (1, 1, 1)
    source_relative_position_bias: bool = True


class FeedForward(nn.Module):
    def __init__(self, cfg: InterBlockConfig) -> None:
        super().__init__()
        hidden = int(2 * (cfg.hidden_dim * cfg.mlp_ratio) / 3)
        hidden = find_multiple(hidden, cfg.multiple_of)
        self.w1 = nn.Linear(cfg.hidden_dim, hidden, bias=False)
        self.w3 = nn.Linear(cfg.hidden_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: InterBlockConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.rope_base = cfg.rope_base
        self.qkv = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 3, bias=False)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.dropout_p = cfg.dropout

    def _split(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        *,
        causal_mask: torch.Tensor | None = None,
        kv_cache: AttentionKVCache | None = None,
        key_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self._split(x)
        q = apply_rope_3d(q, coords, base=self.rope_base)
        k = apply_rope_3d(k, coords, base=self.rope_base)
        if kv_cache is not None:
            k, v = kv_cache.append(k, v)
            attn_mask = None
            is_causal = False
        else:
            attn_mask = (
                None
                if causal_mask is None
                else causal_mask.to(device=x.device).view(1, 1, t, t)
            )
            is_causal = causal_mask is None
        if key_valid_mask is not None:
            expected = (b, k.shape[-2])
            if tuple(key_valid_mask.shape) != expected:
                raise ValueError(
                    f"key_valid_mask must be {expected}, "
                    f"got {tuple(key_valid_mask.shape)}"
                )
            key_valid = key_valid_mask.to(device=x.device, dtype=torch.bool)
            key_allowed = key_valid[:, None, None, :]
            attn_mask = key_allowed if attn_mask is None else attn_mask & key_allowed
            is_causal = False
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.hidden_dim)
        return self.out(out)


class SourceCrossAttention(nn.Module):
    """Cross-attention to source block memory with factorized relative 3D bias."""

    def __init__(self, cfg: InterBlockConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.q = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.kv = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2, bias=False)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.dropout_p = cfg.dropout
        self.use_relative_position_bias = bool(cfg.source_relative_position_bias)

        gd, gh, gw = (int(v) for v in cfg.block_grid_dhw)
        if min(gd, gh, gw) < 1:
            raise ValueError(f"Invalid block_grid_dhw {cfg.block_grid_dhw}")
        self.block_grid_dhw = (gd, gh, gw)
        if self.use_relative_position_bias:
            # Factorized axis tables avoid a large dense 3D table while still
            # assigning every target-source block pair a learned 3D bias.
            self.rel_bias_d = nn.Embedding(2 * gd - 1, self.num_heads)
            self.rel_bias_h = nn.Embedding(2 * gh - 1, self.num_heads)
            self.rel_bias_w = nn.Embedding(2 * gw - 1, self.num_heads)
            nn.init.trunc_normal_(self.rel_bias_d.weight, std=0.02)
            nn.init.trunc_normal_(self.rel_bias_h.weight, std=0.02)
            nn.init.trunc_normal_(self.rel_bias_w.weight, std=0.02)
        else:
            self.rel_bias_d = None
            self.rel_bias_h = None
            self.rel_bias_w = None

    def build_source_kv(
        self, source_memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = source_memory.shape
        k, v = self.kv(source_memory).chunk(2, dim=-1)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        return k, v

    def relative_position_bias(
        self,
        query_coords: torch.Tensor,
        source_coords: torch.Tensor,
        *,
        dtype: torch.dtype,
        device,
    ) -> torch.Tensor | None:
        if not self.use_relative_position_bias:
            return None
        if query_coords.ndim != 2 or query_coords.shape[-1] != 3:
            raise ValueError(
                f"query_coords must be [T,3], got {tuple(query_coords.shape)}"
            )
        if source_coords.ndim != 2 or source_coords.shape[-1] != 3:
            raise ValueError(
                f"source_coords must be [N,3], got {tuple(source_coords.shape)}"
            )

        query = query_coords.to(device=device, dtype=torch.long)
        source = source_coords.to(device=device, dtype=torch.long)
        relative = query[:, None, :] - source[None, :, :]
        gd, gh, gw = self.block_grid_dhw
        d_index = (relative[..., 0] + gd - 1).clamp_(0, 2 * gd - 2)
        h_index = (relative[..., 1] + gh - 1).clamp_(0, 2 * gh - 2)
        w_index = (relative[..., 2] + gw - 1).clamp_(0, 2 * gw - 2)

        bias = (
            self.rel_bias_d(d_index)
            + self.rel_bias_h(h_index)
            + self.rel_bias_w(w_index)
        )
        # [T,N,H] -> [1,H,T,N], additive mask for SDPA.
        return bias.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        source_memory: torch.Tensor,
        query_coords: torch.Tensor,
        source_coords: torch.Tensor,
        source_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        if query_coords.shape[0] != t:
            raise ValueError(
                f"Expected {t} query coordinates, got {query_coords.shape[0]}"
            )
        if source_coords.shape[0] != source_memory.shape[1]:
            raise ValueError(
                f"Expected {source_memory.shape[1]} source coordinates, got {source_coords.shape[0]}"
            )

        q = self.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = (
            source_kv if source_kv is not None else self.build_source_kv(source_memory)
        )
        attn_bias = self.relative_position_bias(
            query_coords,
            source_coords,
            dtype=q.dtype,
            device=q.device,
        )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            is_causal=False,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.hidden_dim)
        return self.out(out)


class InterBlockLayer(nn.Module):
    def __init__(self, cfg: InterBlockConfig) -> None:
        super().__init__()
        self.self_norm = RMSNorm(cfg.hidden_dim, eps=cfg.norm_eps)
        self.self_attn = CausalSelfAttention(cfg)
        self.cross_norm = RMSNorm(cfg.hidden_dim, eps=cfg.norm_eps)
        self.cross_attn = SourceCrossAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_dim, eps=cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        source_memory: torch.Tensor,
        source_coords: torch.Tensor,
        *,
        source_query_coords: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        self_kv_cache: AttentionKVCache | None = None,
        source_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        self_key_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_query_coords is None:
            source_query_coords = coords
        x = x + self.self_attn(
            self.self_norm(x),
            coords,
            causal_mask=causal_mask,
            kv_cache=self_kv_cache,
            key_valid_mask=self_key_valid_mask,
        )
        x = x + self.cross_attn(
            self.cross_norm(x),
            source_memory,
            query_coords=source_query_coords,
            source_coords=source_coords,
            source_kv=source_kv,
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


class InterBlockCausalLlama(nn.Module):
    def __init__(self, cfg: InterBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [InterBlockLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_dim, eps=cfg.norm_eps)
        self.register_buffer(
            "source_block_coords",
            raster_coords_tensor(tuple(int(v) for v in cfg.block_grid_dhw)),
            persistent=False,
        )

    @property
    def head_dim(self) -> int:
        return self.cfg.hidden_dim // self.cfg.num_heads

    def causal_mask(self, seq_len: int, device) -> torch.Tensor:
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()

    def _source_coords(
        self, source_memory: torch.Tensor, source_coords: torch.Tensor | None
    ) -> torch.Tensor:
        coords = self.source_block_coords if source_coords is None else source_coords
        if coords.shape != (source_memory.shape[1], 3):
            raise ValueError(
                f"Expected source coordinates {(source_memory.shape[1], 3)}, got {tuple(coords.shape)}"
            )
        return coords.to(device=source_memory.device)

    def forward(
        self,
        summaries: torch.Tensor,
        source_memory: torch.Tensor,
        coords: torch.Tensor,
        source_coords: torch.Tensor | None = None,
        source_query_coords: torch.Tensor | None = None,
        valid_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = self.causal_mask(summaries.shape[1], summaries.device)
        x = summaries
        coords = coords.to(device=summaries.device)
        if valid_block_mask is not None:
            expected = summaries.shape[:2]
            if tuple(valid_block_mask.shape) != tuple(expected):
                raise ValueError(
                    f"valid_block_mask must be {tuple(expected)}, "
                    f"got {tuple(valid_block_mask.shape)}"
                )
            valid_block_mask = valid_block_mask.to(
                device=summaries.device,
                dtype=torch.bool,
            )
        if source_query_coords is None:
            source_query_coords = coords
        else:
            source_query_coords = source_query_coords.to(device=summaries.device)
            if source_query_coords.shape != coords.shape:
                raise ValueError(
                    f"Expected source_query_coords shape {tuple(coords.shape)}, "
                    f"got {tuple(source_query_coords.shape)}"
                )
        source_coords = self._source_coords(source_memory, source_coords)
        for layer in self.layers:
            layer_kwargs = {
                "source_query_coords": source_query_coords,
                "causal_mask": mask,
            }
            if valid_block_mask is not None:
                layer_kwargs["self_key_valid_mask"] = valid_block_mask
            x = layer(
                x,
                coords,
                source_memory,
                source_coords,
                **layer_kwargs,
            )
        return self.norm(x)

    def init_kv_cache(
        self, batch_size: int, max_seq_len: int, *, device, dtype
    ) -> list[AttentionKVCache]:
        return init_layer_kv_caches(
            len(self.layers),
            batch_size,
            max_seq_len,
            self.cfg.num_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    def build_source_kv_cache(
        self, source_memory: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            layer.cross_attn.build_source_kv(source_memory) for layer in self.layers
        ]

    def forward_step(
        self,
        summary: torch.Tensor,
        step_coord: torch.Tensor,
        source_memory: torch.Tensor,
        *,
        source_coords: torch.Tensor | None = None,
        source_query_coord: torch.Tensor | None = None,
        self_kv_cache: list[AttentionKVCache] | None = None,
        source_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        history_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = summary[:, None, :]
        coords = step_coord.view(1, 3).to(device=summary.device)
        if history_valid_mask is not None:
            if (
                history_valid_mask.ndim != 2
                or history_valid_mask.shape[0] != summary.shape[0]
            ):
                raise ValueError(
                    "history_valid_mask must be [B,T], got "
                    f"{tuple(history_valid_mask.shape)}"
                )
            history_valid_mask = history_valid_mask.to(
                device=summary.device,
                dtype=torch.bool,
            )
        if source_query_coord is None:
            source_query_coords = coords
        else:
            source_query_coords = source_query_coord.view(1, 3).to(
                device=summary.device
            )
        source_coords = self._source_coords(source_memory, source_coords)
        for idx, layer in enumerate(self.layers):
            layer_kwargs = {
                "source_query_coords": source_query_coords,
                "self_kv_cache": (
                    None if self_kv_cache is None else self_kv_cache[idx]
                ),
                "source_kv": (
                    None if source_kv_cache is None else source_kv_cache[idx]
                ),
            }
            if history_valid_mask is not None:
                layer_kwargs["self_key_valid_mask"] = history_valid_mask
            x = layer(
                x,
                coords,
                source_memory,
                source_coords,
                **layer_kwargs,
            )
        return self.norm(x)[:, 0]
