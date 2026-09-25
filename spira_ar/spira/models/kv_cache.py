from __future__ import annotations

import torch


class AttentionKVCache:
    """Simple per-layer cache for fixed-order block decoding."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        *,
        device,
        dtype,
    ) -> None:
        shape = (int(max_batch_size), int(num_heads), int(max_seq_len), int(head_dim))
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.cur_len = 0

    def reset(self) -> None:
        self.cur_len = 0

    def append(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, t, d = k.shape
        end = self.cur_len + t
        if end > self.k.shape[2]:
            raise ValueError(
                f"KV cache overflow: requested {end}, capacity {self.k.shape[2]}"
            )
        self.k[:b, :h, self.cur_len : end, :d] = k
        self.v[:b, :h, self.cur_len : end, :d] = v
        self.cur_len = end
        return self.k[:b, :h, :end, :d], self.v[:b, :h, :end, :d]


def init_layer_kv_caches(
    num_layers: int,
    max_batch_size: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    device,
    dtype,
) -> list[AttentionKVCache]:
    return [
        AttentionKVCache(
            max_batch_size, max_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        for _ in range(int(num_layers))
    ]
