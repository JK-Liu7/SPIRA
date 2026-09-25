from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _split_even(total: int, parts: int = 3) -> list[int]:
    if total % 2:
        total -= 1
    base = total // parts
    sizes = [base for _ in range(parts)]
    for i in range(total - base * parts):
        sizes[i] += 1
    sizes = [s - (s % 2) for s in sizes]
    missing = total - sum(sizes)
    idx = 0
    while missing > 0:
        sizes[idx % parts] += 2
        missing -= 2
        idx += 1
    return sizes


def rope_cos_sin_3d(
    coords: torch.Tensor, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables for coords ``[T,3]`` and one attention head."""

    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be [T,3], got {tuple(coords.shape)}")
    sizes = _split_even(head_dim, 3)
    parts_cos = []
    parts_sin = []
    coords_f = coords.float()
    for axis, dim in enumerate(sizes):
        if dim == 0:
            continue
        inv = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, device=coords.device).float() / max(dim, 1))
        )
        freqs = coords_f[:, axis : axis + 1] * inv[None, :]
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        parts_cos.append(freqs.cos())
        parts_sin.append(freqs.sin())
    cos = torch.cat(parts_cos, dim=-1)
    sin = torch.cat(parts_sin, dim=-1)
    if cos.shape[-1] < head_dim:
        pad = head_dim - cos.shape[-1]
        cos = torch.cat(
            [cos, torch.ones(cos.shape[0], pad, device=coords.device)], dim=-1
        )
        sin = torch.cat(
            [sin, torch.zeros(sin.shape[0], pad, device=coords.device)], dim=-1
        )
    return cos, sin


def apply_rope_3d(
    x: torch.Tensor, coords: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    """Apply 3D rotary embedding to ``[B,H,T,D]`` attention tensors."""

    if x.ndim != 4:
        raise ValueError(f"Expected [B,H,T,D], got {tuple(x.shape)}")
    cos, sin = rope_cos_sin_3d(coords.to(x.device), x.shape[-1], base=base)
    cos = cos.to(dtype=x.dtype).view(1, 1, x.shape[-2], x.shape[-1])
    sin = sin.to(dtype=x.dtype).view(1, 1, x.shape[-2], x.shape[-1])
    return x * cos + rotate_half(x) * sin
