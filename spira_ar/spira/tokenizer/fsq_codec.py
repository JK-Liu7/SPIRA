from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _torch():
    import torch

    return torch


def get_fsq_levels(regularizer: Any) -> list[int]:
    """Read FSQ scalar levels from a VidTok regularizer without hard-coding them."""

    for name in ("_levels", "levels"):
        value = getattr(regularizer, name, None)
        if value is None:
            continue
        if hasattr(value, "detach"):
            return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, Sequence):
            return [int(item) for item in value]
    raise AttributeError("Could not find FSQ levels on VidTok regularizer.")


def fsq_basis(levels: Sequence[int], device: Any | None = None):
    torch = _torch()
    values = [1]
    for level in levels[:-1]:
        values.append(values[-1] * int(level))
    return torch.as_tensor(values, dtype=torch.long, device=device)


def unpack_fsq_indices_to_scalars(
    indices: Any, levels: Sequence[int], scalar_dim: int = 1
):
    """Unpack VidTok FSQ packed indices into per-scalar ids.

    VidTok's FSQRegularizer packs indices with a basis of
    ``cumprod([1] + levels[:-1])``. This must match the implementation in
    ``regularizers.py::indices_to_codes``.
    """

    torch = _torch()
    x = indices.long()
    basis = fsq_basis(levels, device=x.device)
    scalars = [(x // basis_i) % int(level) for basis_i, level in zip(basis, levels)]
    return torch.stack(scalars, dim=scalar_dim)


def pack_fsq_scalars_to_indices(
    scalars: Any, levels: Sequence[int], scalar_dim: int = 0
):
    """Pack per-scalar FSQ ids back into VidTok's 262144-way index."""

    torch = _torch()
    x = scalars.long()
    if scalar_dim != 0:
        x = x.movedim(scalar_dim, 0)
    if x.shape[0] != len(levels):
        raise ValueError(
            f"Expected scalar dimension of {len(levels)}, got shape {tuple(x.shape)}"
        )
    basis = fsq_basis(levels, device=x.device).view(-1, *([1] * (x.ndim - 1)))
    return (x * basis).sum(dim=0).to(torch.int32)


def canonicalize_vidtok_scalars_to_dhw(scalars: Any, levels: Sequence[int]):
    """Convert common VidTok scalar layouts to [d, D_l, H_l, W_l].

    Accepted layouts include [1,d,D,H,W], [d,D,H,W], [1,D,H,W,d],
    and [D,H,W,d].
    """

    d = len(levels)
    x = scalars
    if x.ndim == 5:
        if x.shape[0] != 1:
            raise ValueError(
                f"Only batch size 1 is supported for latent caching, got {tuple(x.shape)}"
            )
        x = x[0]

    if x.ndim != 4:
        raise ValueError(
            f"Expected a 4D scalar tensor after squeezing batch, got {tuple(x.shape)}"
        )

    if x.shape[0] == d:
        return x.contiguous()
    if x.shape[-1] == d:
        return x.permute(3, 0, 1, 2).contiguous()
    raise ValueError(
        f"Could not infer scalar dimension for shape {tuple(x.shape)} and levels={list(levels)}"
    )


def validate_scalar_ranges(
    scalars: Any, levels: Sequence[int], name: str = "fsq_scalars"
) -> None:
    if scalars.ndim < 1 or scalars.shape[0] != len(levels):
        raise ValueError(
            f"{name} must be [d,...] with d={len(levels)}, got {tuple(scalars.shape)}"
        )
    for idx, level in enumerate(levels):
        values = scalars[idx]
        min_value = int(values.min().item())
        max_value = int(values.max().item())
        if min_value < 0 or max_value >= int(level):
            raise ValueError(
                f"{name}[{idx}] out of range for level {level}: min={min_value}, max={max_value}"
            )
