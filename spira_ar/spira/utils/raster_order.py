from __future__ import annotations

from collections.abc import Sequence

import torch


def as_3tuple(values: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain 3 values, got {values!r}")
    return tuple(int(v) for v in values)  # type: ignore[return-value]


def raster_order_3d(
    grid_d: int = 6, grid_h: int = 6, grid_w: int = 6
) -> list[tuple[int, int, int]]:
    return [
        (d, h, w) for d in range(grid_d) for h in range(grid_h) for w in range(grid_w)
    ]


def raster_coords_tensor(
    block_grid_dhw: Sequence[int] = (6, 6, 6),
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    gd, gh, gw = as_3tuple(block_grid_dhw, "block_grid_dhw")
    return torch.as_tensor(raster_order_3d(gd, gh, gw), dtype=dtype, device=device)


def intra_block_coords_tensor(
    block_size_dhw: Sequence[int] = (4, 4, 4),
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    bd, bh, bw = as_3tuple(block_size_dhw, "block_size_dhw")
    coords = [(d, h, w) for d in range(bd) for h in range(bh) for w in range(bw)]
    return torch.as_tensor(coords, dtype=dtype, device=device)


def modality_to_id(modality: str | None, num_buckets: int = 64) -> int:
    text = (modality or "").strip().lower()
    if not text:
        return 0
    value = 0
    for idx, byte in enumerate(text.encode("utf-8")):
        value = (value + (idx + 1) * int(byte)) % int(num_buckets)
    return int(value)


def modalities_to_ids(
    modalities: Sequence[str] | torch.Tensor | str | None, num_buckets: int = 64
) -> torch.Tensor:
    if modalities is None:
        return torch.zeros(1, dtype=torch.long)
    if isinstance(modalities, torch.Tensor):
        return modalities.long()
    if isinstance(modalities, str):
        return torch.tensor([modality_to_id(modalities, num_buckets)], dtype=torch.long)
    return torch.tensor(
        [modality_to_id(item, num_buckets) for item in modalities], dtype=torch.long
    )
