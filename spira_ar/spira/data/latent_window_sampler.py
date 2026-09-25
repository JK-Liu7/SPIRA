from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import torch


def as_dhw_window(window: Sequence[int]) -> tuple[int, int, int]:
    if len(window) != 3:
        raise ValueError(f"window must contain 3 values, got {window!r}")
    return tuple((int(value) for value in window))


def _candidate_starts(length: int, window: int, steps: int = 5) -> list[int]:
    if length < window:
        raise ValueError(f"Length {length} is smaller than window {window}")
    if length == window:
        return [0]
    values = (
        torch.linspace(0, length - window, steps=max(2, steps)).round().long().tolist()
    )
    return sorted(set((int(value) for value in values)))


def find_best_valid_window(
    valid_mask: torch.Tensor, window: Sequence[int] = (24, 24, 24)
) -> tuple[int, int, int]:
    wd, wh, ww = as_dhw_window(window)
    d, h, w = valid_mask.shape
    best_ratio = -1.0
    best_start = (0, 0, 0)
    for d0 in _candidate_starts(d, wd):
        for h0 in _candidate_starts(h, wh):
            for w0 in _candidate_starts(w, ww):
                ratio = float(
                    valid_mask[d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww]
                    .float()
                    .mean()
                    .item()
                )
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = (d0, h0, w0)
    return best_start


def sample_valid_latent_window(
    valid_mask: torch.Tensor,
    window: Sequence[int] = (24, 24, 24),
    min_valid_ratio: float = 0.85,
    max_tries: int = 100,
    generator: torch.Generator | None = None,
) -> tuple[int, int, int]:
    if valid_mask.ndim != 3:
        raise ValueError(f"Expected valid_mask [D,H,W], got {tuple(valid_mask.shape)}")
    wd, wh, ww = as_dhw_window(window)
    d, h, w = valid_mask.shape
    if d < wd or h < wh or w < ww:
        raise ValueError(
            f"Latent shape {(d, h, w)} is smaller than requested window {(wd, wh, ww)}"
        )
    for _ in range(max_tries):
        d0 = int(torch.randint(0, d - wd + 1, (), generator=generator).item())
        h0 = int(torch.randint(0, h - wh + 1, (), generator=generator).item())
        w0 = int(torch.randint(0, w - ww + 1, (), generator=generator).item())
        ratio = float(
            valid_mask[d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww].float().mean().item()
        )
        if ratio >= min_valid_ratio:
            return (d0, h0, w0)
    return find_best_valid_window(valid_mask, window=window)


def sample_random_latent_window(
    valid_mask: torch.Tensor,
    window: Sequence[int] = (24, 24, 24),
    generator: torch.Generator | None = None,
) -> tuple[int, int, int]:
    """Sample a pure random latent crop start, matching RandSpatialCrop behavior."""
    if valid_mask.ndim != 3:
        raise ValueError(f"Expected valid_mask [D,H,W], got {tuple(valid_mask.shape)}")
    wd, wh, ww = as_dhw_window(window)
    d, h, w = valid_mask.shape
    if d < wd or h < wh or w < ww:
        raise ValueError(
            f"Latent shape {(d, h, w)} is smaller than requested window {(wd, wh, ww)}"
        )
    d0 = int(torch.randint(0, d - wd + 1, (), generator=generator).item())
    h0 = int(torch.randint(0, h - wh + 1, (), generator=generator).item())
    w0 = int(torch.randint(0, w - ww + 1, (), generator=generator).item())
    return (d0, h0, w0)


def crop_latent_window(
    cache: dict[str, Any],
    start_dhw: Sequence[int],
    window: Sequence[int] = (24, 24, 24),
):
    d0, h0, w0 = as_dhw_window(start_dhw)
    wd, wh, ww = as_dhw_window(window)
    out = {
        "source_fsq_scalars": cache["source_fsq_scalars"][
            :, d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww
        ],
        "target_fsq_scalars": cache["target_fsq_scalars"][
            :, d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww
        ],
        "valid_mask_latent": cache["valid_mask_latent"][
            d0 : d0 + wd, h0 : h0 + wh, w0 : w0 + ww
        ],
        "window_start_dhw": torch.as_tensor([d0, h0, w0], dtype=torch.long),
    }
    return out


def patchify_valid_mask(
    valid_mask: torch.Tensor, block_size_dhw: Sequence[int] = (4, 4, 4)
) -> torch.Tensor:
    bd, bh, bw = as_dhw_window(block_size_dhw)
    d, h, w = valid_mask.shape
    if d % bd or h % bh or w % bw:
        raise ValueError(
            f"valid_mask shape {(d, h, w)} must be divisible by block_size {(bd, bh, bw)}"
        )
    return (
        valid_mask.reshape(d // bd, bd, h // bh, bh, w // bw, bw)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(d // bd, h // bh, w // bw, bd * bh * bw)
    )
