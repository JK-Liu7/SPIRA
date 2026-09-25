from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


PATCH_SIZE_HWD = (192, 192, 16)
STRIDE_HWD = (144, 144, 12)
COMPRESSION_HWD = (8, 8, 4)
LATENT_TILE_HWD = (24, 24, 4)
LATENT_STRIDE_HWD = (18, 18, 3)


def as_3tuple(
    values: Sequence[int] | Sequence[float], name: str
) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain 3 values, got {values!r}")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def stride_from_overlap(
    patch_hwd: Sequence[int], overlap: float
) -> tuple[int, int, int]:
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0,1), got {overlap}")
    return tuple(int(round(int(dim) * (1.0 - overlap))) for dim in patch_hwd)  # type: ignore[return-value]


def coverage_padded_size(n: int, patch: int, stride: int, comp: int) -> int:
    """Pad to patch + k * stride and keep the result compression-divisible."""

    if patch % comp != 0 or stride % comp != 0:
        raise ValueError(
            f"patch={patch} and stride={stride} must be divisible by comp={comp}"
        )
    if n <= patch:
        out = patch
    else:
        out = patch + math.ceil((n - patch) / stride) * stride
    out = math.ceil(out / comp) * comp
    return int(out)


def compute_padded_shape_hwd(
    shape_hwd: Sequence[int],
    dataset: str,
    patch_hwd: Sequence[int] = PATCH_SIZE_HWD,
    stride_hwd: Sequence[int] = STRIDE_HWD,
    comp_hwd: Sequence[int] = COMPRESSION_HWD,
    pad_mode: str = "auto",
    min_latent_shape_dhw: Sequence[int] | None = None,
) -> tuple[int, int, int]:
    shape = as_3tuple(shape_hwd, "shape_hwd")
    patch = as_3tuple(patch_hwd, "patch_hwd")
    stride = as_3tuple(stride_hwd, "stride_hwd")
    comp = as_3tuple(comp_hwd, "comp_hwd")
    mode = pad_mode.lower()
    dataset_key = dataset.lower().replace("-", "").replace("_", "")
    no_pad = mode == "no_pad" or (
        mode == "auto" and dataset_key in {"brats", "brats24"}
    )
    if no_pad:
        out = shape
    else:
        out = tuple(
            coverage_padded_size(n, p, s, c)
            for n, p, s, c in zip(shape, patch, stride, comp)
        )
    if min_latent_shape_dhw is not None:
        min_dhw = as_3tuple(min_latent_shape_dhw, "min_latent_shape_dhw")
        min_hwd = (min_dhw[1] * comp[0], min_dhw[2] * comp[1], min_dhw[0] * comp[2])
        out = tuple(max(current, minimum) for current, minimum in zip(out, min_hwd))
    return out  # type: ignore[return-value]


def compute_starts(
    n: int,
    patch: int,
    stride: int,
    require_divisible_by: int,
    include_last: bool = True,
) -> list[int]:
    if n < patch:
        raise ValueError(
            f"Dimension {n} is smaller than patch {patch}; pad before building patches."
        )
    starts = list(range(0, n - patch + 1, stride))
    if include_last:
        last = n - patch
        if not starts or starts[-1] != last:
            starts.append(last)
    starts = sorted(set(starts))
    bad = [start for start in starts if start % require_divisible_by != 0]
    if bad:
        raise ValueError(
            f"Patch starts must map exactly to latent grid. n={n}, patch={patch}, "
            f"stride={stride}, comp={require_divisible_by}, bad={bad}"
        )
    return starts


def build_patch_index_table(
    shape_hwd: Sequence[int],
    patch_hwd: Sequence[int] = PATCH_SIZE_HWD,
    stride_hwd: Sequence[int] = STRIDE_HWD,
    comp_hwd: Sequence[int] = COMPRESSION_HWD,
    include_last: bool = True,
) -> tuple[list[dict[str, int]], tuple[list[int], list[int], list[int]]]:
    shape = as_3tuple(shape_hwd, "shape_hwd")
    patch = as_3tuple(patch_hwd, "patch_hwd")
    stride = as_3tuple(stride_hwd, "stride_hwd")
    comp = as_3tuple(comp_hwd, "comp_hwd")
    hs = compute_starts(
        shape[0], patch[0], stride[0], comp[0], include_last=include_last
    )
    ws = compute_starts(
        shape[1], patch[1], stride[1], comp[1], include_last=include_last
    )
    ds = compute_starts(
        shape[2], patch[2], stride[2], comp[2], include_last=include_last
    )
    table: list[dict[str, int]] = []
    for ih, h0 in enumerate(hs):
        for iw, w0 in enumerate(ws):
            for id_, d0 in enumerate(ds):
                table.append(
                    {
                        "patch_id": len(table),
                        "h0": h0,
                        "h1": h0 + patch[0],
                        "w0": w0,
                        "w1": w0 + patch[1],
                        "d0": d0,
                        "d1": d0 + patch[2],
                        "ih": ih,
                        "iw": iw,
                        "id": id_,
                        "num_h": len(hs),
                        "num_w": len(ws),
                        "num_d": len(ds),
                    }
                )
    return table, (hs, ws, ds)


def pad_chwd_to_shape(x: Any, padded_shape_hwd: Sequence[int], value: float = -1.0):
    """Pad [C,H,W,D] on the right side to padded_shape_hwd."""

    import torch.nn.functional as F

    if x.ndim != 4:
        raise ValueError(f"Expected [C,H,W,D], got {tuple(x.shape)}")
    _, h, w, d = x.shape
    hp, wp, dp = as_3tuple(padded_shape_hwd, "padded_shape_hwd")
    if h > hp or w > wp or d > dp:
        raise ValueError(f"Cannot pad shape {(h, w, d)} down to {(hp, wp, dp)}")
    return F.pad(x, (0, dp - d, 0, wp - w, 0, hp - h), mode="constant", value=value)


def crop_chwd(x: Any, patch_meta: dict[str, int]):
    if x.ndim != 4:
        raise ValueError(f"Expected [C,H,W,D], got {tuple(x.shape)}")
    return x[
        :,
        patch_meta["h0"] : patch_meta["h1"],
        patch_meta["w0"] : patch_meta["w1"],
        patch_meta["d0"] : patch_meta["d1"],
    ].contiguous()


def padding_info(
    original_shape_hwd: Sequence[int], padded_shape_hwd: Sequence[int]
) -> dict[str, Any]:
    original = as_3tuple(original_shape_hwd, "original_shape_hwd")
    padded = as_3tuple(padded_shape_hwd, "padded_shape_hwd")
    return {
        "pad_width_hwd": [(0, int(p - o)) for o, p in zip(original, padded)],
        "pad_value": -1.0,
        "pad_side": "end",
    }
