from __future__ import annotations
import torch


def _sliding_starts(length: int, window: int, stride: int) -> list[int]:
    if length < window:
        raise ValueError(
            f"Latent dimension {length} is smaller than inference window {window}."
        )
    if length == window:
        return [0]
    starts = list(range(0, length - window + 1, max(int(stride), 1)))
    last = length - window
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set((int(value) for value in starts)))


def _ownership_slices(
    starts: list[int], idx: int, window: int, length: int
) -> tuple[slice, slice]:
    start = int(starts[idx])
    center = start + window / 2.0
    if idx == 0:
        left = 0
    else:
        prev_center = int(starts[idx - 1]) + window / 2.0
        left = int(round((prev_center + center) / 2.0))
    if idx == len(starts) - 1:
        right = length
    else:
        next_center = int(starts[idx + 1]) + window / 2.0
        right = int(round((center + next_center) / 2.0))
    src_a = max(0, left - start)
    src_b = min(window, right - start)
    dst_a = start + src_a
    dst_b = start + src_b
    if src_b <= src_a or dst_b <= dst_a:
        raise ValueError(
            f"Invalid ownership slice for start={start}, window={window}, length={length}."
        )
    return (slice(src_a, src_b), slice(dst_a, dst_b))


def _latent_window_metas(
    spatial_shape: tuple[int, int, int],
    window: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> list[dict[str, tuple[int, int, int] | tuple[slice, slice, slice]]]:
    starts_d = _sliding_starts(spatial_shape[0], window[0], stride[0])
    starts_h = _sliding_starts(spatial_shape[1], window[1], stride[1])
    starts_w = _sliding_starts(spatial_shape[2], window[2], stride[2])
    metas = []
    for id_, d0 in enumerate(starts_d):
        src_d, dst_d = _ownership_slices(starts_d, id_, window[0], spatial_shape[0])
        for ih, h0 in enumerate(starts_h):
            src_h, dst_h = _ownership_slices(starts_h, ih, window[1], spatial_shape[1])
            for iw, w0 in enumerate(starts_w):
                src_w, dst_w = _ownership_slices(
                    starts_w, iw, window[2], spatial_shape[2]
                )
                metas.append(
                    {
                        "start": (d0, h0, w0),
                        "src_slices": (src_d, src_h, src_w),
                        "dst_slices": (dst_d, dst_h, dst_w),
                    }
                )
    return metas


def _image_blend_weight(
    spatial_shape: tuple[int, int, int], mode: str = "gaussian"
) -> torch.Tensor:
    mode = str(mode).lower()
    if mode not in {"gaussian", "constant"}:
        mode = "gaussian"
    if mode == "constant":
        return torch.ones(spatial_shape, dtype=torch.float32)
    axes = []
    for size in spatial_shape:
        if size <= 1:
            axes.append(torch.ones(size, dtype=torch.float32))
            continue
        coord = torch.linspace(-1.0, 1.0, int(size), dtype=torch.float32)
        sigma = 0.125
        weight = torch.exp(-0.5 * (coord / sigma) ** 2)
        weight = weight / weight.max().clamp_min(1e-06)
        axes.append(weight.clamp_min(0.001))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
