from __future__ import annotations

from collections.abc import Sequence

from .grid_patch_with_coords import COMPRESSION_HWD, as_3tuple


def keep_bounds_1d(
    start_l: int, tile_l: int, full_l: int, starts_l: Sequence[int], idx: int
):
    """Nearest-center ownership bounds in latent coordinates."""

    center = start_l + tile_l / 2.0
    if idx == 0:
        left = 0
    else:
        prev_center = starts_l[idx - 1] + tile_l / 2.0
        left = int(round((prev_center + center) / 2.0))

    if idx == len(starts_l) - 1:
        right = full_l
    else:
        next_center = starts_l[idx + 1] + tile_l / 2.0
        right = int(round((center + next_center) / 2.0))

    src_a = max(0, left - start_l)
    src_b = min(tile_l, right - start_l)
    dst_a = start_l + src_a
    dst_b = start_l + src_b
    if src_b <= src_a or dst_b <= dst_a:
        raise ValueError(
            f"Invalid ownership slice: start_l={start_l}, tile_l={tile_l}, "
            f"full_l={full_l}, idx={idx}, starts_l={list(starts_l)}"
        )
    return slice(src_a, src_b), slice(dst_a, dst_b)


def latent_shape_dhw_from_hwd(
    padded_shape_hwd: Sequence[int],
    comp_hwd: Sequence[int] = COMPRESSION_HWD,
) -> tuple[int, int, int]:
    h, w, d = as_3tuple(padded_shape_hwd, "padded_shape_hwd")
    ch, cw, cd = as_3tuple(comp_hwd, "comp_hwd")
    if h % ch or w % cw or d % cd:
        raise ValueError(
            f"padded_shape_hwd={(h, w, d)} is not divisible by comp_hwd={(ch, cw, cd)}"
        )
    return (d // cd, h // ch, w // cw)


def stitch_one_patch(
    full_scalars,
    filled_mask,
    patch_scalars_dhw,
    patch_meta: dict[str, int],
    starts_hwd: tuple[Sequence[int], Sequence[int], Sequence[int]],
    latent_shape_dhw: Sequence[int],
    comp_hwd: Sequence[int] = COMPRESSION_HWD,
) -> None:
    comp = as_3tuple(comp_hwd, "comp_hwd")
    hs, ws, ds = starts_hwd
    starts_h_l = [int(start) // comp[0] for start in hs]
    starts_w_l = [int(start) // comp[1] for start in ws]
    starts_d_l = [int(start) // comp[2] for start in ds]

    ih, iw, id_ = patch_meta["ih"], patch_meta["iw"], patch_meta["id"]
    h0_l = patch_meta["h0"] // comp[0]
    w0_l = patch_meta["w0"] // comp[1]
    d0_l = patch_meta["d0"] // comp[2]
    full_d, full_h, full_w = as_3tuple(latent_shape_dhw, "latent_shape_dhw")

    _, tile_d, tile_h, tile_w = patch_scalars_dhw.shape
    src_h, dst_h = keep_bounds_1d(h0_l, tile_h, full_h, starts_h_l, ih)
    src_w, dst_w = keep_bounds_1d(w0_l, tile_w, full_w, starts_w_l, iw)
    src_d, dst_d = keep_bounds_1d(d0_l, tile_d, full_d, starts_d_l, id_)

    full_scalars[:, dst_d, dst_h, dst_w] = patch_scalars_dhw[:, src_d, src_h, src_w]
    filled_mask[dst_d, dst_h, dst_w] += 1


def stitch_all_patches(
    encoded_patches,
    patch_table: Sequence[dict[str, int]],
    starts_hwd: tuple[Sequence[int], Sequence[int], Sequence[int]],
    padded_shape_hwd: Sequence[int],
    levels: Sequence[int],
    comp_hwd: Sequence[int] = COMPRESSION_HWD,
):
    import torch

    latent_shape_dhw = latent_shape_dhw_from_hwd(padded_shape_hwd, comp_hwd=comp_hwd)
    full = torch.full(
        (len(levels), *latent_shape_dhw), fill_value=-1, dtype=torch.int16
    )
    filled = torch.zeros(latent_shape_dhw, dtype=torch.int16)

    if len(encoded_patches) != len(patch_table):
        raise ValueError(
            f"Expected {len(patch_table)} encoded patches, got {len(encoded_patches)}"
        )
    for patch_scalars_dhw, meta in zip(encoded_patches, patch_table):
        stitch_one_patch(
            full,
            filled,
            patch_scalars_dhw,
            meta,
            starts_hwd,
            latent_shape_dhw,
            comp_hwd=comp_hwd,
        )

    if not bool((filled > 0).all().item()):
        missing = int((filled == 0).sum().item())
        raise RuntimeError(
            f"{missing} latent cells were never filled. Check patch starts and ownership bounds."
        )
    if not bool((filled == 1).all().item()):
        duplicate = int((filled > 1).sum().item())
        raise RuntimeError(
            f"{duplicate} latent cells were written multiple times. Check ownership bounds."
        )
    if not bool((full >= 0).all().item()):
        raise RuntimeError("Invalid -1 scalar id remains after stitching.")
    return full, filled
