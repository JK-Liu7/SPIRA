from __future__ import annotations

from collections.abc import Sequence

import torch

from spira_ar.spira.utils.raster_order import as_3tuple, raster_coords_tensor


class BlockPatchify3D:
    """Patchify FSQ scalar grids in global ``[B, d, D, H, W]`` convention."""

    def __init__(
        self,
        latent_shape_dhw: Sequence[int] = (24, 24, 24),
        block_size_dhw: Sequence[int] = (4, 4, 4),
    ):
        self.latent_shape_dhw = as_3tuple(latent_shape_dhw, "latent_shape_dhw")
        self.block_size_dhw = as_3tuple(block_size_dhw, "block_size_dhw")
        d, h, w = self.latent_shape_dhw
        bd, bh, bw = self.block_size_dhw
        if d % bd or h % bh or w % bw:
            raise ValueError(
                f"latent_shape {self.latent_shape_dhw} must be divisible by block_size {self.block_size_dhw}"
            )
        self.block_grid_dhw = (d // bd, h // bh, w // bw)
        self.num_blocks = (
            self.block_grid_dhw[0] * self.block_grid_dhw[1] * self.block_grid_dhw[2]
        )
        self.tokens_per_block = bd * bh * bw

    def coords(self, device=None) -> torch.Tensor:
        return raster_coords_tensor(self.block_grid_dhw, device=device)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,d,D,H,W], got {tuple(x.shape)}")
        b, scalar_dim, d, h, w = x.shape
        if (d, h, w) != self.latent_shape_dhw:
            raise ValueError(
                f"Expected latent shape {self.latent_shape_dhw}, got {(d, h, w)}"
            )
        bd, bh, bw = self.block_size_dhw
        gd, gh, gw = self.block_grid_dhw
        out = x.reshape(b, scalar_dim, gd, bd, gh, bh, gw, bw)
        out = out.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        return out.reshape(b, self.num_blocks, self.tokens_per_block, scalar_dim)

    def unpatchify(self, blocks: torch.Tensor) -> torch.Tensor:
        if blocks.ndim != 4:
            raise ValueError(f"Expected [B,N,V,d], got {tuple(blocks.shape)}")
        b, n, v, scalar_dim = blocks.shape
        if n != self.num_blocks or v != self.tokens_per_block:
            raise ValueError(
                f"Expected N={self.num_blocks}, V={self.tokens_per_block}, got N={n}, V={v}"
            )
        bd, bh, bw = self.block_size_dhw
        gd, gh, gw = self.block_grid_dhw
        out = blocks.reshape(b, gd, gh, gw, bd, bh, bw, scalar_dim)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        d, h, w = self.latent_shape_dhw
        return out.reshape(b, scalar_dim, d, h, w)


def patchify_3d(
    x: torch.Tensor, block_size_dhw: Sequence[int] = (4, 4, 4)
) -> torch.Tensor:
    return BlockPatchify3D(
        tuple(int(v) for v in x.shape[-3:]), block_size_dhw
    ).patchify(x)


def unpatchify_3d(
    blocks: torch.Tensor,
    latent_shape_dhw: Sequence[int] = (24, 24, 24),
    block_size_dhw: Sequence[int] = (4, 4, 4),
) -> torch.Tensor:
    return BlockPatchify3D(latent_shape_dhw, block_size_dhw).unpatchify(blocks)
