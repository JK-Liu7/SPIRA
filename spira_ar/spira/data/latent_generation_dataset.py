from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data import Dataset

from spira_ar.spira.tokenizer.grid_patch_with_coords import crop_chwd


class ModalityPatchDataset(Dataset):
    """In-memory patch dataset used to batch VidTok encoding during cache generation."""

    def __init__(self, volume_chwd, patch_table: Sequence[dict[str, int]]) -> None:
        self.volume_chwd = volume_chwd
        self.patch_table = list(patch_table)

    def __len__(self) -> int:
        return len(self.patch_table)

    def __getitem__(self, index: int):
        patch_meta = self.patch_table[index]
        return {
            "patch_id": int(patch_meta["patch_id"]),
            "crop": crop_chwd(self.volume_chwd, patch_meta),
        }
