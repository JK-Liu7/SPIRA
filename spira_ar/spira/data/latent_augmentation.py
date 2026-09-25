from __future__ import annotations
from collections.abc import Mapping
import torch

DEFAULT_AUGMENTATION = {
    "enabled": True,
    "rand_flip": {"enabled": True, "prob_per_axis": {"D": 0.0, "H": 0.5, "W": 0.5}},
    "rand_rotate90": {"enabled": False},
}
LATENT_AUGMENT_KEYS = ("source_fsq_scalars", "target_fsq_scalars", "valid_mask_latent")
AXIS_TO_SPATIAL_DIM = {"D": 0, "H": 1, "W": 2}


def _cfg_get(mapping: Mapping | None, *keys, default=None):
    value = mapping or {}
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _seed_from_generator(generator: torch.Generator | None) -> int | None:
    if generator is None:
        return None
    return int(torch.randint(0, 2**31 - 1, (), generator=generator).item())


def _build_monai_latent_augmentation(cfg: Mapping | None):
    cfg = cfg or DEFAULT_AUGMENTATION
    if not bool(cfg.get("enabled", True)):
        return None
    try:
        from monai.transforms import Compose, RandFlipd, RandRotate90d
    except ImportError as exc:
        raise RuntimeError(
            "MONAI is required for latent augmentation with RandFlipd/RandRotate90d."
        ) from exc
    transforms = []
    flip_cfg = cfg.get("rand_flip", {})
    if bool(flip_cfg.get("enabled", True)):
        probs = flip_cfg.get("prob_per_axis", {})
        for axis_name in ("D", "H", "W"):
            prob = float(probs.get(axis_name, 0.0 if axis_name == "D" else 0.5))
            if prob > 0:
                transforms.append(
                    RandFlipd(
                        keys=LATENT_AUGMENT_KEYS,
                        prob=prob,
                        spatial_axis=AXIS_TO_SPATIAL_DIM[axis_name],
                        allow_missing_keys=True,
                    )
                )
    rot_cfg = cfg.get("rand_rotate90", {})
    if bool(rot_cfg.get("enabled", False)):
        prob = float(rot_cfg.get("prob", 0.0))
        if prob > 0:
            axes = tuple((str(axis) for axis in rot_cfg.get("axes", ("H", "W"))))
            if len(axes) != 2 or any(
                (axis not in AXIS_TO_SPATIAL_DIM for axis in axes)
            ):
                raise ValueError(
                    f"rand_rotate90.axes must be two of D/H/W, got {axes!r}"
                )
            transforms.append(
                RandRotate90d(
                    keys=LATENT_AUGMENT_KEYS,
                    prob=prob,
                    max_k=int(rot_cfg.get("max_k", 3)),
                    spatial_axes=(
                        AXIS_TO_SPATIAL_DIM[axes[0]],
                        AXIS_TO_SPATIAL_DIM[axes[1]],
                    ),
                    allow_missing_keys=True,
                )
            )
    return Compose(transforms) if transforms else None


def apply_monai_latent_augmentation(
    sample: dict,
    *,
    cfg: Mapping | None = None,
    generator: torch.Generator | None = None,
):
    """Apply synchronized MONAI dict augmentations to one cropped latent sample."""
    transform = _build_monai_latent_augmentation(cfg)
    if transform is None:
        return sample
    seed = _seed_from_generator(generator)
    if seed is not None and hasattr(transform, "set_random_state"):
        transform.set_random_state(seed=seed)
    data = dict(sample)
    mask = data.get("valid_mask_latent")
    if mask is not None and getattr(mask, "ndim", None) == 3:
        data["valid_mask_latent"] = mask.unsqueeze(0)
    data = transform(data)
    source = torch.as_tensor(data["source_fsq_scalars"]).long().contiguous()
    target = torch.as_tensor(data["target_fsq_scalars"]).long().contiguous()
    out_mask = torch.as_tensor(data["valid_mask_latent"]).bool().contiguous()
    if out_mask.ndim == 4 and out_mask.shape[0] == 1:
        out_mask = out_mask[0]
    sample["source_fsq_scalars"] = source
    sample["target_fsq_scalars"] = target
    sample["valid_mask_latent"] = out_mask.contiguous()
    return sample


def augment_latent_pair(
    source: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    cfg: Mapping | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply synchronized grid-preserving latent augmentations with MONAI.

    ``source`` and ``target`` are expected as ``[d, D, H, W]``. No interpolation
    or arbitrary geometric transform is applied to discrete FSQ ids.
    """
    if valid_mask is None:
        valid_mask = torch.ones(
            source.shape[1:], dtype=torch.bool, device=source.device
        )
        had_mask = False
    else:
        had_mask = True
    sample = {
        "source_fsq_scalars": source,
        "target_fsq_scalars": target,
        "valid_mask_latent": valid_mask,
    }
    sample = apply_monai_latent_augmentation(sample, cfg=cfg, generator=generator)
    return (
        sample["source_fsq_scalars"],
        sample["target_fsq_scalars"],
        sample["valid_mask_latent"] if had_mask else None,
    )
