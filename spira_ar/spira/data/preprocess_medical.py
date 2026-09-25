from __future__ import annotations
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import math


def modality_foreground_mask(image, modality: str):
    threshold = -950 if modality.upper() in {"CT", "CBCT"} else 0
    return image > threshold


class MultiModalityForegroundMaskd:
    def __init__(
        self,
        image_keys: Sequence[str],
        modality_by_key: dict[str, str],
        foreground_key: str = "foreground",
    ) -> None:
        self.image_keys = list(image_keys)
        self.modality_by_key = dict(modality_by_key)
        self.foreground_key = foreground_key

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.image_keys:
            raise ValueError("image_keys must not be empty.")
        reference = data[self.image_keys[0]]
        reference_shape = tuple(reference.shape)
        reference_affine = getattr(reference, "affine", None)
        mask = None
        for key in self.image_keys:
            image = data[key]
            if tuple(image.shape) != reference_shape:
                raise ValueError(
                    f"All modalities must share the same grid before shared CropForegroundd: {self.image_keys[0]}={reference_shape}, {key}={tuple(image.shape)}."
                )
            image_affine = getattr(image, "affine", None)
            if reference_affine is not None and image_affine is not None:
                import torch

                if not torch.allclose(
                    torch.as_tensor(reference_affine, dtype=torch.float64),
                    torch.as_tensor(image_affine, dtype=torch.float64),
                    atol=0.001,
                ):
                    raise ValueError(
                        f"All modalities must share the same affine before shared CropForegroundd. Bad key: {key}"
                    )
            current = modality_foreground_mask(image, self.modality_by_key.get(key, ""))
            mask = current if mask is None else mask | current
        data[self.foreground_key] = mask
        return data


class ModalityNormalizeIntensityd:
    def __init__(
        self,
        key: str = "image",
        low: float = 0.5,
        high: float = 99.5,
        max_quantile_samples: int | None = 1000000,
        modality_key: str = "modality",
    ) -> None:
        self.key = key
        self.low = low
        self.high = high
        self.max_quantile_samples = (
            None if max_quantile_samples is None else int(max_quantile_samples)
        )
        self.modality_key = modality_key

    def _sample_for_quantile(self, values):
        if self.max_quantile_samples is None or self.max_quantile_samples <= 0:
            return values
        num_values = values.numel()
        if num_values <= self.max_quantile_samples:
            return values
        stride = math.ceil(num_values / self.max_quantile_samples)
        return values[::stride]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import torch

        x = data[self.key].float()
        modality = str(data.get(self.modality_key, data.get("modality", ""))).upper()
        if modality in {"CT", "CBCT"}:
            x = (x.clamp(-1000, 1000) + 1000) / 2000
        else:
            values = x[x > 0]
            if values.numel() == 0:
                values = x.reshape(-1)
            values = self._sample_for_quantile(values).contiguous()
            lo = torch.quantile(values, self.low / 100.0)
            hi = torch.quantile(values, self.high / 100.0)
            x = ((x - lo) / (hi - lo).clamp_min(1e-06)).clamp(0, 1)
        data[self.key] = x * 2 - 1
        return data


def normalize_dataset_name(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    if key in {"brats", "brats24"}:
        return "brats24"
    if key in {"synthrad", "synthrad2025"}:
        return "synthrad"
    if key in {"autopet", "autopetiii"}:
        return "autopet"
    raise ValueError(f"Unsupported dataset: {name}")


def _split_modalities(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    if value.lower() in {"all", "any", "*"}:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _record_path(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    raise KeyError(f"Record is missing all path keys: {keys}")


MODALITY_ALIASES = {
    "t1": "t1n",
    "t1n": "t1n",
    "t2": "t2w",
    "t2w": "t2w",
    "flair": "t2f",
    "t2f": "t2f",
    "t1ce": "t1c",
    "t1c": "t1c",
    "mr": "MRI",
    "mri": "MRI",
    "ct": "CT",
    "cbct": "CBCT",
    "pet": "PET",
}
DEFAULT_LATENT_MODALITIES = {"autopet": ("CT", "PET")}


def canonical_modality_name(modality: str) -> str:
    key = str(modality).strip()
    return MODALITY_ALIASES.get(key.lower(), key)


def modality_field_name(modality: str) -> str:
    text = canonical_modality_name(modality)
    text = re.sub("[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return f"mod_{text or 'image'}"


def _requested_modalities(modalities: str | Sequence[str] | None) -> set[str] | None:
    if modalities is None:
        return None
    if isinstance(modalities, str):
        values = _split_modalities(modalities)
    else:
        values = [str(item) for item in modalities]
    if not values:
        return None
    return {canonical_modality_name(item).lower() for item in values}


def _case_prefix(record: dict[str, Any], dataset_key: str) -> tuple[Any, ...]:
    if dataset_key == "brats24":
        return (record.get("cohort", ""), record.get("case_id", ""))
    if dataset_key == "synthrad":
        return (
            record.get("task", ""),
            record.get("anatomy", ""),
            record.get("case_id", ""),
        )
    return (record.get("case_id", ""),)


def build_modality_case_records(
    records: Sequence[dict[str, Any]],
    dataset: str,
    modalities: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    dataset_key = normalize_dataset_name(dataset)
    requested = _requested_modalities(modalities)
    if (
        modalities is None
        and requested is None
        and (dataset_key in DEFAULT_LATENT_MODALITIES)
    ):
        requested = {
            canonical_modality_name(item).lower()
            for item in DEFAULT_LATENT_MODALITIES[dataset_key]
        }
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        modality = canonical_modality_name(str(record.get("modality", "")))
        if not modality:
            continue
        if requested is not None and modality.lower() not in requested:
            continue
        grouped[_case_prefix(record, dataset_key)][modality] = record
    cases: list[dict[str, Any]] = []
    for key, by_modality in sorted(grouped.items()):
        if not by_modality:
            continue
        first = next(iter(by_modality.values()))
        case_id = str(first.get("case_id", key[-1] if key else "case"))
        case_record = {
            "dataset": "BraTS24"
            if dataset_key == "brats24"
            else "SynthRAD"
            if dataset_key == "synthrad"
            else "AutoPET",
            "case_id": case_id,
            "group_id": "_".join((str(part) for part in key if part)),
            "modalities": {
                modality: _record_path(record, "image_path", "image")
                for modality, record in sorted(by_modality.items())
            },
            "modality_records": {
                modality: dict(record)
                for modality, record in sorted(by_modality.items())
            },
            "cohort": str(first.get("cohort", "")),
            "task": str(first.get("task", "")),
            "anatomy": str(first.get("anatomy", "")),
            "tracer": str(first.get("tracer", "")),
            "split": str(first.get("split", "")),
        }
        cases.append(case_record)
    return cases


def safe_modality_cache_stem(case_record: dict[str, Any], modality: str) -> str:
    base_parts = [
        str(case_record.get("task", "")),
        str(case_record.get("anatomy", "")),
        str(case_record.get("cohort", "")),
        str(case_record.get("tracer", "")),
        str(case_record.get("case_id", "case")),
        canonical_modality_name(modality),
    ]
    base = "_".join((part for part in base_parts if part))
    base = re.sub("[^A-Za-z0-9_.=-]+", "_", base).strip("_")
    return base or "case"


def build_modality_full_volume_transform(
    dataset: str, data_cfg: dict[str, Any], modalities: Sequence[str]
):
    from monai.transforms import (
        Compose,
        CropForegroundd,
        DeleteItemsd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        Resized,
        ScaleIntensityRangePercentilesd,
    )

    dataset_key = normalize_dataset_name(dataset)
    modalities = [canonical_modality_name(item) for item in modalities]
    image_keys = [modality_field_name(item) for item in modalities]
    modality_by_key = {modality_field_name(item): item for item in modalities}
    foreground_key = "foreground"
    orientation = data_cfg.get("orientation", "RAS")
    transforms: list[Any] = [
        LoadImaged(keys=image_keys),
        EnsureChannelFirstd(keys=image_keys),
        Orientationd(keys=image_keys, axcodes=orientation),
    ]
    norm_cfg = data_cfg.get("normalization", {})
    if dataset_key == "brats24":
        transforms.extend(
            [
                CropForegroundd(
                    keys=image_keys, source_key=image_keys[0], allow_smaller=True
                ),
                ScaleIntensityRangePercentilesd(
                    keys=image_keys,
                    lower=float(norm_cfg.get("percentile_low", 0.5)),
                    upper=float(norm_cfg.get("percentile_high", 99.5)),
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                ),
                Resized(
                    keys=image_keys,
                    spatial_size=configured_crop_size_hwd(data_cfg),
                    mode=tuple(("trilinear" for _ in image_keys)),
                ),
            ]
        )
    else:
        transforms.extend(
            [
                MultiModalityForegroundMaskd(
                    image_keys=image_keys,
                    modality_by_key=modality_by_key,
                    foreground_key=foreground_key,
                ),
                CropForegroundd(
                    keys=[*image_keys, foreground_key],
                    source_key=foreground_key,
                    allow_smaller=True,
                ),
            ]
        )
        for modality, key in zip(modalities, image_keys):
            transforms.append(
                ModalityNormalizeIntensityd(
                    key=key,
                    low=float(norm_cfg.get("percentile_low", 0.5)),
                    high=float(norm_cfg.get("percentile_high", 99.5)),
                    max_quantile_samples=norm_cfg.get("max_quantile_samples", 1000000),
                    modality_key=f"{key}_modality",
                )
            )
        transforms.append(DeleteItemsd(keys=[foreground_key]))
    transforms.append(EnsureTyped(keys=image_keys))
    return Compose(transforms)


def _tensor_to_plain_list(value: Any):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return [_tensor_to_plain_list(item) for item in value]
    return value


def affine_to_spacing_hwd(affine: Any) -> list[float] | None:
    if affine is None:
        return None
    import numpy as np

    array = np.asarray(_tensor_to_plain_list(affine), dtype=float)
    if array.ndim == 3:
        array = array[0]
    if array.shape[-2:] != (4, 4):
        return None
    spacing = [float(np.linalg.norm(array[:3, idx])) for idx in range(3)]
    return spacing if all((value > 0 for value in spacing)) else None


def configured_crop_size_hwd(data_cfg: dict[str, Any]) -> tuple[int, int, int]:
    size = data_cfg.get(
        "crop_size", data_cfg.get("full_spatial_size_hwd", (256, 256, 128))
    )
    if len(size) != 3:
        raise ValueError(
            f"data.crop_size/full_spatial_size_hwd must have 3 values, got {size!r}"
        )
    return tuple((int(value) for value in size))


def tensor_metadata(tensor: Any) -> dict[str, Any]:
    meta = getattr(tensor, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
    affine = getattr(tensor, "affine", None)
    if affine is None:
        affine = meta.get("affine") or meta.get("original_affine")
    return {
        "affine": _tensor_to_plain_list(affine),
        "spacing": affine_to_spacing_hwd(affine),
        "meta": {
            key: _tensor_to_plain_list(value)
            for key, value in meta.items()
            if key
            in {
                "filename_or_obj",
                "original_affine",
                "spatial_shape",
                "original_channel_dim",
            }
        },
    }


def load_and_preprocess_modality_case(
    transform: Any, case_record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    data: dict[str, Any] = {}
    modalities = [canonical_modality_name(item) for item in case_record["modalities"]]
    for modality in modalities:
        key = modality_field_name(modality)
        data[key] = case_record["modalities"][modality]
        data[f"{key}_modality"] = modality
    transformed = transform(data)
    volumes: dict[str, Any] = {}
    metadata_by_modality: dict[str, Any] = {}
    shape_hwd: tuple[int, int, int] | None = None
    for modality in modalities:
        key = modality_field_name(modality)
        volume = torch.as_tensor(transformed[key]).float()
        if volume.ndim != 4 or volume.shape[0] != 1:
            raise ValueError(
                f"Expected {modality} [1,H,W,D], got {tuple(volume.shape)} for {case_record.get('case_id')}"
            )
        current_shape = tuple((int(dim) for dim in volume.shape[1:]))
        if shape_hwd is None:
            shape_hwd = current_shape
        elif current_shape != shape_hwd:
            raise ValueError(
                f"Shared preprocessing produced inconsistent shapes for {case_record.get('case_id')}: expected {shape_hwd}, got {current_shape} for {modality}"
            )
        volumes[modality] = volume
        metadata_by_modality[modality] = tensor_metadata(transformed[key])
    meta = {
        "case_id": str(case_record.get("case_id", "")),
        "group_id": str(case_record.get("group_id", "")),
        "dataset": str(case_record.get("dataset", "")),
        "modalities": modalities,
        "modality_image_paths": dict(case_record.get("modalities", {})),
        "task": str(case_record.get("task", "")),
        "anatomy": str(case_record.get("anatomy", "")),
        "cohort": str(case_record.get("cohort", "")),
        "tracer": str(case_record.get("tracer", "")),
        "split": str(case_record.get("split", "")),
        "metadata_by_modality": metadata_by_modality,
    }
    return (volumes, meta)


from spira_ar.spira.utils.config import load_config, PROJECT_ROOT


def load_yaml_config(path):
    return load_config(path)


def default_tokenizer_config_path(dataset):
    return (
        PROJECT_ROOT
        / "configs"
        / ("latent_" + normalize_dataset_name(dataset) + ".yaml")
    )


def load_configured_records(cfg, dataset, split):
    data = cfg["data"]
    root = Path(data["data_root"])
    splits = (
        ["train", "val"] if split == "all" else ["train" if split == "train" else "val"]
    )
    records = []
    for current in splits:
        path = Path(data[current + "_datalist"])
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get(
                "training" if current == "train" else "validation", value.get("data")
            )
        if not isinstance(value, list):
            raise ValueError("Datalist must contain a list of modality records")
        for raw in value:
            item = dict(raw)
            if not item.get("case_id") or not item.get("modality"):
                raise ValueError("Every record requires case_id and modality")
            p = Path(_record_path(item, "image_path", "image")).expanduser()
            item["image"] = str(p if p.is_absolute() else root / p)
            item["image_path"] = item["image"]
            item["split"] = current
            records.append(item)
    return records
