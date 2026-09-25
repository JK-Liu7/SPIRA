from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable
import torch
from torch.utils.data import Dataset
from .latent_window_sampler import crop_latent_window, sample_valid_latent_window
from .latent_augmentation import DEFAULT_AUGMENTATION, apply_monai_latent_augmentation
from spira_ar.spira.utils.raster_order import modality_to_id
from spira_ar.spira.data.latent_root_resolver import rebase_manifest_record

try:
    from monai.transforms import Transform as _MonaiTransform
except ImportError:
    _MonaiTransform = object
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


def canonical_modality_name(modality: str | None) -> str:
    text = str(modality or "").strip()
    return MODALITY_ALIASES.get(text.lower(), text)


def _same_modality(left: str | None, right: str | None) -> bool:
    return (
        canonical_modality_name(left).lower() == canonical_modality_name(right).lower()
    )


def torch_load(path: Path | str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    out = Path(path)
    if not out.is_absolute() and base_dir is not None:
        out = base_dir / out
    return out


def _resolve_manifest_cache_path(
    row: dict[str, Any],
    key: str,
    *,
    latent_root: Path | str | None,
    manifest_base: Path | None = None,
    modality: str | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a manifest entry relative to the configured latent root."""
    value = row.get(key)
    if not value:
        raise KeyError(f"Manifest row has no {key!r}; keys={list(row.keys())}")
    if latent_root is None:
        return _resolve_path(value, manifest_base)
    resolver_row = dict(row)
    resolver_row.pop("cache_relpath", None)
    resolver_row["cache_path"] = str(value)
    if modality:
        resolver_row["modality"] = modality
    return Path(
        rebase_manifest_record(resolver_row, latent_root, must_exist=must_exist)[
            "cache_path"
        ]
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def load_cache_index(path: Path | str) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(path.rglob("*.pt"))
    if path.suffix.lower() == ".jsonl":
        rows: list[Path] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(Path(row.get("cache_path") or row.get("path")))
        return rows
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("caches", data.get("data", []))
        return [
            Path(row.get("cache_path") if isinstance(row, dict) else row)
            for row in data
        ]
    raise ValueError(f"Unsupported cache index path: {path}")


def _cache_is_paired(cache: dict[str, Any]) -> bool:
    return "source_fsq_scalars" in cache and "target_fsq_scalars" in cache


def _cache_is_modality(cache: dict[str, Any]) -> bool:
    return "fsq_scalars" in cache and "modality" in cache


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple((int(v) for v in value))


def _group_key(cache: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(cache.get("dataset", "")).lower(),
        str(cache.get("group_id") or cache.get("case_id", "")),
        _shape_tuple(cache.get("latent_shape_dhw")),
        _shape_tuple(cache.get("padded_shape_hwd")),
    )


def _record_from_paired_cache(path: Path, cache: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_path": str(path),
        "case_id": str(cache.get("case_id", path.stem)),
        "dataset": str(cache.get("dataset", "")),
        "source_modality": str(cache.get("source_modality", "")),
        "target_modality": str(cache.get("target_modality", "")),
        "pair_id": str(
            cache.get("pair_id") or f"{cache.get('case_id', path.stem)}_pair"
        ),
    }


def build_latent_pair_records(
    cache_index: Path | str | None = None,
    *,
    latent_root: Path | str | None = None,
    dataset: str | None = None,
    source_modality: str | None = None,
    target_modality: str | None = None,
) -> list[dict[str, Any]]:
    """Build source-target latent cache pairs.

    Supports three inputs:
    - paired cache files with ``source_fsq_scalars`` and ``target_fsq_scalars``;
    - manifest rows with ``source_cache_path`` and ``target_cache_path``;
    - modality cache files with ``fsq_scalars`` grouped by dataset/case.
    """
    index_path = (
        Path(cache_index)
        if cache_index is not None
        else Path(latent_root)
        if latent_root is not None
        else None
    )
    if index_path is None:
        raise ValueError("Either cache_index or latent_root must be provided.")
    explicit_pairs: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    manifest_base = index_path.parent if index_path.is_file() else index_path
    if index_path.is_file() and index_path.suffix.lower() == ".jsonl":
        for row in _iter_jsonl(index_path):
            if "source_cache_path" in row and "target_cache_path" in row:
                source_path = _resolve_manifest_cache_path(
                    row,
                    "source_cache_path",
                    latent_root=latent_root,
                    manifest_base=manifest_base,
                    modality=str(row.get("source_modality", "") or "") or None,
                )
                target_path = _resolve_manifest_cache_path(
                    row,
                    "target_cache_path",
                    latent_root=latent_root,
                    manifest_base=manifest_base,
                    modality=str(row.get("target_modality", "") or "") or None,
                )
                explicit_pairs.append(
                    {
                        **row,
                        "manifest_source_cache_path": str(
                            row.get("source_cache_path", "")
                        ),
                        "manifest_target_cache_path": str(
                            row.get("target_cache_path", "")
                        ),
                        "source_cache_path": str(source_path),
                        "target_cache_path": str(target_path),
                    }
                )
            else:
                cache_key = (
                    "cache_path"
                    if row.get("cache_path")
                    else "path"
                    if row.get("path")
                    else None
                )
                if cache_key:
                    candidate_paths.append(
                        _resolve_manifest_cache_path(
                            row,
                            cache_key,
                            latent_root=latent_root,
                            manifest_base=manifest_base,
                            modality=str(row.get("modality", "") or "") or None,
                        )
                    )
    else:
        candidate_paths.extend(load_cache_index(index_path))
    modality_rows: dict[tuple[Any, ...], dict[str, tuple[Path, dict[str, Any]]]] = {}
    paired_records: list[dict[str, Any]] = []
    for path in candidate_paths:
        cache = torch_load(path)
        if dataset and str(cache.get("dataset", "")).lower() not in {
            "",
            str(dataset).lower(),
        }:
            continue
        if _cache_is_paired(cache):
            paired_records.append(_record_from_paired_cache(path, cache))
            continue
        if not _cache_is_modality(cache):
            continue
        modality = canonical_modality_name(str(cache.get("modality", "")))
        if source_modality and target_modality:
            if not (
                _same_modality(modality, source_modality)
                or _same_modality(modality, target_modality)
            ):
                continue
        modality_rows.setdefault(_group_key(cache), {})[modality.lower()] = (
            path,
            cache,
        )
    source_key = (
        canonical_modality_name(source_modality).lower() if source_modality else None
    )
    target_key = (
        canonical_modality_name(target_modality).lower() if target_modality else None
    )
    modality_pairs: list[dict[str, Any]] = []
    if source_key and target_key:
        for _, by_modality in sorted(
            modality_rows.items(), key=lambda item: str(item[0])
        ):
            if source_key not in by_modality or target_key not in by_modality:
                continue
            source_path, source_cache = by_modality[source_key]
            target_path, target_cache = by_modality[target_key]
            case_id = str(
                source_cache.get("case_id")
                or target_cache.get("case_id")
                or source_path.stem
            )
            modality_pairs.append(
                {
                    "source_cache_path": str(source_path),
                    "target_cache_path": str(target_path),
                    "case_id": case_id,
                    "group_id": str(
                        source_cache.get("group_id") or target_cache.get("group_id", "")
                    ),
                    "dataset": str(
                        source_cache.get("dataset") or target_cache.get("dataset", "")
                    ),
                    "source_modality": str(
                        source_cache.get("modality", source_modality)
                    ),
                    "target_modality": str(
                        target_cache.get("modality", target_modality)
                    ),
                    "pair_id": f"{case_id}_{source_modality}_to_{target_modality}",
                }
            )
    records = [*explicit_pairs, *modality_pairs, *paired_records]
    if source_modality:
        records = [
            row
            for row in records
            if _same_modality(row.get("source_modality"), source_modality)
        ]
    if target_modality:
        records = [
            row
            for row in records
            if _same_modality(row.get("target_modality"), target_modality)
        ]
    if not records:
        raise ValueError(
            f"No source-target latent pairs were found. Check latent_root/manifest and source_modality={source_modality!r}, target_modality={target_modality!r}."
        )
    return records


class LoadLatentPairTransform(_MonaiTransform):
    """Deterministic latent-cache reader used inside MONAI Compose/PersistentDataset."""

    def __init__(self, *, num_modality_buckets: int = 64) -> None:
        if _MonaiTransform is not object:
            super().__init__()
        self.num_modality_buckets = int(num_modality_buckets)

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        if "cache_path" in record and record["cache_path"]:
            cache_path = Path(record["cache_path"])
            cache = torch_load(cache_path)
            source = cache["source_fsq_scalars"].long()
            target = cache["target_fsq_scalars"].long()
            mask = cache["valid_mask_latent"].bool()
            fsq_levels = [int(v) for v in cache["fsq_levels"]]
            source_modality = str(
                cache.get("source_modality", record.get("source_modality", ""))
            )
            target_modality = str(
                cache.get("target_modality", record.get("target_modality", ""))
            )
            meta = cache
        else:
            source_path = Path(record["source_cache_path"])
            target_path = Path(record["target_cache_path"])
            source_cache = torch_load(source_path)
            target_cache = torch_load(target_path)
            source = source_cache["fsq_scalars"].long()
            target = target_cache["fsq_scalars"].long()
            source_mask = source_cache.get("valid_mask_latent")
            target_mask = target_cache.get("valid_mask_latent")
            if source_mask is None:
                source_mask = torch.ones(source.shape[1:], dtype=torch.bool)
            if target_mask is None:
                target_mask = torch.ones(target.shape[1:], dtype=torch.bool)
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"Source/target latent shapes differ: {tuple(source.shape)} vs {tuple(target.shape)}"
                )
            if tuple(source_mask.shape) != tuple(target_mask.shape):
                raise ValueError(
                    f"Source/target valid masks differ: {tuple(source_mask.shape)} vs {tuple(target_mask.shape)}"
                )
            mask = source_mask.bool() & target_mask.bool()
            fsq_levels = [
                int(v)
                for v in source_cache.get(
                    "fsq_levels", target_cache.get("fsq_levels", [])
                )
            ]
            target_levels = [int(v) for v in target_cache.get("fsq_levels", fsq_levels)]
            if fsq_levels != target_levels:
                raise ValueError(
                    f"Source/target FSQ levels differ: {fsq_levels} vs {target_levels}"
                )
            source_modality = str(
                source_cache.get("modality", record.get("source_modality", ""))
            )
            target_modality = str(
                target_cache.get("modality", record.get("target_modality", ""))
            )
            meta = {**source_cache, **record}
        out = {
            "source_fsq_scalars": source.contiguous(),
            "target_fsq_scalars": target.contiguous(),
            "valid_mask_latent": mask.contiguous(),
            "fsq_levels": torch.as_tensor(fsq_levels, dtype=torch.long),
            "cache_path": str(record.get("cache_path", "")),
            "source_cache_path": str(record.get("source_cache_path", "")),
            "target_cache_path": str(record.get("target_cache_path", "")),
            "case_id": str(record.get("case_id", meta.get("case_id", ""))),
            "group_id": str(record.get("group_id", meta.get("group_id", ""))),
            "pair_id": str(record.get("pair_id", meta.get("pair_id", ""))),
            "dataset": str(record.get("dataset", meta.get("dataset", ""))),
            "source_modality": source_modality,
            "target_modality": target_modality,
            "source_modality_id": torch.tensor(
                modality_to_id(source_modality, self.num_modality_buckets),
                dtype=torch.long,
            ),
            "target_modality_id": torch.tensor(
                modality_to_id(target_modality, self.num_modality_buckets),
                dtype=torch.long,
            ),
            "latent_shape_dhw": torch.as_tensor(
                meta.get("latent_shape_dhw", mask.shape), dtype=torch.long
            ),
            "original_shape_hwd": meta.get("original_shape_hwd"),
            "padded_shape_hwd": meta.get("padded_shape_hwd"),
        }
        return out


def build_monai_latent_preprocess_transform(*, num_modality_buckets: int = 64):
    try:
        from monai.transforms import Compose
    except ImportError as exc:
        raise RuntimeError(
            "MONAI is required for latent preprocessing transforms."
        ) from exc
    return Compose([LoadLatentPairTransform(num_modality_buckets=num_modality_buckets)])


def _center_start(
    shape: tuple[int, int, int], window: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple((max((dim - win) // 2, 0) for dim, win in zip(shape, window)))


class LatentWindowDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        window_dhw: tuple[int, int, int] = (24, 24, 24),
        min_valid_ratio: float = 0.85,
        max_tries: int = 100,
        seed: int = 2026,
        training: bool = True,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.window_dhw = tuple((int(value) for value in window_dhw))
        self.min_valid_ratio = float(min_valid_ratio)
        self.max_tries = int(max_tries)
        self.seed = int(seed)
        self.training = bool(training)
        self.augmentation = (
            augmentation if augmentation is not None else DEFAULT_AUGMENTATION
        )
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        full = dict(self.base_dataset[index])
        valid_mask = full["valid_mask_latent"].bool()
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * max(len(self), 1) + int(index))
        shape = tuple((int(v) for v in valid_mask.shape))
        if shape == self.window_dhw:
            start = (0, 0, 0)
        elif self.training:
            start = sample_valid_latent_window(
                valid_mask,
                window=self.window_dhw,
                min_valid_ratio=self.min_valid_ratio,
                max_tries=self.max_tries,
                generator=generator,
            )
        else:
            start = _center_start(shape, self.window_dhw)
        sample = crop_latent_window(full, start, window=self.window_dhw)
        if (
            self.training
            and self.augmentation
            and bool(self.augmentation.get("enabled", True))
        ):
            sample = apply_monai_latent_augmentation(
                sample, cfg=self.augmentation, generator=generator
            )
        valid_ratio = sample["valid_mask_latent"].float().mean()
        for key in (
            "cache_path",
            "source_cache_path",
            "target_cache_path",
            "case_id",
            "group_id",
            "pair_id",
            "dataset",
            "source_modality",
            "target_modality",
            "source_modality_id",
            "target_modality_id",
            "fsq_levels",
            "latent_shape_dhw",
            "original_shape_hwd",
            "padded_shape_hwd",
        ):
            if key in full:
                sample[key] = full[key]
        sample["valid_ratio"] = valid_ratio
        return sample


class _ListDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], transform: Any) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transform(dict(self.records[index]))


class LatentPairDataset(LatentWindowDataset):
    """AR dataset that reads precomputed VidTok latent caches without MONAI caching."""

    def __init__(
        self,
        cache_index: Path | str,
        latent_root: Path | str | None = None,
        window_dhw: tuple[int, int, int] = (24, 24, 24),
        min_valid_ratio: float = 0.85,
        max_tries: int = 100,
        seed: int = 2026,
        dataset: str | None = None,
        source_modality: str | None = None,
        target_modality: str | None = None,
        training: bool = True,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        records = build_latent_pair_records(
            cache_index,
            latent_root=latent_root,
            dataset=dataset,
            source_modality=source_modality,
            target_modality=target_modality,
        )
        base = _ListDataset(records, build_monai_latent_preprocess_transform())
        super().__init__(
            base,
            window_dhw=window_dhw,
            min_valid_ratio=min_valid_ratio,
            max_tries=max_tries,
            seed=seed,
            training=training,
            augmentation=augmentation,
        )


def build_persistent_latent_pair_dataset(
    *,
    cache_index: Path | str | None = None,
    latent_root: Path | str | None = None,
    persistent_cache_dir: Path | str,
    dataset: str | None = None,
    source_modality: str | None = None,
    target_modality: str | None = None,
    window_dhw: tuple[int, int, int] = (24, 24, 24),
    min_valid_ratio: float = 0.85,
    max_tries: int = 100,
    seed: int = 2026,
    training: bool = True,
    augmentation: dict[str, Any] | None = None,
) -> LatentWindowDataset:
    try:
        from monai.data import PersistentDataset
    except ImportError as exc:
        raise RuntimeError(
            "MONAI is required for PersistentDataset latent loading."
        ) from exc
    records = build_latent_pair_records(
        cache_index,
        latent_root=latent_root,
        dataset=dataset,
        source_modality=source_modality,
        target_modality=target_modality,
    )
    persistent_kwargs: dict[str, Any] = {}
    base = PersistentDataset(
        data=records,
        transform=build_monai_latent_preprocess_transform(),
        cache_dir=str(persistent_cache_dir),
        **persistent_kwargs,
    )
    return LatentWindowDataset(
        base,
        window_dhw=window_dhw,
        min_valid_ratio=min_valid_ratio,
        max_tries=max_tries,
        seed=seed,
        training=training,
        augmentation=augmentation,
    )
