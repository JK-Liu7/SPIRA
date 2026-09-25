from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
SPIRA_ROOT = THIS_FILE.parents[2]
if str(SPIRA_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIRA_ROOT))
from spira_ar.spira.data.latent_generation_dataset import ModalityPatchDataset
from spira_ar.spira.data.preprocess_medical import (
    build_modality_case_records,
    build_modality_full_volume_transform,
    default_tokenizer_config_path,
    load_and_preprocess_modality_case,
    load_configured_records,
    load_yaml_config,
    normalize_dataset_name,
    safe_modality_cache_stem,
)
from spira_ar.spira.tokenizer.center_stitch import (
    keep_bounds_1d,
    latent_shape_dhw_from_hwd,
    stitch_all_patches,
)
from spira_ar.spira.tokenizer.fsq_codec import (
    get_fsq_levels,
    pack_fsq_scalars_to_indices,
    validate_scalar_ranges,
)
from spira_ar.spira.tokenizer.grid_patch_with_coords import (
    COMPRESSION_HWD,
    PATCH_SIZE_HWD,
    build_patch_index_table,
    compute_padded_shape_hwd,
    pad_chwd_to_shape,
    padding_info,
    stride_from_overlap,
)
from spira_ar.spira.tokenizer.valid_mask import (
    build_image_valid_mask_hwd,
    downsample_valid_mask_to_latent,
)
from spira_ar.spira.tokenizer.vidtok_wrapper import (
    encode_crops_to_fsq_scalars,
    load_medical_vidtok_wrapper,
)


def parse_hwd(
    values: list[int] | tuple[int, int, int], name: str
) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must have 3 values, got {values!r}")
    return tuple((int(value) for value in values))


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def patch_table_with_latent_metadata(
    patch_table: list[dict[str, int]],
    starts_hwd: tuple[list[int], list[int], list[int]],
    padded_shape_hwd: tuple[int, int, int],
    patch_size_hwd: tuple[int, int, int],
    comp_hwd: tuple[int, int, int],
) -> list[dict[str, int | bool]]:
    latent_shape_dhw = latent_shape_dhw_from_hwd(padded_shape_hwd, comp_hwd=comp_hwd)
    starts_h_l = [start // comp_hwd[0] for start in starts_hwd[0]]
    starts_w_l = [start // comp_hwd[1] for start in starts_hwd[1]]
    starts_d_l = [start // comp_hwd[2] for start in starts_hwd[2]]
    tile_h_l = patch_size_hwd[0] // comp_hwd[0]
    tile_w_l = patch_size_hwd[1] // comp_hwd[1]
    tile_d_l = patch_size_hwd[2] // comp_hwd[2]
    full_d, full_h, full_w = latent_shape_dhw
    enriched: list[dict[str, int | bool]] = []
    for meta in patch_table:
        row: dict[str, int | bool] = dict(meta)
        h0_l = int(meta["h0"] // comp_hwd[0])
        w0_l = int(meta["w0"] // comp_hwd[1])
        d0_l = int(meta["d0"] // comp_hwd[2])
        src_h, dst_h = keep_bounds_1d(
            h0_l, tile_h_l, full_h, starts_h_l, int(meta["ih"])
        )
        src_w, dst_w = keep_bounds_1d(
            w0_l, tile_w_l, full_w, starts_w_l, int(meta["iw"])
        )
        src_d, dst_d = keep_bounds_1d(
            d0_l, tile_d_l, full_d, starts_d_l, int(meta["id"])
        )
        row.update(
            {
                "latent_h0": h0_l,
                "latent_h1": h0_l + tile_h_l,
                "latent_w0": w0_l,
                "latent_w1": w0_l + tile_w_l,
                "latent_d0": d0_l,
                "latent_d1": d0_l + tile_d_l,
                "keep_src_h0": int(src_h.start),
                "keep_src_h1": int(src_h.stop),
                "keep_src_w0": int(src_w.start),
                "keep_src_w1": int(src_w.stop),
                "keep_src_d0": int(src_d.start),
                "keep_src_d1": int(src_d.stop),
                "keep_dst_h0": int(dst_h.start),
                "keep_dst_h1": int(dst_h.stop),
                "keep_dst_w0": int(dst_w.start),
                "keep_dst_w1": int(dst_w.stop),
                "keep_dst_d0": int(dst_d.start),
                "keep_dst_d1": int(dst_d.stop),
                "is_first_h": int(meta["ih"]) == 0,
                "is_last_h": int(meta["ih"]) == int(meta["num_h"]) - 1,
                "is_first_w": int(meta["iw"]) == 0,
                "is_last_w": int(meta["iw"]) == int(meta["num_w"]) - 1,
                "is_first_d": int(meta["id"]) == 0,
                "is_last_d": int(meta["id"]) == int(meta["num_d"]) - 1,
            }
        )
        enriched.append(row)
    return enriched


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def cache_record_summary(
    cache_path: Path, case_record: dict[str, Any], modality: str, status: str
) -> dict[str, Any]:
    return {
        "cache_path": str(cache_path),
        "status": status,
        "dataset": case_record.get("dataset", ""),
        "case_id": case_record.get("case_id", ""),
        "group_id": case_record.get("group_id", ""),
        "modality": modality,
        "available_modalities": sorted(case_record.get("modalities", {}).keys()),
    }


@torch.no_grad()
def encode_volume_patches(
    *,
    wrapper: Any,
    volume_chwd: torch.Tensor,
    patch_table: list[dict[str, int]],
    levels: list[int],
    precision: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[torch.Tensor]:
    dataset = ModalityPatchDataset(volume_chwd, patch_table)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=device.type == "cuda",
    )
    patch_latents: list[torch.Tensor | None] = [None for _ in patch_table]
    for batch in loader:
        crops = batch["crop"]
        patch_ids = batch["patch_id"].tolist()
        scalars_b, _ = encode_crops_to_fsq_scalars(
            wrapper, crops, levels=levels, precision=precision, device=device
        )
        for item_idx, patch_id in enumerate(patch_ids):
            patch_latents[int(patch_id)] = scalars_b[item_idx]
    missing = [idx for idx, value in enumerate(patch_latents) if value is None]
    if missing:
        raise RuntimeError(
            f"Missing encoded patch latents for patch ids: {missing[:10]}"
        )
    return [value for value in patch_latents if value is not None]


def save_modality_cache(
    *,
    output_path: Path,
    full_scalars: torch.Tensor,
    valid_mask_latent: torch.Tensor,
    valid_mask_latent_soft: torch.Tensor,
    original_shape_hwd: tuple[int, int, int],
    padded_shape_hwd: tuple[int, int, int],
    latent_shape_dhw: tuple[int, int, int],
    patch_table: list[dict[str, int | bool]],
    starts_hwd: tuple[list[int], list[int], list[int]],
    levels: list[int],
    meta: dict[str, Any],
    modality: str,
    save_packed_index: bool,
) -> None:
    modality_metadata = meta.get("metadata_by_modality", {}).get(modality, {})
    cache: dict[str, Any] = {
        "fsq_scalars": full_scalars,
        "valid_mask_latent": valid_mask_latent.bool(),
        "valid_mask_latent_soft": valid_mask_latent_soft.to(torch.float16),
        "original_shape_hwd": list(original_shape_hwd),
        "padded_shape_hwd": list(padded_shape_hwd),
        "latent_shape_dhw": list(latent_shape_dhw),
        "padding_info": padding_info(original_shape_hwd, padded_shape_hwd),
        "patch_index_table": patch_table,
        "patch_starts_hwd": [list(axis) for axis in starts_hwd],
        "fsq_levels": list(levels),
        "spacing": modality_metadata.get("spacing"),
        "affine": modality_metadata.get("affine"),
        "modality_metadata": modality_metadata,
        "metadata_by_modality": meta.get("metadata_by_modality", {}),
        "case_id": meta.get("case_id", ""),
        "group_id": meta.get("group_id", ""),
        "dataset": meta.get("dataset", ""),
        "modality": modality,
        "available_modalities": list(meta.get("modalities", [])),
        "modality_image_path": meta.get("modality_image_paths", {}).get(modality, ""),
        "modality_image_paths": meta.get("modality_image_paths", {}),
        "task": meta.get("task", ""),
        "anatomy": meta.get("anatomy", ""),
        "cohort": meta.get("cohort", ""),
        "tracer": meta.get("tracer", ""),
        "split": meta.get("split", ""),
        "normalization_info": {
            "input_range": [-1, 1],
            "padding_value": -1.0,
            "modality": modality,
            "preprocessing_scope": "case_shared_all_modalities",
        },
        "stitching_info": {
            "mode": "nearest_center_center_crop",
            "allow_scalar_blending": False,
        },
    }
    if save_packed_index:
        cache["packed_index"] = pack_fsq_scalars_to_indices(
            full_scalars, levels, scalar_dim=0
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)


@torch.no_grad()
def process_case(
    *,
    case_record: dict[str, Any],
    transform: Any,
    wrapper: Any,
    levels: list[int],
    output_dir: Path,
    dataset: str,
    patch_size_hwd: tuple[int, int, int],
    stride_hwd: tuple[int, int, int],
    comp_hwd: tuple[int, int, int],
    pad_mode: str,
    min_latent_shape_dhw: tuple[int, int, int],
    include_last_patch: bool,
    precision: str,
    save_packed_index: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    overwrite: bool,
) -> list[dict[str, Any]]:
    volumes, meta = load_and_preprocess_modality_case(transform, case_record)
    first_volume = next(iter(volumes.values()))
    original_shape_hwd = tuple((int(dim) for dim in first_volume.shape[1:]))
    padded_shape_hwd = compute_padded_shape_hwd(
        original_shape_hwd,
        dataset=dataset,
        patch_hwd=patch_size_hwd,
        stride_hwd=stride_hwd,
        comp_hwd=comp_hwd,
        pad_mode=pad_mode,
        min_latent_shape_dhw=min_latent_shape_dhw,
    )
    patch_table, starts_hwd = build_patch_index_table(
        padded_shape_hwd,
        patch_hwd=patch_size_hwd,
        stride_hwd=stride_hwd,
        comp_hwd=comp_hwd,
        include_last=include_last_patch,
    )
    valid_mask_hwd = build_image_valid_mask_hwd(original_shape_hwd, padded_shape_hwd)
    valid_mask_latent, valid_mask_latent_soft = downsample_valid_mask_to_latent(
        valid_mask_hwd, comp_hwd=comp_hwd, threshold=0.5
    )
    latent_shape_dhw = tuple((int(dim) for dim in valid_mask_latent.shape))
    enriched_patch_table = patch_table_with_latent_metadata(
        patch_table,
        starts_hwd,
        padded_shape_hwd,
        patch_size_hwd=patch_size_hwd,
        comp_hwd=comp_hwd,
    )
    rows: list[dict[str, Any]] = []
    for modality, volume in volumes.items():
        cache_path = (
            output_dir
            / modality
            / f"{safe_modality_cache_stem(case_record, modality)}.pt"
        )
        if cache_path.exists() and (not overwrite):
            rows.append(
                cache_record_summary(
                    cache_path, case_record, modality, "skipped_existing"
                )
            )
            continue
        volume_pad = pad_chwd_to_shape(volume, padded_shape_hwd, value=-1.0)
        patch_latents = encode_volume_patches(
            wrapper=wrapper,
            volume_chwd=volume_pad,
            patch_table=patch_table,
            levels=levels,
            precision=precision,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        full_scalars, filled = stitch_all_patches(
            patch_latents,
            patch_table,
            starts_hwd,
            padded_shape_hwd,
            levels,
            comp_hwd=comp_hwd,
        )
        validate_scalar_ranges(full_scalars, levels, name=f"{modality}_fsq_scalars")
        if not bool((filled == 1).all().item()):
            raise RuntimeError(
                f"Unexpected non-unique filled mask for {case_record.get('case_id')} {modality}"
            )
        save_modality_cache(
            output_path=cache_path,
            full_scalars=full_scalars,
            valid_mask_latent=valid_mask_latent,
            valid_mask_latent_soft=valid_mask_latent_soft,
            original_shape_hwd=original_shape_hwd,
            padded_shape_hwd=padded_shape_hwd,
            latent_shape_dhw=latent_shape_dhw,
            patch_table=enriched_patch_table,
            starts_hwd=starts_hwd,
            levels=levels,
            meta=meta,
            modality=modality,
            save_packed_index=save_packed_index,
        )
        rows.append(
            {
                **cache_record_summary(cache_path, case_record, modality, "written"),
                "original_shape_hwd": list(original_shape_hwd),
                "padded_shape_hwd": list(padded_shape_hwd),
                "latent_shape_dhw": list(latent_shape_dhw),
                "num_patches": len(patch_table),
                "valid_ratio": float(valid_mask_latent.float().mean().item()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute per-modality full-volume VidTok-FSQ latent caches for SPIRA AR."
    )
    parser.add_argument("--dataset", default="brats24")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Dataset preprocessing and frozen tokenizer configuration.",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "val", "all"), default="all"
    )
    parser.add_argument(
        "--modalities",
        default=None,
        help="Optional comma-separated modality filter, e.g. t1,t2,flair,t1ce or CT,PET.",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path, default=None
    )
    parser.add_argument(
        "--manifest-out", "--manifest_out", dest="manifest_out", type=Path, default=None
    )
    parser.add_argument(
        "--vidtok-root", "--vidtok_root", dest="vidtok_root", type=Path, default=None
    )
    parser.add_argument(
        "--vidtok-config",
        "--vidtok_config",
        dest="vidtok_config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--ckpt",
        "--vidtok-ckpt",
        "--vidtok_ckpt",
        dest="ckpt",
        type=Path,
        default=None,
        help="Tokenizer checkpoint; defaults to vidtok.ckpt_path in the configuration.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=2,
        help="VidTok crop encoding batch size.",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=8,
        help="Patch DataLoader workers used during latent generation.",
    )
    parser.add_argument(
        "--patch-size-hwd",
        "--patch_size_hwd",
        dest="patch_size_hwd",
        nargs=3,
        type=int,
        default=list(PATCH_SIZE_HWD),
    )
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument(
        "--compression-hwd",
        "--compression_hwd",
        dest="compression_hwd",
        nargs=3,
        type=int,
        default=list(COMPRESSION_HWD),
    )
    parser.add_argument(
        "--pad-mode",
        "--pad_mode",
        dest="pad_mode",
        choices=("auto", "no_pad", "coverage"),
        default="auto",
    )
    parser.add_argument(
        "--min-latent-shape-dhw",
        "--min_latent_shape_dhw",
        dest="min_latent_shape_dhw",
        nargs=3,
        type=int,
        default=[32, 32, 32],
    )
    parser.add_argument(
        "--no-include-last-patch",
        "--no_include_last_patch",
        dest="no_include_last_patch",
        action="store_true",
    )
    parser.add_argument(
        "--save-packed-index",
        "--save_packed_index",
        dest="save_packed_index",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
    )
    parser.add_argument(
        "--no-save-packed-index",
        "--no_save_packed_index",
        dest="save_packed_index",
        action="store_false",
    )
    parser.add_argument(
        "--dtype",
        choices=("int16",),
        default="int16",
        help="Latent scalar dtype; currently fixed to int16.",
    )
    parser.add_argument(
        "--max-cases", "--max_cases", dest="max_cases", type=int, default=None
    )
    parser.add_argument(
        "--start-index", "--start_index", dest="start_index", type=int, default=0
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Build case list and print counts without loading VidTok.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = normalize_dataset_name(args.dataset)
    config_path = args.config or default_tokenizer_config_path(dataset)
    cfg = load_yaml_config(config_path)
    configure_vidtok_paths(cfg, args.vidtok_root, args.vidtok_config, args.ckpt)
    records = load_configured_records(cfg, dataset=dataset, split=args.split)
    case_records = build_modality_case_records(
        records, dataset=dataset, modalities=args.modalities
    )
    if args.start_index:
        case_records = case_records[args.start_index :]
    if args.max_cases is not None:
        case_records = case_records[: args.max_cases]
    if not case_records:
        raise RuntimeError("No cases matched the requested dataset/modality filters.")
    output_dir = (
        Path(args.output_dir) if args.output_dir else SPIRA_ROOT / "latents" / dataset
    )
    manifest_out = (
        args.manifest_out or output_dir / f"manifest_{dataset}_{args.split}.jsonl"
    )
    print(f"Dataset: {dataset}")
    print(f"Config: {config_path}")
    print(f"Cases to process: {len(case_records)}")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_out}")
    print(f"Batch size: {args.batch_size}")
    if args.dry_run:
        preview = case_records[:5]
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return
    device = torch.device(args.device)
    wrapper = (
        load_medical_vidtok_wrapper(
            cfg, ckpt_path=cfg["vidtok"].get("ckpt_path"), lightweight_loss=True
        )
        .to(device)
        .eval()
    )
    levels = get_fsq_levels(wrapper.model.regularization)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "fsq_levels.json", levels)
    patch_size_hwd = parse_hwd(args.patch_size_hwd, "patch_size_hwd")
    stride_hwd = stride_from_overlap(patch_size_hwd, args.overlap)
    comp_hwd = parse_hwd(args.compression_hwd, "compression_hwd")
    min_latent_shape_dhw = parse_hwd(args.min_latent_shape_dhw, "min_latent_shape_dhw")
    rows: list[dict[str, Any]] = []
    for case_record in tqdm(case_records, desc="Precomputing VidTok latent caches"):
        transform = build_modality_full_volume_transform(
            dataset, cfg["data"], list(case_record["modalities"].keys())
        )
        try:
            case_rows = process_case(
                case_record=case_record,
                transform=transform,
                wrapper=wrapper,
                levels=levels,
                output_dir=output_dir,
                dataset=dataset,
                patch_size_hwd=patch_size_hwd,
                stride_hwd=stride_hwd,
                comp_hwd=comp_hwd,
                pad_mode=args.pad_mode,
                min_latent_shape_dhw=min_latent_shape_dhw,
                include_last_patch=not args.no_include_last_patch,
                precision=args.precision,
                save_packed_index=args.save_packed_index,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                overwrite=args.overwrite,
            )
            rows.extend(case_rows)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote manifest with {len(rows)} modality-cache rows to {manifest_out}")


def configure_vidtok_paths(cfg, vidtok_root, vidtok_config, ckpt):
    for name, value in [
        ("repo_root", vidtok_root),
        ("config_path", vidtok_config),
        ("ckpt_path", ckpt),
    ]:
        if value is not None:
            cfg.setdefault("vidtok", {})[name] = str(Path(value).expanduser().resolve())


if __name__ == "__main__":
    main()
