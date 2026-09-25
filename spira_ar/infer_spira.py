"""Source-only full-volume SPIRA deployment with categorical latent stitching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spira_ar.validate_spira import load_checkpoint_model
from spira_ar.spira.data.latent_pair_dataset import canonical_modality_name
from spira_ar.spira.data.latent_root_resolver import resolve_manifest_cache_path
from spira_ar.spira.tokenizer.fsq_codec import validate_scalar_ranges
from spira_ar.spira.utils.config import load_config
from spira_ar.spira.utils.runtime import autocast_context, valid_token_mask_from_batch
from spira_ar.spira.utils.volume_inference import (
    _latent_window_metas,
    _image_blend_weight,
)
from spira_ar.spira.utils.vidtok_decode import (
    load_vidtok_decoder_wrapper,
    decode_fsq_scalars_to_volume_hwd,
)


@torch.no_grad()
def predict_full_volume(
    model,
    scalars,
    valid_mask,
    *,
    device,
    precision="fp32",
    overlap=0.25,
    use_kv_cache=True,
):
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    shape = tuple(scalars.shape[-3:])
    roi = tuple(model.patchifier.latent_shape_dhw)
    padded = tuple(max(a, b) for a, b in zip(shape, roi))
    padding = tuple(
        value
        for size, full in reversed(list(zip(shape, padded)))
        for value in (0, full - size)
    )
    source = F.pad(scalars.long(), padding)
    mask = F.pad(valid_mask.bool(), padding)
    stride = tuple(max(1, round(n * (1 - overlap))) for n in roi)
    metas = _latent_window_metas(padded, roi, stride)
    prediction = torch.full((len(model.levels), *padded), -1, dtype=torch.long)
    gates = torch.zeros(padded, dtype=torch.float32)
    weights = torch.zeros(padded, dtype=torch.float32)
    blend = _image_blend_weight(roi, mode="gaussian")
    for meta in metas:
        start = meta["start"]
        slices = tuple(slice(s, s + size) for s, size in zip(start, roi))
        crop = source[(slice(None), *slices)].unsqueeze(0).to(device)
        crop_mask = mask[slices]
        token_mask = valid_token_mask_from_batch(
            {"valid_mask_latent": crop_mask.unsqueeze(0).to(device)},
            model.patchifier.block_size_dhw,
        )
        with autocast_context(device, precision):
            out = model.generate(
                crop,
                valid_token_mask=token_mask,
                greedy=True,
                use_kv_cache=use_kv_cache,
            )
        discrete = out["target_fsq_scalars"][0].cpu()
        src, dst = meta["src_slices"], meta["dst_slices"]
        prediction[(slice(None), *dst)] = discrete[(slice(None), *src)]
        gate = (
            model.patchifier.unpatchify(out["retraction_active"].unsqueeze(-1))[0, 0]
            .float()
            .cpu()
        )
        weight = blend * crop_mask.float()
        gates[slices] += gate * weight
        weights[slices] += weight
    if bool((prediction < 0).any()):
        raise RuntimeError("Unassigned cells remain after categorical stitching")
    crop = tuple(slice(0, n) for n in shape)
    return prediction[(slice(None), *crop)], (gates / weights.clamp_min(1e-6))[crop]


def source_paths(args, cfg):
    if args.source_cache:
        return [args.source_cache]
    data = cfg["data"]
    rows = [
        json.loads(line)
        for line in Path(data["val_manifest"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    modality = canonical_modality_name(data["source_modality"])
    return [
        resolve_manifest_cache_path(row, data["latent_root"], must_exist=True)
        for row in rows
        if canonical_modality_name(str(row.get("modality", ""))) == modality
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="One source modality cache; otherwise use validation manifest sources.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Also decode predictions to NIfTI in the preprocessed image grid.",
    )
    parser.add_argument("--tokenizer-config", type=Path)
    parser.add_argument("--tokenizer-ckpt", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model, levels = load_checkpoint_model(args.ckpt, cfg, device, args.use_ema)
    paths = source_paths(args, cfg)
    if args.max_cases is not None:
        paths = paths[: args.max_cases]
    if not paths:
        raise RuntimeError("No source modality caches were found")
    settings = cfg.get("inference", {})
    window = settings.get("sliding_window", {})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(tqdm(paths, desc="SPIRA volumes")):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if list(cache["fsq_levels"]) != levels:
            raise ValueError("Source cache and AR checkpoint use different FSQ levels")
        source = cache["fsq_scalars"].long()
        validate_scalar_ranges(source, levels)
        mask = cache.get(
            "valid_mask_latent", torch.ones(source.shape[-3:], dtype=torch.bool)
        )
        prediction, gate = predict_full_volume(
            model,
            source,
            mask,
            device=device,
            precision=cfg["train"].get("precision", "fp32"),
            overlap=float(window.get("overlap", 0.25)),
            use_kv_cache=bool(settings.get("use_kv_cache", True)),
        )
        case_dir = args.output_dir / f"{index:04d}_{path.stem}"
        case_dir.mkdir(parents=True, exist_ok=False)
        torch.save(
            {
                "fsq_scalars": prediction.to(torch.int16),
                "gate": gate,
                "fsq_levels": levels,
                "valid_mask_latent": mask,
            },
            case_dir / "prediction.pt",
        )
        if args.decode:
            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            decoder, _ = load_vidtok_decoder_wrapper(
                dataset=cfg["data"]["dataset"],
                config_path=args.tokenizer_config,
                ckpt_path=args.tokenizer_ckpt,
                vidtok_root=None,
                vidtok_config=None,
                device=device,
            )
            from spira_ar.spira.tokenizer.fsq_codec import get_fsq_levels

            if get_fsq_levels(decoder.model.regularization) != levels:
                raise ValueError("Decoder and AR checkpoint use different FSQ levels")
            volume = decode_fsq_scalars_to_volume_hwd(
                wrapper=decoder,
                scalars=prediction,
                levels=levels,
                device=device,
                roi_size=tuple(window.get("decode_latent_window_size", (8, 32, 32))),
                overlap=float(window.get("decode_overlap", 0.25)),
                sw_batch_size=int(window.get("decode_sw_batch_size", 1)),
                mode=window.get("decode_blend", "gaussian"),
                precision=window.get("decode_precision", "fp32"),
            )
            original = cache.get("original_shape_hwd", volume.shape[-3:])
            volume = volume[0, : original[0], : original[1], : original[2]].numpy()
            import nibabel as nib

            affine = cache.get("affine")
            if affine is None:
                raise ValueError(
                    "Decoding to NIfTI requires preprocessed affine metadata in the cache"
                )
            nib.save(
                nib.Nifti1Image(volume, np.asarray(affine)),
                case_dir / "prediction.nii.gz",
            )
            del decoder
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.to(device)
        (case_dir / "summary.json").write_text(
            json.dumps(
                {
                    "case_id": cache.get("case_id"),
                    "latent_shape_dhw": list(prediction.shape[-3:]),
                    "gate_mean": float(gate[mask].mean()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
