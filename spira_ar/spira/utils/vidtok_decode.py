from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import torch
from spira_ar.spira.data.preprocess_medical import (
    default_tokenizer_config_path,
    load_yaml_config,
    normalize_dataset_name,
)
from spira_ar.spira.tokenizer.fsq_codec import pack_fsq_scalars_to_indices
from spira_ar.spira.tokenizer.vidtok_wrapper import (
    autocast_context,
    load_medical_vidtok_wrapper,
)

SPIRA_ROOT = Path(__file__).resolve().parents[3]
DECODER_OUTPUT_MODES = ("middle", "average")


def load_vidtok_decoder_wrapper(
    *,
    dataset: str,
    config_path: Path | None,
    ckpt_path: Path | None,
    vidtok_root: Path | None,
    vidtok_config: Path | None,
    device: torch.device,
):
    cfg = load_yaml_config(
        config_path or default_tokenizer_config_path(normalize_dataset_name(dataset))
    )
    configure_vidtok_paths(cfg, vidtok_root, vidtok_config, ckpt_path)
    wrapper = load_medical_vidtok_wrapper(
        cfg, ckpt_path=cfg["vidtok"].get("ckpt_path"), lightweight_loss=True
    )
    return (wrapper.to(device).eval(), cfg)


def _decode_rgb_to_gray(x_rgb: torch.Tensor, decoder_output_mode: str) -> torch.Tensor:
    if decoder_output_mode not in DECODER_OUTPUT_MODES:
        raise ValueError(
            f"Unsupported decoder_output_mode: {decoder_output_mode!r}. Expected one of {DECODER_OUTPUT_MODES}."
        )
    if decoder_output_mode == "average":
        return x_rgb.mean(dim=1, keepdim=True).clamp(-1, 1)
    middle = x_rgb.shape[1] // 2
    return x_rgb[:, middle : middle + 1].clamp(-1, 1)


@torch.no_grad()
def decode_fsq_scalars_to_volume_hwd(
    *,
    wrapper: Any,
    scalars: torch.Tensor,
    levels: Sequence[int],
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int = 1,
    overlap: float = 0.5,
    mode: str = "gaussian",
    decoder_output_mode: str = "middle",
    precision: str = "fp32",
) -> torch.Tensor:
    """Decode full FSQ scalar latents to a MONAI-style volume ``[1,H,W,D]``.

    ``sliding_window_inference`` runs over latent windows in ``[D,H,W]`` order.
    The VidTok decoder returns ``[B,1,T,H,W]``; this function converts it back
    to ``[B,1,H,W,D]`` before returning the first sample.
    """
    if scalars.ndim != 4:
        raise ValueError(f"Expected [d,D,H,W] FSQ scalars, got {tuple(scalars.shape)}")
    if scalars.shape[0] != len(levels):
        raise ValueError(
            f"Expected scalar dim {len(levels)}, got {tuple(scalars.shape)}"
        )
    try:
        from monai.inferers import sliding_window_inference
    except ImportError as exc:
        raise RuntimeError(
            "MONAI is required for VidTok decoder sliding-window inference."
        ) from exc
    levels_list = [int(value) for value in levels]
    latent = scalars.unsqueeze(0).float()

    def predictor(latent_win: torch.Tensor) -> torch.Tensor:
        latent_win = latent_win.to(device=device).round().long()
        packed = torch.stack(
            [
                pack_fsq_scalars_to_indices(sample, levels_list, scalar_dim=0)
                for sample in latent_win
            ],
            dim=0,
        ).long()
        with autocast_context(device, precision):
            x_rgb = wrapper.model.decode(packed, decode_from_indices=True)
        return _decode_rgb_to_gray(x_rgb.float(), decoder_output_mode)

    decoded_tdhw = sliding_window_inference(
        inputs=latent,
        roi_size=roi_size,
        sw_batch_size=max(1, int(sw_batch_size)),
        predictor=predictor,
        overlap=float(overlap),
        mode=str(mode).lower(),
        sw_device=device,
        device=torch.device("cpu"),
    ).clamp(-1, 1)
    decoded_hwd = decoded_tdhw.permute(0, 1, 3, 4, 2).contiguous()
    return decoded_hwd[0].cpu()


def volume_hwd_to_metric_tensor(volume_hwd: torch.Tensor) -> torch.Tensor:
    """Convert ``[1,H,W,D]`` or ``[H,W,D]`` to VidTok/MONAI metric layout ``[1,1,D,H,W]``."""
    x = volume_hwd.detach().float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected [1,H,W,D] or [H,W,D], got {tuple(x.shape)}")
    return x.unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous()


def configure_vidtok_paths(cfg, vidtok_root, vidtok_config, ckpt):
    for key, value in [
        ("repo_root", vidtok_root),
        ("config_path", vidtok_config),
        ("ckpt_path", ckpt),
    ]:
        if value is not None:
            cfg.setdefault("vidtok", {})[key] = str(Path(value).expanduser().resolve())
