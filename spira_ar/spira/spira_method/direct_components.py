from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import torch
import torch.nn as nn

from spira_ar.spira.models.inter_block_llamagen import RMSNorm


def endpoint_distribution_features(
    source_logits: Sequence[torch.Tensor],
    full_logits: Sequence[torch.Tensor],
    valid_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached endpoint TV and logit-correction RMS per latent token.

    Total variation is averaged equally across FSQ scalar factors.  Logit RMS
    treats the factors' category axes as one concatenated correction vector,
    matching ``RMS(z_full - z_source)`` even when factor cardinalities differ.
    """

    if not source_logits or len(source_logits) != len(full_logits):
        raise ValueError(
            "source/full factorized logits must be nonempty and have equal counts"
        )
    reference_shape = tuple(source_logits[0].shape[:-1])
    if len(reference_shape) != 3:
        raise ValueError(
            "factorized endpoint logits must be [B,N,V,K], got "
            f"{tuple(source_logits[0].shape)}"
        )

    device_type = source_logits[0].device.type
    fp32_context = (
        torch.autocast(device_type=device_type, enabled=False)
        if device_type in {"cpu", "cuda"}
        else nullcontext()
    )
    with torch.no_grad(), fp32_context:
        tv_factors: list[torch.Tensor] = []
        residual_square_sum = torch.zeros(
            reference_shape,
            device=source_logits[0].device,
            dtype=torch.float32,
        )
        residual_element_count = 0
        for source, full in zip(source_logits, full_logits, strict=True):
            if source.shape != full.shape:
                raise ValueError(
                    "source/full factorized logit shapes differ: "
                    f"{tuple(source.shape)} vs {tuple(full.shape)}"
                )
            if tuple(source.shape[:-1]) != reference_shape:
                raise ValueError(
                    "all factorized logits must share [B,N,V], got "
                    f"{tuple(source.shape[:-1])} vs {reference_shape}"
                )
            source_fp32 = source.detach().float()
            full_fp32 = full.detach().float()
            source_probability = source_fp32.softmax(dim=-1)
            full_probability = full_fp32.softmax(dim=-1)
            tv_factors.append(
                0.5 * (source_probability - full_probability).abs().sum(dim=-1)
            )
            residual = full_fp32 - source_fp32
            residual_square_sum = residual_square_sum + residual.square().sum(dim=-1)
            residual_element_count += int(residual.shape[-1])

        endpoint_tv = torch.stack(tv_factors, dim=0).mean(dim=0)
        logit_residual_rms = torch.log1p(
            (residual_square_sum / float(residual_element_count)).sqrt()
        )
        if valid_token_mask is not None:
            if tuple(valid_token_mask.shape) != reference_shape:
                raise ValueError(
                    "valid_token_mask must match endpoint logits [B,N,V]: "
                    f"{tuple(valid_token_mask.shape)} vs {reference_shape}"
                )
            valid = valid_token_mask.to(
                device=endpoint_tv.device,
                dtype=torch.bool,
            )
            endpoint_tv = endpoint_tv.masked_fill(~valid, 0.0)
            logit_residual_rms = logit_residual_rms.masked_fill(~valid, 0.0)
    return endpoint_tv, logit_residual_rms


class SpiraRetractionGateHead(nn.Module):
    """Predict one prefix-correction retraction per latent token.

    Hidden endpoint features and the two endpoint-distribution statistics are
    detached, so the retraction KL can update only this module and cannot
    reshape the shared autoregressive predictor or factorized FSQ head.
    """

    def __init__(self, hidden_dim: int, gate_hidden_dim: int = 128) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        gate_hidden_dim = int(gate_hidden_dim)
        if hidden_dim < 1 or gate_hidden_dim < 1:
            raise ValueError("hidden_dim and gate_hidden_dim must be positive")
        self.source_norm = RMSNorm(hidden_dim)
        self.residual_norm = RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, gate_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(gate_hidden_dim),
            nn.Linear(gate_hidden_dim, 1),
        )

    def forward(
        self,
        h_source: torch.Tensor,
        h_full: torch.Tensor,
        source_logits: Sequence[torch.Tensor],
        full_logits: Sequence[torch.Tensor],
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h_source.ndim != 4:
            raise ValueError(f"h_source must be [B,N,V,C], got {tuple(h_source.shape)}")
        if h_full.shape != h_source.shape:
            raise ValueError(
                "h_full must match h_source [B,N,V,C]: "
                f"{tuple(h_full.shape)} vs {tuple(h_source.shape)}"
            )
        valid = None
        if valid_token_mask is not None:
            if tuple(valid_token_mask.shape) != tuple(h_source.shape[:-1]):
                raise ValueError(
                    "valid_token_mask must match gate output [B,N,V]: "
                    f"{tuple(valid_token_mask.shape)} vs {tuple(h_source.shape[:-1])}"
                )
            valid = valid_token_mask.to(
                device=h_source.device,
                dtype=torch.bool,
            )

        endpoint_tv, logit_residual_rms = endpoint_distribution_features(
            source_logits,
            full_logits,
            valid,
        )
        source = h_source.detach()
        residual = (h_full - h_source).detach()
        if valid is not None:
            expanded_valid = valid.unsqueeze(-1)
            source = source.masked_fill(~expanded_valid, 0.0)
            residual = residual.masked_fill(~expanded_valid, 0.0)

        device_type = source.device.type
        fp32_context = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in {"cpu", "cuda"}
            else nullcontext()
        )
        with fp32_context:
            gate_input = torch.cat(
                [
                    self.source_norm(source).float(),
                    self.residual_norm(residual).float(),
                    endpoint_tv.unsqueeze(-1),
                    logit_residual_rms.unsqueeze(-1),
                ],
                dim=-1,
            )
            gate_logit = self.mlp(gate_input.float()).squeeze(-1)
            retraction = torch.sigmoid(gate_logit)
        retraction = retraction.to(dtype=h_source.dtype)
        if valid is not None:
            retraction = retraction.masked_fill(~valid, 0.0)
        return retraction
