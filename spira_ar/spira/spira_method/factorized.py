from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from spira_ar.spira.models.factorized_fsq_head import FactorizedFSQHead


@dataclass(frozen=True)
class PairedFactorizedRollout:
    source_logits: list[torch.Tensor]
    full_logits: list[torch.Tensor]
    mixed_logits: list[torch.Tensor]
    predictions: torch.Tensor


def _broadcast_gate(gate: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    value = gate.to(device=logits.device, dtype=logits.dtype)
    while value.ndim < logits.ndim:
        value = value.unsqueeze(-1)
    return value


def _mix_one(
    source: torch.Tensor,
    full: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """FP32 interpolation with exact endpoint bypasses.

    Exact 0/1 gates are common for source/full controls and for the oracle
    grid.  Bypassing arithmetic at those endpoints guarantees that the
    full-raster control is bitwise identical to the standard raster logits.
    """

    alpha = _broadcast_gate(gate, source).float()
    source_fp32 = source.float()
    full_fp32 = full.float()
    interpolated = source_fp32 + alpha * (full_fp32 - source_fp32)
    return torch.where(
        alpha <= 0.0,
        source_fp32,
        torch.where(alpha >= 1.0, full_fp32, interpolated),
    )


def mix_factorized_logits(
    source_logits: Sequence[torch.Tensor],
    full_logits: Sequence[torch.Tensor],
    gate: torch.Tensor,
    *,
    detach_endpoints: bool = False,
) -> list[torch.Tensor]:
    if len(source_logits) != len(full_logits):
        raise ValueError("source/full factorized logit counts differ")
    mixed: list[torch.Tensor] = []
    for source, full in zip(source_logits, full_logits):
        if source.shape != full.shape:
            raise ValueError(
                f"source/full logit shapes differ: {tuple(source.shape)} vs "
                f"{tuple(full.shape)}"
            )
        if detach_endpoints:
            source = source.detach()
            full = full.detach()
        mixed.append(_mix_one(source, full, gate))
    return mixed


def paired_factorized_logits_and_predictions(
    head: FactorizedFSQHead,
    source_context: torch.Tensor,
    full_context: torch.Tensor,
    gate: torch.Tensor,
    gt_scalars: torch.Tensor | None = None,
    *,
    conditioning: str,
    greedy: bool = True,
    temperature: float = 1.0,
) -> PairedFactorizedRollout:
    """Roll out source/full endpoints with exactly one shared scalar prefix."""

    if source_context.shape != full_context.shape:
        raise ValueError(
            "source/full contexts differ: "
            f"{tuple(source_context.shape)} vs {tuple(full_context.shape)}"
        )
    conditioning = str(conditioning).lower()
    if conditioning not in {"teacher", "predicted"}:
        raise ValueError("conditioning must be 'teacher' or 'predicted'")
    if conditioning == "teacher" and gt_scalars is None:
        raise ValueError("gt_scalars is required for teacher conditioning")
    if gt_scalars is not None and gt_scalars.shape[-1] != head.scalar_dim:
        raise ValueError(
            f"Expected scalar dim {head.scalar_dim}, got {tuple(gt_scalars.shape)}"
        )

    # Counterfactual history residuals are intentionally kept in FP32.  Promote
    # both endpoints together so exact zero residuals cannot create an
    # artificial source/full difference merely from asymmetric input dtypes.
    context_dtype = torch.promote_types(
        source_context.dtype,
        full_context.dtype,
    )
    source_context = source_context.to(dtype=context_dtype)
    full_context = full_context.to(dtype=context_dtype)
    previous = torch.zeros_like(source_context)
    source_logits_all: list[torch.Tensor] = []
    full_logits_all: list[torch.Tensor] = []
    mixed_logits_all: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for scalar_index, level in enumerate(head.levels):
        source_hidden = head.scalar_mlps[scalar_index](source_context + previous)
        full_hidden = head.scalar_mlps[scalar_index](full_context + previous)
        source_logits = head._head_forward_fp32(
            head.heads[scalar_index],
            source_hidden,
        )
        full_logits = head._head_forward_fp32(
            head.heads[scalar_index],
            full_hidden,
        )
        mixed_logits = _mix_one(source_logits, full_logits, gate)
        probabilities = (mixed_logits / max(float(temperature), 1.0e-6)).softmax(dim=-1)
        if greedy:
            sample = probabilities.argmax(dim=-1)
        else:
            sample = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]),
                num_samples=1,
            ).reshape(probabilities.shape[:-1])
        source_logits_all.append(source_logits)
        full_logits_all.append(full_logits)
        mixed_logits_all.append(mixed_logits)
        predictions.append(sample)

        if conditioning == "teacher":
            scalar = gt_scalars[..., scalar_index].long().clamp(0, level - 1)
        else:
            scalar = sample
        # The same teacher or mixed predicted scalar is fed to both endpoints.
        previous = previous + head.prev_scalar_embeds[scalar_index](scalar)

    return PairedFactorizedRollout(
        source_logits=source_logits_all,
        full_logits=full_logits_all,
        mixed_logits=mixed_logits_all,
        predictions=torch.stack(predictions, dim=-1),
    )
