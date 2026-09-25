from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedFSQHead(nn.Module):
    """Autoregressive scalar head for FSQ tokens, avoiding a 262144-way head."""

    def __init__(
        self,
        levels: Sequence[int],
        hidden_dim: int = 768,
        label_smoothing: float = 0.0,
        ordinal_loss_enabled: bool = False,
        ordinal_loss_weight: float = 0.0,
        ordinal_loss_type: str = "l1",
        ordinal_value_mapping: str = "vidtok_fsq",
        ordinal_reduction: str = "masked_mean",
    ) -> None:
        super().__init__()
        self.levels = [int(v) for v in levels]
        self.label_smoothing = float(label_smoothing)
        self.ordinal_loss_enabled = bool(ordinal_loss_enabled)
        self.ordinal_loss_weight = float(ordinal_loss_weight)
        self.ordinal_loss_type = str(ordinal_loss_type).lower()
        self.ordinal_value_mapping = str(ordinal_value_mapping).lower()
        self.ordinal_reduction = str(ordinal_reduction).lower()
        if self.ordinal_loss_type != "l1":
            raise ValueError(f"Unsupported ordinal loss type: {ordinal_loss_type!r}")
        if self.ordinal_value_mapping != "vidtok_fsq":
            raise ValueError(
                f"Unsupported ordinal value mapping: {ordinal_value_mapping!r}"
            )
        if self.ordinal_reduction != "masked_mean":
            raise ValueError(f"Unsupported ordinal reduction: {ordinal_reduction!r}")
        if self.ordinal_loss_weight < 0.0:
            raise ValueError("ordinal_loss_weight must be non-negative")
        for index, level in enumerate(self.levels):
            if level < 2:
                raise ValueError(
                    f"FSQ level at scalar {index} must be at least 2, got {level}"
                )
            half_width = level // 2
            values = (
                torch.arange(level, dtype=torch.float32) - float(half_width)
            ) / float(half_width)
            self.register_buffer(f"ordinal_values_{index}", values, persistent=False)
        self.prev_scalar_embeds = nn.ModuleList(
            [nn.Embedding(level, hidden_dim) for level in self.levels]
        )
        self.scalar_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in self.levels
            ]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, level, bias=False) for level in self.levels]
        )
        for head in self.heads:
            nn.init.normal_(head.weight, std=0.02)

    @property
    def scalar_dim(self) -> int:
        return len(self.levels)

    @staticmethod
    def _head_forward_fp32(head: nn.Linear, hidden: torch.Tensor) -> torch.Tensor:
        """Run the small categorical projection in FP32, even under autocast.

        ``head.weight.float()`` deliberately keeps this robust when the parent
        model has been explicitly converted to BF16/FP16 rather than merely
        executed under autocast. The cast remains differentiable, so gradients
        still propagate to the original head parameters.
        """
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            bias = None if head.bias is None else head.bias.float()
            return F.linear(hidden.float(), head.weight.float(), bias)

    def ordinal_loss(
        self,
        logits: Sequence[torch.Tensor],
        gt_scalars: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid: torch.Tensor | None = None
        valid_indices: torch.Tensor | None = None
        if mask is not None:
            if tuple(mask.shape) != tuple(gt_scalars.shape[:-1]):
                raise ValueError(
                    f"ordinal mask must match target token layout: {tuple(mask.shape)} vs {tuple(gt_scalars.shape[:-1])}"
                )
            valid = mask.to(device=gt_scalars.device, dtype=torch.bool)
            valid_indices = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        errors: list[torch.Tensor] = []
        for j, logits_j in enumerate(logits):
            values = getattr(self, f"ordinal_values_{j}").to(device=logits_j.device)
            if valid_indices is None:
                probs = logits_j.float().softmax(dim=-1)
                expected = probs @ values
                target = values[gt_scalars[..., j].long().clamp(0, self.levels[j] - 1)]
                errors.append((expected - target).abs())
                continue
            flat_logits = logits_j.float().reshape(-1, logits_j.shape[-1])
            flat_targets = gt_scalars[..., j].long().reshape(-1)
            flat_error = flat_logits.new_zeros(flat_targets.shape)
            if valid_indices.numel():
                selected_logits = flat_logits.index_select(0, valid_indices)
                selected_targets = flat_targets.index_select(0, valid_indices)
                selected_probs = selected_logits.softmax(dim=-1)
                selected_expected = selected_probs @ values
                selected_target = values[selected_targets]
                selected_error = (selected_expected - selected_target).abs()
                flat_error = flat_error.index_copy(0, valid_indices, selected_error)
            errors.append(flat_error.reshape(gt_scalars.shape[:-1]))
        error = torch.stack(errors, dim=-1)
        if valid is None:
            return error.mean()
        valid_factor = valid.unsqueeze(-1).expand_as(error)
        selected = error.masked_select(valid_factor)
        if selected.numel():
            return selected.mean()
        zero = logits[0].reshape(-1)[:0].sum().float()
        for value in logits[1:]:
            zero = zero + value.reshape(-1)[:0].sum().float()
        return zero
