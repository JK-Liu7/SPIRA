from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _expanded_token_mask(
    mask: torch.Tensor | None,
    token_shape: torch.Size | tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Validate and broadcast a boolean token mask to ``token_shape``."""

    if mask is None:
        return None
    valid = mask.to(device=device, dtype=torch.bool)
    while valid.ndim < len(token_shape):
        valid = valid.unsqueeze(-1)
    try:
        return valid.expand(tuple(token_shape))
    except RuntimeError as exc:
        raise ValueError(
            "mask is not broadcastable to the token layout: "
            f"{tuple(mask.shape)} vs {tuple(token_shape)}"
        ) from exc


def _zero_with_logit_gradient(logits: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return scalar zero connected to logits without reading any logit value."""

    if not logits:
        raise ValueError("factorized logits must not be empty")
    zero = logits[0].reshape(-1)[:0].sum()
    for value in logits[1:]:
        zero = zero + value.reshape(-1)[:0].sum()
    return zero.float()


def _masked_cross_entropy_rows(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor | None,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    """Compute CE only for valid rows and return zero at invalid rows.

    Selecting valid rows before ``F.cross_entropy`` prevents invalid target IDs
    or non-finite padded logits from entering the numerical computation at all.
    """

    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_targets = targets.reshape(-1).long()
    if valid is None:
        return F.cross_entropy(
            flat_logits,
            flat_targets,
            reduction="none",
            label_smoothing=float(label_smoothing),
        ).reshape(targets.shape)

    flat_valid = valid.reshape(-1)
    valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
    flat_output = flat_logits.new_zeros(flat_valid.shape)
    if valid_indices.numel() == 0:
        return flat_output.reshape(targets.shape)

    selected = F.cross_entropy(
        flat_logits.index_select(0, valid_indices),
        flat_targets.index_select(0, valid_indices),
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    return flat_output.index_copy(0, valid_indices, selected).reshape(targets.shape)


def factorized_cross_entropy(
    logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean per-token summed CE and per-token unsmoothed NLL.

    ``targets`` is ``[..., d]``. Each logits tensor is ``[..., levels[j]]``.
    The returned NLL has shape ``targets.shape[:-1]`` and sums CE over scalar
    dimensions, matching ``-log product_j p(q_j)``.

    When ``mask`` is supplied, invalid rows are removed *before* CE. This makes
    padding robust to sentinel target IDs and non-finite padded logits, rather
    than relying on a post-hoc multiplication by zero.
    """

    if not logits:
        raise ValueError("factorized_cross_entropy requires logits")
    if targets.shape[-1] != len(logits):
        raise ValueError(
            f"targets last dim {targets.shape[-1]} does not match {len(logits)} logits"
        )
    valid = _expanded_token_mask(
        mask,
        targets.shape[:-1],
        device=targets.device,
    )
    ce_parts: list[torch.Tensor] = []
    nll_parts: list[torch.Tensor] = []
    for j, logits_j in enumerate(logits):
        if logits_j.shape[:-1] != targets.shape[:-1]:
            raise ValueError(
                f"logits[{j}] prefix {tuple(logits_j.shape[:-1])} does not "
                f"match targets {tuple(targets.shape[:-1])}"
            )
        ce = _masked_cross_entropy_rows(
            logits_j,
            targets[..., j],
            valid,
            label_smoothing=float(label_smoothing),
        )
        if float(label_smoothing) == 0.0:
            nll_j = ce
        else:
            nll_j = _masked_cross_entropy_rows(
                logits_j,
                targets[..., j],
                valid,
                label_smoothing=0.0,
            )
        ce_parts.append(ce)
        nll_parts.append(nll_j)

    ce_token = torch.stack(ce_parts, dim=-1).sum(dim=-1)
    nll = torch.stack(nll_parts, dim=-1).sum(dim=-1)
    if valid is None:
        loss = ce_token.mean()
    else:
        valid_values = ce_token.masked_select(valid)
        loss = (
            valid_values.mean()
            if valid_values.numel()
            else _zero_with_logit_gradient(logits)
        )
    return loss, nll


def _masked_entropy_rows(
    logits: torch.Tensor,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    token_shape = logits.shape[:-1]
    if valid is None:
        log_probs = flat_logits.log_softmax(dim=-1)
        probs = log_probs.exp()
        return (-(probs * log_probs).sum(dim=-1)).reshape(token_shape)

    flat_valid = valid.reshape(-1)
    valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
    flat_output = flat_logits.new_zeros(flat_valid.shape)
    if valid_indices.numel() == 0:
        return flat_output.reshape(token_shape)
    selected_logits = flat_logits.index_select(0, valid_indices)
    log_probs = selected_logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    selected = -(probs * log_probs).sum(dim=-1)
    return flat_output.index_copy(0, valid_indices, selected).reshape(token_shape)


def factorized_entropy(
    logits: Sequence[torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not logits:
        raise ValueError("factorized_entropy requires logits")
    expected_prefix = tuple(logits[0].shape[:-1])
    valid = _expanded_token_mask(
        mask,
        expected_prefix,
        device=logits[0].device,
    )
    entropy_parts = []
    for index, logits_j in enumerate(logits):
        if tuple(logits_j.shape[:-1]) != expected_prefix:
            raise ValueError(
                "factorized logits prefixes differ: "
                f"{tuple(logits_j.shape[:-1])} vs {expected_prefix}"
            )
        entropy_parts.append(_masked_entropy_rows(logits_j, valid))
    return torch.stack(entropy_parts, dim=-1).sum(dim=-1)


def factorized_normalized_entropy(
    logits: Sequence[torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return equal-FSQ-dimension mean entropy normalized to ``[0, 1]``.

    Each factorized scalar classifier is normalized by its own ``log(C_k)``
    before the scalar dimensions are averaged.  This keeps scalars with
    different codebook sizes comparable and makes the result a ranking score,
    not a calibrated probability of image-space error.
    """

    if not logits:
        raise ValueError("factorized_normalized_entropy requires logits")
    expected_prefix: tuple[int, ...] | None = None
    valid: torch.Tensor | None = None
    normalized_parts: list[torch.Tensor] = []
    for index, logits_j in enumerate(logits):
        if logits_j.ndim < 2:
            raise ValueError(
                f"logits[{index}] must end in a class dimension, got "
                f"{tuple(logits_j.shape)}"
            )
        prefix = tuple(logits_j.shape[:-1])
        if expected_prefix is None:
            expected_prefix = prefix
            valid = _expanded_token_mask(
                mask,
                prefix,
                device=logits_j.device,
            )
        elif prefix != expected_prefix:
            raise ValueError(
                f"factorized logits prefixes differ: {prefix} vs {expected_prefix}"
            )
        num_classes = int(logits_j.shape[-1])
        if num_classes <= 1:
            raise ValueError(f"logits[{index}] must have at least two classes")
        entropy = _masked_entropy_rows(logits_j, valid)
        normalized_parts.append(entropy / math.log(num_classes))
    return torch.stack(normalized_parts, dim=-1).mean(dim=-1).clamp(0.0, 1.0)


def block_reduce(
    token_values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is None:
        return token_values.mean(dim=-1)
    valid = _expanded_token_mask(
        mask,
        token_values.shape,
        device=token_values.device,
    )
    assert valid is not None
    safe_values = torch.where(
        valid,
        token_values,
        torch.zeros((), device=token_values.device, dtype=token_values.dtype),
    )
    denom = valid.to(dtype=token_values.dtype).sum(dim=-1).clamp_min(1.0)
    return safe_values.sum(dim=-1) / denom
