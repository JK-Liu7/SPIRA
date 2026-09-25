from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from spira_ar.spira.spira_method.factorized import mix_factorized_logits
from spira_ar.spira.utils.nll_entropy import block_reduce


BISECTION_STEPS = 10
NUMERIC_TOL = 1.0e-6


def _validated_mask(
    targets: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
) -> torch.Tensor:
    expected = targets.shape[:-1]
    if valid_token_mask is None:
        return torch.ones(expected, device=targets.device, dtype=torch.bool)
    if tuple(valid_token_mask.shape) != tuple(expected):
        raise ValueError(
            "valid_token_mask must match target token layout: "
            f"{tuple(valid_token_mask.shape)} vs {tuple(expected)}"
        )
    return valid_token_mask.to(device=targets.device, dtype=torch.bool)


def _validated_layout_mask(
    prefix: tuple[int, ...],
    valid_token_mask: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if valid_token_mask is None:
        return None
    if tuple(valid_token_mask.shape) != prefix:
        raise ValueError(
            "valid_token_mask must match token layout: "
            f"{tuple(valid_token_mask.shape)} vs {prefix}"
        )
    return valid_token_mask.to(device=device, dtype=torch.bool)


def detach_factorized_logits(
    logits: Sequence[torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    if not logits:
        raise ValueError("factorized logits must not be empty")
    return [value.detach().to(dtype=dtype) for value in logits]


def _valid_cross_entropy_rows(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Compute NLL only on valid rows and fill invalid rows with exact zero."""

    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_targets = targets.long().reshape(-1)
    flat_valid = valid.reshape(-1)
    indices = flat_valid.nonzero(as_tuple=False).flatten()
    flat_output = flat_logits.new_zeros(flat_valid.shape)
    if indices.numel() == 0:
        return flat_output.reshape(valid.shape)
    selected_nll = F.cross_entropy(
        flat_logits.index_select(0, indices),
        flat_targets.index_select(0, indices),
        reduction="none",
    )
    return flat_output.index_copy(0, indices, selected_nll).reshape(valid.shape)


def factorized_token_nll(
    logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return equal-factor mean categorical NLL for each latent token.

    Invalid rows are selected out before cross entropy. Consequently sentinel
    target IDs and non-finite padding logits cannot contaminate the oracle or
    regret diagnostics.
    """

    if targets.shape[-1] != len(logits):
        raise ValueError(
            f"targets contain {targets.shape[-1]} factors but received "
            f"{len(logits)} logit tensors"
        )
    valid = _validated_mask(targets, valid_token_mask)
    parts: list[torch.Tensor] = []
    for factor, value in enumerate(logits):
        if value.shape[:-1] != targets.shape[:-1]:
            raise ValueError(
                f"logits[{factor}] prefix {tuple(value.shape[:-1])} does not "
                f"match targets {tuple(targets.shape[:-1])}"
            )
        parts.append(
            _valid_cross_entropy_rows(
                value,
                targets[..., factor],
                valid,
            )
        )
    return torch.stack(parts, dim=-1).mean(dim=-1)


def mix_retracted_factorized_logits(
    source_logits: Sequence[torch.Tensor],
    full_logits: Sequence[torch.Tensor],
    retraction: torch.Tensor,
    *,
    detach_endpoints: bool = False,
    valid_token_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Return ``z_full - s * (z_full - z_source)`` for every FSQ factor.

    When ``valid_token_mask`` is supplied, invalid endpoint rows and gate
    values are replaced before interpolation.  This is stronger than masking
    the downstream loss: it also prevents a zero upstream gradient from being
    multiplied by a non-finite invalid endpoint during autograd.
    """

    if len(source_logits) != len(full_logits) or not source_logits:
        raise ValueError("source/full factorized logit counts differ")
    expected_prefix = tuple(source_logits[0].shape[:-1])
    valid = _validated_layout_mask(
        expected_prefix,
        valid_token_mask,
        device=source_logits[0].device,
    )
    if tuple(retraction.shape) != expected_prefix:
        raise ValueError(
            "retraction must match endpoint token layout: "
            f"{tuple(retraction.shape)} vs {expected_prefix}"
        )
    if valid is not None:
        retraction = torch.where(
            valid,
            retraction,
            torch.zeros_like(retraction),
        )
        expanded_valid = valid.unsqueeze(-1)
        source_logits = [
            torch.where(expanded_valid, value, torch.zeros_like(value))
            for value in source_logits
        ]
        full_logits = [
            torch.where(expanded_valid, value, torch.zeros_like(value))
            for value in full_logits
        ]

    history_coefficient = 1.0 - retraction
    return mix_factorized_logits(
        source_logits,
        full_logits,
        history_coefficient,
        detach_endpoints=detach_endpoints,
    )


@dataclass(frozen=True)
class ZeroRegretRetractionTarget:
    s_star: torch.Tensor
    full_nll_token: torch.Tensor
    selected_nll_token: torch.Tensor
    full_nll: torch.Tensor
    selected_nll: torch.Tensor
    valid_mask: torch.Tensor
    valid_block_mask: torch.Tensor
    valid_token_count: torch.Tensor


@torch.no_grad()
def solve_zero_regret_retraction(
    source_logits: Sequence[torch.Tensor],
    full_logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    steps: int = BISECTION_STEPS,
    numeric_tol: float = NUMERIC_TOL,
) -> ZeroRegretRetractionTarget:
    """Find the largest target-aware retraction with no full-endpoint regret."""

    if int(steps) != BISECTION_STEPS:
        raise ValueError(
            f"SPIRA fixes bisection to {BISECTION_STEPS} steps; got {steps}"
        )
    if float(numeric_tol) != NUMERIC_TOL:
        raise ValueError(f"SPIRA fixes numeric_tol to {NUMERIC_TOL}; got {numeric_tol}")
    valid = _validated_mask(targets, valid_token_mask)
    source = detach_factorized_logits(source_logits)
    full = detach_factorized_logits(full_logits)
    full_nll = factorized_token_nll(full, targets, valid)
    source_nll = factorized_token_nll(source, targets, valid)
    if not bool(torch.isfinite(full_nll.masked_select(valid)).all()) or not bool(
        torch.isfinite(source_nll.masked_select(valid)).all()
    ):
        raise FloatingPointError("SPIRA oracle requires finite valid endpoint NLLs")

    safe_at_one = valid & (source_nll <= full_nll + NUMERIC_TOL)
    low = torch.zeros_like(full_nll)
    high = torch.ones_like(full_nll)
    active = valid & ~safe_at_one
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (low + high)
        mid_logits = mix_retracted_factorized_logits(
            source,
            full,
            mid,
            valid_token_mask=valid,
        )
        mid_nll = factorized_token_nll(mid_logits, targets, valid)
        if not bool(torch.isfinite(mid_nll.masked_select(valid)).all()):
            raise FloatingPointError("SPIRA oracle encountered non-finite valid NLL")
        mid_safe = mid_nll <= full_nll + NUMERIC_TOL
        low = torch.where(active & mid_safe, mid, low)
        high = torch.where(active & ~mid_safe, mid, high)

    s_star = torch.where(safe_at_one, torch.ones_like(low), low)
    s_star = torch.where(valid, s_star, torch.zeros_like(s_star))
    selected_logits = mix_retracted_factorized_logits(
        source,
        full,
        s_star,
        valid_token_mask=valid,
    )
    selected_nll = factorized_token_nll(selected_logits, targets, valid)
    if not bool(torch.isfinite(selected_nll.masked_select(valid)).all()):
        raise FloatingPointError("SPIRA oracle selected a non-finite valid endpoint")
    valid_block = valid.any(dim=-1)
    return ZeroRegretRetractionTarget(
        s_star=s_star.detach(),
        full_nll_token=full_nll.detach(),
        selected_nll_token=selected_nll.detach(),
        full_nll=block_reduce(full_nll, valid).detach(),
        selected_nll=block_reduce(selected_nll, valid).detach(),
        valid_mask=valid,
        valid_block_mask=valid_block,
        valid_token_count=valid.sum(dim=-1),
    )


def factorized_forward_kl(
    teacher_logits: Sequence[torch.Tensor],
    student_logits: Sequence[torch.Tensor],
    valid_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return equal-factor ``KL(teacher || student)`` without token reduction.

    If a mask is supplied, log-softmax is evaluated only for valid rows. This
    prevents NaN/Inf padding logits from entering either the forward value or
    backward graph.
    """

    if len(teacher_logits) != len(student_logits) or not teacher_logits:
        raise ValueError("factorized teacher/student logit counts differ")
    parts: list[torch.Tensor] = []
    expected_prefix: tuple[int, ...] | None = None
    valid: torch.Tensor | None = None
    for factor, (teacher, student) in enumerate(zip(teacher_logits, student_logits)):
        if teacher.shape != student.shape:
            raise ValueError(
                f"factor {factor} teacher/student shapes differ: "
                f"{tuple(teacher.shape)} vs {tuple(student.shape)}"
            )
        prefix = tuple(teacher.shape[:-1])
        if expected_prefix is None:
            expected_prefix = prefix
            valid = _validated_layout_mask(
                prefix,
                valid_token_mask,
                device=teacher.device,
            )
        elif prefix != expected_prefix:
            raise ValueError("factorized logit prefixes differ")

        flat_teacher = teacher.float().reshape(-1, teacher.shape[-1])
        flat_student = student.float().reshape(-1, student.shape[-1])
        if valid is None:
            log_p_teacher = F.log_softmax(flat_teacher, dim=-1)
            p_teacher = log_p_teacher.exp().detach()
            log_p_student = F.log_softmax(flat_student, dim=-1)
            factor_kl = (p_teacher * (log_p_teacher.detach() - log_p_student)).sum(
                dim=-1
            )
            parts.append(factor_kl.reshape(prefix))
            continue

        flat_valid = valid.reshape(-1)
        indices = flat_valid.nonzero(as_tuple=False).flatten()
        flat_output = flat_teacher.new_zeros(flat_valid.shape)
        if indices.numel():
            selected_teacher = flat_teacher.index_select(0, indices)
            selected_student = flat_student.index_select(0, indices)
            log_p_teacher = F.log_softmax(selected_teacher, dim=-1)
            p_teacher = log_p_teacher.exp().detach()
            log_p_student = F.log_softmax(selected_student, dim=-1)
            selected_kl = (p_teacher * (log_p_teacher.detach() - log_p_student)).sum(
                dim=-1
            )
            flat_output = flat_output.index_copy(0, indices, selected_kl)
        parts.append(flat_output.reshape(prefix))
    return torch.stack(parts, dim=-1).mean(dim=-1)


def factorized_retraction_kl(
    oracle_logits: Sequence[torch.Tensor],
    predicted_logits: Sequence[torch.Tensor],
    valid_token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_kl = factorized_forward_kl(
        oracle_logits,
        predicted_logits,
        valid_token_mask,
    )
    if tuple(valid_token_mask.shape) != tuple(token_kl.shape):
        raise ValueError(
            "valid_token_mask must match KL token layout: "
            f"{tuple(valid_token_mask.shape)} vs {tuple(token_kl.shape)}"
        )
    valid = valid_token_mask.to(device=token_kl.device, dtype=torch.bool)
    valid_values = token_kl.masked_select(valid)
    if valid_values.numel():
        loss = valid_values.mean()
    else:
        # Preserve a scalar that is connected to the predicted endpoint graph,
        # while reading no padded logit value. Backward therefore remains valid
        # for a degenerate all-padding local batch and produces exact zero gate
        # gradients.
        loss = predicted_logits[0].reshape(-1)[:0].sum().float()
        for value in predicted_logits[1:]:
            loss = loss + value.reshape(-1)[:0].sum().float()
    return loss, torch.where(valid, token_kl, torch.zeros_like(token_kl))


@torch.no_grad()
def positive_regret_diagnostics(
    predicted_logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    full_nll_token: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = valid_token_mask.to(device=targets.device, dtype=torch.bool)
    predicted_nll = factorized_token_nll(
        [value.detach() for value in predicted_logits],
        targets,
        valid,
    )
    regret = predicted_nll - full_nll_token.detach()
    valid_values = regret.masked_select(valid)
    if valid_values.numel() == 0:
        zero = predicted_nll.new_zeros(())
        rate = mean_positive = zero
    else:
        rate = (valid_values > NUMERIC_TOL).float().mean()
        mean_positive = valid_values.clamp_min(0.0).mean()
    return {
        "predicted_nll_token": predicted_nll.detach(),
        "regret_token": torch.where(
            valid,
            regret,
            torch.zeros_like(regret),
        ).detach(),
        "positive_regret_rate": rate.detach(),
        "mean_positive_regret": mean_positive.detach(),
    }
