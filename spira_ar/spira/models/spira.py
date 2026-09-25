from __future__ import annotations
import copy
from collections.abc import Callable, Sequence
from typing import Any
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from spira_ar.spira.models.block_patchify3d import BlockPatchify3D
from spira_ar.spira.models.factorized_fsq_head import FactorizedFSQHead
from spira_ar.spira.models.inter_block_llamagen import (
    InterBlockCausalLlama,
    InterBlockConfig,
)
from spira_ar.spira.models.intra_block_biattn import (
    IntraBlockBiAttn,
    TargetBlockEmbedder,
)
from spira_ar.spira.models.next_block_predictor import NextBlockPredictor
from spira_ar.spira.models.source_block_encoder import SourceBlockEncoder
from spira_ar.spira.models.source_conditioned_bov import SourceConditionedBOVBlock
from spira_ar.spira.utils.raster_order import intra_block_coords_tensor
from spira_ar.spira.spira_method.direct_components import SpiraRetractionGateHead
from spira_ar.spira.spira_method.direct_config import resolve_spira_config
from spira_ar.spira.spira_method.direct_target import (
    NUMERIC_TOL,
    ZeroRegretRetractionTarget,
    factorized_forward_kl,
    factorized_retraction_kl,
    mix_retracted_factorized_logits,
    positive_regret_diagnostics,
    solve_zero_regret_retraction,
)
from spira_ar.spira.spira_method.factorized import (
    PairedFactorizedRollout,
    mix_factorized_logits,
    paired_factorized_logits_and_predictions,
)
from spira_ar.spira.utils.nll_entropy import (
    block_reduce,
    factorized_cross_entropy,
    factorized_entropy,
    factorized_normalized_entropy,
)


class SPIRA(nn.Module):
    """Source-Prefix Intervention-aware Residual Autoregression.

    The same raster predictor defines both nested endpoints:

        hS_i = q(S, empty_history, i)
        hF_i = q(S, H_<i, i)

    and one shared factorized FSQ head produces ``zS`` and ``zF``.  The gate
    directly predicts correction retraction ``s_i,u`` for every spatial latent
    token, shared across that token's FSQ scalar dimensions, and generation uses

        z_i,u = zF_i,u - s_i,u * (zF_i,u - zS_i,u).

    Hence ``s_i=0`` is the exact standard raster endpoint and ``s_i=1`` is the
    source-only endpoint. Gate input, endpoint logits, and the zero-regret
    oracle are detached, so the forward-KL supervision updates only the gate.
    """

    def __init__(
        self,
        *,
        levels: Sequence[int],
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        latent_shape_dhw: Sequence[int] = (24, 24, 24),
        block_size_dhw: Sequence[int] = (4, 4, 4),
        intra_num_layers: int = 1,
        intra_num_heads: int = 8,
        source_local_num_layers: int = 1,
        source_local_num_heads: int | None = None,
        source_global_num_layers: int = 4,
        source_global_num_heads: int | None = None,
        source_relative_position_bias: bool = True,
        next_block_local_num_layers: int = 1,
        next_block_local_num_heads: int | None = None,
        dropout: float = 0.05,
        label_smoothing: float = 0.0,
        current_token_gate_cfg: dict[str, Any] | None = None,
        ordinal_loss_cfg: dict[str, Any] | None = None,
        bov_init_cfg: dict[str, Any] | None = None,
        target_history_cfg: dict[str, Any] | None = None,
        gradient_checkpointing: bool = False,
        spira_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        if not isinstance(gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean")
        self.gradient_checkpointing = gradient_checkpointing
        self.levels = [int(v) for v in levels]
        self.hidden_dim = int(hidden_dim)
        self.patchifier = BlockPatchify3D(latent_shape_dhw, block_size_dhw)
        self.num_blocks = self.patchifier.num_blocks
        self.tokens_per_block = self.patchifier.tokens_per_block
        coords = self.patchifier.coords()
        self.register_buffer("block_coords", coords, persistent=False)
        self.register_buffer(
            "block_indices",
            torch.arange(self.num_blocks, dtype=torch.long),
            persistent=False,
        )
        block_size = torch.tensor(self.patchifier.block_size_dhw, dtype=torch.long)
        intra_coords = intra_block_coords_tensor(self.patchifier.block_size_dhw)
        token_coords = coords[:, None, :] * block_size.view(
            1, 1, 3
        ) + intra_coords.view(1, self.tokens_per_block, 3)
        history_token_coords = torch.cat([token_coords[:1], token_coords[:-1]], dim=0)
        history_block_coords = torch.cat([coords[:1], coords[:-1]], dim=0)
        self.register_buffer("block_token_coords", token_coords, persistent=False)
        self.register_buffer(
            "history_token_coords", history_token_coords, persistent=False
        )
        self.register_buffer(
            "history_block_coords", history_block_coords, persistent=False
        )
        bov_cfg = bov_init_cfg or {}
        self.use_source_conditioned_bov = bool(bov_cfg.get("enabled", False))
        history_cfg = target_history_cfg or {}
        history_mode = str(history_cfg.get("mode", "block_summary")).lower()
        if history_mode != "block_summary":
            raise ValueError("target_history.mode must be block_summary")
        self.target_history_mode = history_mode
        source_local_num_heads = (
            intra_num_heads
            if source_local_num_heads is None
            else int(source_local_num_heads)
        )
        source_global_num_heads = (
            num_heads
            if source_global_num_heads is None
            else int(source_global_num_heads)
        )
        next_block_local_num_heads = (
            intra_num_heads
            if next_block_local_num_heads is None
            else int(next_block_local_num_heads)
        )
        self.source_encoder = SourceBlockEncoder(
            self.levels,
            hidden_dim,
            self.tokens_per_block,
            block_grid_dhw=self.patchifier.block_grid_dhw,
            local_num_layers=source_local_num_layers,
            local_num_heads=source_local_num_heads,
            global_num_layers=source_global_num_layers,
            global_num_heads=source_global_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.target_embedder = TargetBlockEmbedder(
            self.levels, hidden_dim, self.tokens_per_block, dropout=dropout
        )
        self.intra_block = IntraBlockBiAttn(
            hidden_dim=hidden_dim,
            num_heads=intra_num_heads,
            num_layers=intra_num_layers,
            dropout=dropout,
        )
        inter_cfg = InterBlockConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            block_grid_dhw=self.patchifier.block_grid_dhw,
            source_relative_position_bias=source_relative_position_bias,
        )
        self.inter_block = InterBlockCausalLlama(inter_cfg)
        self.bov_builder = (
            SourceConditionedBOVBlock(
                hidden_dim, self.tokens_per_block, self.patchifier.latent_shape_dhw
            )
            if self.use_source_conditioned_bov
            else None
        )
        gate_cfg = current_token_gate_cfg
        if gate_cfg is None:
            gate_cfg = {
                "enabled": True,
                "learnable": False,
                "init": 1.0,
                "activation": "sigmoid",
            }
        self.next_block_predictor = NextBlockPredictor(
            hidden_dim,
            self.tokens_per_block,
            local_num_layers=next_block_local_num_layers,
            local_num_heads=next_block_local_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            current_token_gate_enabled=bool(gate_cfg.get("enabled", True)),
            current_token_gate_learnable=bool(gate_cfg.get("learnable", False)),
            current_token_gate_init=float(gate_cfg.get("init", 1.0)),
            current_token_gate_activation=str(gate_cfg.get("activation", "sigmoid")),
        )
        ordinal_cfg = ordinal_loss_cfg or {}
        self.fsq_head = FactorizedFSQHead(
            self.levels,
            hidden_dim,
            label_smoothing=label_smoothing,
            ordinal_loss_enabled=bool(ordinal_cfg.get("enabled", False)),
            ordinal_loss_weight=float(ordinal_cfg.get("weight", 0.0)),
            ordinal_loss_type=str(ordinal_cfg.get("type", "l1")),
            ordinal_value_mapping=str(ordinal_cfg.get("value_mapping", "vidtok_fsq")),
            ordinal_reduction=str(ordinal_cfg.get("reduction", "masked_mean")),
        )
        if not self.use_source_conditioned_bov:
            raise ValueError("SPIRA requires source_conditioned_bov.enabled=true")
        self.spira_cfg = copy.deepcopy(spira_cfg)
        self.spira_gate_head = SpiraRetractionGateHead(
            self.hidden_dim, gate_hidden_dim=int(spira_cfg["gate"]["hidden_dim"])
        )
        self._spira_training_step = int(spira_cfg["max_steps"])
        self._spira_evaluation_diagnostics = False
        self._spira_generation_valid_token_mask = None
        self._latest_spira_summary = {}
        self._latest_spira_gradient_summary = {}

    def _apply_retraction_mode(
        self,
        predicted_sufficiency: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        *,
        first_block_is_b0: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted = predicted_sufficiency.clone()
        if predicted.ndim != 3:
            raise ValueError(
                f"predicted_sufficiency must be [B,N,V], got {tuple(predicted.shape)}"
            )
        if valid_token_mask is None:
            valid_token = torch.ones_like(predicted, dtype=torch.bool)
        else:
            if tuple(valid_token_mask.shape) != tuple(predicted.shape):
                raise ValueError(
                    f"valid_token_mask must match predicted sufficiency [B,N,V]: {tuple(valid_token_mask.shape)} vs {tuple(predicted.shape)}"
                )
            valid_token = valid_token_mask.to(device=predicted.device, dtype=torch.bool)
        predicted = predicted.masked_fill(~valid_token, 0.0)
        active = predicted
        predicted_alpha = (1.0 - predicted).masked_fill(~valid_token, 0.0)
        active_alpha = (1.0 - active).masked_fill(~valid_token, 0.0)
        return (predicted, active, predicted_alpha, active_alpha)

    @staticmethod
    def _attach_target_diagnostics(
        out: dict[str, Any], target: ZeroRegretRetractionTarget
    ) -> None:
        out.update(
            {
                "spira_s_star": target.s_star,
                "spira_s_star_nll_block": target.selected_nll,
                "spira_s_star_nll_token": target.selected_nll_token,
                "spira_full_nll_block": target.full_nll,
                "spira_full_nll_token": target.full_nll_token,
                "spira_valid_token_count_block": target.valid_token_count,
                "spira_oracle_is_training_only": True,
            }
        )

    def _checkpointed_call(
        self,
        function: Callable[..., Any],
        /,
        *args: Any,
        checkpoint_enabled: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Run a pure region with non-reentrant activation checkpointing.

        Integer FSQ inputs do not require gradients, so the reentrant variant
        is not valid here.  The training/grad guards also keep EMA teachers,
        validation, inference, and stateful KV-cache generation on their
        ordinary forward paths.
        """
        if not (
            checkpoint_enabled
            and self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return function(*args, **kwargs)
        return activation_checkpoint(
            function, *args, use_reentrant=False, preserve_rng_state=True, **kwargs
        )

    def _empty_history_contexts(
        self,
        *,
        source_memory: torch.Tensor,
        source_tokens: torch.Tensor,
        coords: torch.Tensor,
        indices: torch.Tensor,
        token_coords: torch.Tensor,
        history_coords: torch.Tensor,
        history_layout,
    ) -> torch.Tensor:
        cache = getattr(self, "_actionability_source_context_cache", None)
        cache_enabled = (
            cache is not None and (not self.training) and (not torch.is_grad_enabled())
        )
        if (
            cache_enabled
            and cache.get("memory") is source_memory
            and (cache.get("tokens") is source_tokens)
            and torch.equal(cache["indices"], indices)
        ):
            return cache["contexts"]
        batch_size = source_memory.shape[0]
        empty_blocks = torch.zeros(
            batch_size,
            self.num_blocks - 1,
            self.tokens_per_block,
            len(self.levels),
            device=source_memory.device,
            dtype=torch.long,
        )
        empty_mask = torch.zeros(
            batch_size,
            self.num_blocks - 1,
            self.tokens_per_block,
            device=source_memory.device,
            dtype=torch.bool,
        )
        contexts, _ = self._unified_prediction_contexts(
            source_memory=source_memory,
            source_tokens=source_tokens,
            conditioning_blocks=empty_blocks,
            conditioning_valid_token_mask=empty_mask,
            coords=coords,
            indices=indices,
            token_coords=token_coords,
            history_coords=history_coords,
            history_layout=history_layout,
            return_history_attention_stats=False,
        )
        if cache_enabled:
            cache.update(
                memory=source_memory,
                tokens=source_tokens,
                indices=indices.detach().clone(),
                contexts=contexts,
            )
        return contexts

    def _gate_from_source_context(
        self,
        source_context: torch.Tensor,
        endpoint_residual: torch.Tensor | None = None,
        valid_token_mask: torch.Tensor | None = None,
        *,
        source_logits: Sequence[torch.Tensor],
        full_logits: Sequence[torch.Tensor],
        first_block_is_b0: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Predict sufficiency from the source anchor and prefix correction.

        The positional ``(source_context, valid_token_mask)`` form remains
        supported for inherited diagnostics and endpoint diagnostics.
        The active correction-aware path passes the detached hidden residual
        ``h_full - h_source`` explicitly.
        """
        if (
            endpoint_residual is not None
            and endpoint_residual.ndim == 3
            and (endpoint_residual.dtype == torch.bool)
            and (valid_token_mask is None)
        ):
            valid_token_mask = endpoint_residual
            endpoint_residual = None
        if endpoint_residual is None:
            endpoint_residual = torch.zeros_like(source_context)
        full_context = source_context + endpoint_residual
        raw_sufficiency = self.spira_gate_head(
            source_context, full_context, source_logits, full_logits, valid_token_mask
        )
        predicted_sufficiency, active_sufficiency, predicted_alpha, active_alpha = (
            self._apply_retraction_mode(
                raw_sufficiency, valid_token_mask, first_block_is_b0=first_block_is_b0
            )
        )
        return (
            predicted_sufficiency,
            predicted_alpha,
            active_alpha,
            {
                "retraction_predicted": predicted_sufficiency,
                "retraction_active": active_sufficiency,
                "history_coefficient_predicted": predicted_alpha,
                "history_coefficient_active": active_alpha,
            },
        )

    @torch.no_grad()
    def _generate_with_unified_bov(
        self,
        *,
        source_memory: torch.Tensor,
        source_tokens: torch.Tensor,
        target_blocks_gt: torch.Tensor | None,
        coords: torch.Tensor,
        indices: torch.Tensor,
        token_coords: torch.Tensor,
        history_coords: torch.Tensor,
        history_layout,
        source_modality_id: torch.Tensor,
        target_modality_id: torch.Tensor,
        greedy: bool,
        temperature: float,
        use_kv_cache: bool,
        compute_source_only: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del compute_source_only, source_modality_id, target_modality_id
        batch_size = source_memory.shape[0]
        device = source_memory.device
        source_context_all = self._empty_history_contexts(
            source_memory=source_memory,
            source_tokens=source_tokens,
            coords=coords,
            indices=indices,
            token_coords=token_coords,
            history_coords=history_coords,
            history_layout=history_layout,
        )
        generation_mask = self._spira_generation_valid_token_mask
        conditioning_mask = (
            torch.ones(
                batch_size,
                self.num_blocks,
                self.tokens_per_block,
                device=device,
                dtype=torch.bool,
            )
            if generation_mask is None
            else generation_mask.to(device=device, dtype=torch.bool)
        )
        need_diagnostics = bool(
            self._spira_evaluation_diagnostics or target_blocks_gt is not None
        )
        bov_hidden = self.build_bov_hidden(
            source_memory,
            source_tokens,
            first_block_index=indices[0],
            first_token_coords=token_coords[0],
        )
        generated: list[torch.Tensor] = []
        history_hidden: list[torch.Tensor] = []
        history_summaries: list[torch.Tensor] = []
        history_valid_tokens: list[torch.Tensor] = []
        mixed_nll_tokens: list[torch.Tensor] = []
        source_nll_tokens: list[torch.Tensor] = []
        full_nll_tokens: list[torch.Tensor] = []
        entropy_tokens: list[torch.Tensor] = []
        normalized_entropy_tokens: list[torch.Tensor] = []
        endpoint_residuals: list[torch.Tensor] = []
        gate_field_steps: dict[str, list[torch.Tensor]] = {}
        endpoint_effect_kl_tokens: list[torch.Tensor] = []
        actual_retraction_tokens: list[torch.Tensor] = []
        self_cache = None
        source_cache = None
        if use_kv_cache:
            self_cache = self.inter_block.init_kv_cache(
                batch_size, self.num_blocks, device=device, dtype=source_memory.dtype
            )
            source_cache = self.inter_block.build_source_kv_cache(source_memory)
        for step in range(self.num_blocks):
            if step == 0:
                current_hidden = bov_hidden
                current_valid = torch.ones(
                    batch_size, self.tokens_per_block, device=device, dtype=torch.bool
                )
            else:
                history_block = generated[-1]
                history_controller = getattr(self, "_actionability_controller", None)
                if history_controller is not None:
                    history_block = history_controller.condition_history(
                        history_block, step
                    )
                embedded = self.target_embedder(history_block[:, None])[:, 0]
                current_valid = conditioning_mask[:, step - 1]
                current_hidden = self.intra_block(
                    embedded, token_valid_mask=current_valid
                )
            current_summary = self.intra_block.pool(
                current_hidden, token_valid_mask=current_valid
            )
            history_hidden.append(current_hidden)
            history_summaries.append(current_summary)
            history_valid_tokens.append(current_valid)
            stacked_history_valid = torch.stack(history_valid_tokens, dim=1)
            if use_kv_cache:
                block_context = self.inter_block.forward_step(
                    current_summary,
                    history_coords[step],
                    source_memory,
                    source_coords=self.block_coords.to(device=device),
                    source_query_coord=coords[step],
                    self_kv_cache=self_cache,
                    source_kv_cache=source_cache,
                    history_valid_mask=stacked_history_valid.any(dim=-1),
                )
            else:
                block_context = self._run_inter_block(
                    torch.stack(history_summaries, dim=1),
                    source_memory,
                    history_coords[: step + 1],
                    source_coords=self.block_coords.to(device=device),
                    source_query_coords=coords[: step + 1],
                    valid_block_mask=stacked_history_valid.any(dim=-1),
                )[:, -1]
            full_context = self._run_next_block_predictor(
                block_context,
                coords[step : step + 1],
                current_block_hidden=current_hidden,
                current_coords=history_coords[step : step + 1],
                source_memory=source_memory,
                source_tokens=source_tokens,
                next_block_indices=indices[step : step + 1],
                coord_max=coords.max(dim=0).values,
            )[:, 0]
            source_context = source_context_all[:, step]
            if step == 0:
                source_context = full_context
            endpoint_residual = full_context.float() - source_context.float()
            if need_diagnostics:
                endpoint_residuals.append(endpoint_residual.detach())
            step_valid_token_mask = conditioning_mask[:, step : step + 1]
            gt_block = None if target_blocks_gt is None else target_blocks_gt[:, step]
            endpoint_probe = paired_factorized_logits_and_predictions(
                self.fsq_head,
                source_context,
                full_context,
                torch.ones_like(
                    step_valid_token_mask[:, 0], dtype=source_context.dtype
                ),
                gt_block,
                conditioning="predicted",
                greedy=True,
                temperature=temperature,
            )
            (
                _predicted_sufficiency_step,
                _predicted_alpha_step,
                active_alpha_step,
                gate_fields_step,
            ) = self._gate_from_source_context(
                source_context[:, None],
                endpoint_residual[:, None],
                step_valid_token_mask,
                source_logits=[
                    value[:, None] for value in endpoint_probe.source_logits
                ],
                full_logits=[value[:, None] for value in endpoint_probe.full_logits],
                first_block_is_b0=step == 0,
            )
            del _predicted_sufficiency_step, _predicted_alpha_step
            actionability = getattr(self, "_actionability_controller", None)
            if actionability is not None:
                selected = actionability.select_action(
                    model=self,
                    step=step,
                    source_context=source_context,
                    full_context=full_context,
                    probe=endpoint_probe,
                    valid=step_valid_token_mask[:, 0],
                    target=gt_block,
                    predicted_s=gate_fields_step["retraction_predicted"][:, 0],
                    temperature=temperature,
                )
                if selected.shape != step_valid_token_mask[:, 0].shape:
                    raise ValueError("Actionability action must be [B,V]")
                if not bool(torch.isfinite(selected).all()) or bool(
                    ((selected < 0) | (selected > 1)).any()
                ):
                    raise ValueError("Actionability action must be finite in [0,1]")
                selected = selected.masked_fill(~step_valid_token_mask[:, 0], 0)
                active_alpha_step = (1 - selected[:, None]).masked_fill(
                    ~step_valid_token_mask, 0
                )
                gate_fields_step["retraction_active"] = selected[:, None]
                gate_fields_step["history_coefficient_active"] = active_alpha_step
            for name, value in gate_fields_step.items():
                gate_field_steps.setdefault(name, []).append(value)
            rollout = paired_factorized_logits_and_predictions(
                self.fsq_head,
                source_context,
                full_context,
                active_alpha_step[:, 0],
                gt_block,
                conditioning="predicted",
                greedy=greedy,
                temperature=temperature,
            )
            if actionability is not None:
                actionability.observe(rollout=rollout, greedy=greedy)

            def scaled_logits(items: Sequence[torch.Tensor]) -> list[torch.Tensor]:
                return [item.float() / max(float(temperature), 1e-06) for item in items]

            mixed_scaled = scaled_logits(rollout.mixed_logits)
            full_scaled = scaled_logits(rollout.full_logits)
            output_valid = conditioning_mask[:, step]
            actual_retraction_tokens.append(
                factorized_forward_kl(full_scaled, mixed_scaled, output_valid)
            )
            generated.append(rollout.predictions)
            if need_diagnostics:
                score_target = rollout.predictions if gt_block is None else gt_block
                source_scaled = scaled_logits(rollout.source_logits)
                endpoint_effect_kl_tokens.append(
                    factorized_forward_kl(full_scaled, source_scaled, output_valid)
                )
                _, mixed_nll = factorized_cross_entropy(
                    mixed_scaled, score_target, mask=output_valid, label_smoothing=0.0
                )
                _, source_nll = factorized_cross_entropy(
                    source_scaled, score_target, mask=output_valid, label_smoothing=0.0
                )
                _, full_nll = factorized_cross_entropy(
                    full_scaled, score_target, mask=output_valid, label_smoothing=0.0
                )
                mixed_nll_tokens.append(mixed_nll)
                source_nll_tokens.append(source_nll)
                full_nll_tokens.append(full_nll)
                entropy_tokens.append(
                    factorized_entropy(mixed_scaled, mask=output_valid)
                )
                normalized_entropy_tokens.append(
                    factorized_normalized_entropy(mixed_scaled, mask=output_valid)
                )
        target_blocks = torch.stack(generated, dim=1)
        gate_output_fields = {
            name: torch.cat(values, dim=1) for name, values in gate_field_steps.items()
        }
        actual_retraction_token = torch.stack(actual_retraction_tokens, dim=1)
        diagnostic_valid = conditioning_mask.to(device=device, dtype=torch.bool)
        actual_retraction_token = torch.where(
            diagnostic_valid,
            actual_retraction_token,
            torch.zeros_like(actual_retraction_token),
        )
        stats: dict[str, Any] = {
            "spira_gate_granularity": "latent_token",
            "actual_retraction_token": actual_retraction_token,
            "actual_retraction_block": block_reduce(
                actual_retraction_token, diagnostic_valid
            ),
            "spira_valid_token_mask": diagnostic_valid,
            **gate_output_fields,
            "source_only_nll_contract": "same_spira_predictor_empty_vs_generated_prefix",
            "full_endpoint_contract": "exact_standard_raster_predictor",
        }
        if need_diagnostics:
            nll_token = torch.stack(mixed_nll_tokens, dim=1)
            source_nll_token = torch.stack(source_nll_tokens, dim=1)
            full_nll_token = torch.stack(full_nll_tokens, dim=1)
            entropy_token = torch.stack(entropy_tokens, dim=1)
            normalized_entropy_token = torch.stack(normalized_entropy_tokens, dim=1)
            endpoint_effect_kl_token = torch.stack(endpoint_effect_kl_tokens, dim=1)
            endpoint_effect_kl_token = torch.where(
                diagnostic_valid,
                endpoint_effect_kl_token,
                torch.zeros_like(endpoint_effect_kl_token),
            )
            endpoint_residual = torch.stack(endpoint_residuals, dim=1)
            residual_diagnostics = self._residual_norm_diagnostics(
                endpoint_residual, source_context_all, diagnostic_valid
            )
            stats.update(
                {
                    "nll_token": nll_token,
                    "nll_block": block_reduce(nll_token, diagnostic_valid),
                    "entropy_token": entropy_token,
                    "entropy_block": block_reduce(entropy_token, diagnostic_valid),
                    "normalized_entropy_token": normalized_entropy_token,
                    "normalized_entropy_block": block_reduce(
                        normalized_entropy_token, diagnostic_valid
                    ),
                    "prediction_risk_token": normalized_entropy_token,
                    "prediction_risk_block": block_reduce(
                        normalized_entropy_token, diagnostic_valid
                    ),
                    "nll_kind": "self" if target_blocks_gt is None else "ground_truth",
                    "nll_source_only_token": source_nll_token,
                    "nll_source_only_block": block_reduce(
                        source_nll_token, diagnostic_valid
                    ),
                    "nll_full_history_token": full_nll_token,
                    "nll_full_history_block": block_reduce(
                        full_nll_token, diagnostic_valid
                    ),
                    "delta_nll_token": source_nll_token - full_nll_token,
                    "delta_nll_block": block_reduce(source_nll_token, diagnostic_valid)
                    - block_reduce(full_nll_token, diagnostic_valid),
                    "endpoint_effect_kl_token": endpoint_effect_kl_token,
                    "endpoint_effect_kl_block": block_reduce(
                        endpoint_effect_kl_token, diagnostic_valid
                    ),
                    **residual_diagnostics,
                }
            )
        return (target_blocks, stats)

    def _loss_parts(
        self,
        logits: Sequence[torch.Tensor],
        targets: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        *,
        label_smoothing: float,
        include_ordinal: bool,
        include_distribution_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        init_mask = None if valid_token_mask is None else valid_token_mask[:, 0]
        next_mask = None if valid_token_mask is None else valid_token_mask[:, 1:]
        logits_init = [item[:, 0] for item in logits]
        logits_next = [item[:, 1:] for item in logits]
        ce, nll_token = factorized_cross_entropy(
            logits,
            targets,
            mask=valid_token_mask,
            label_smoothing=float(label_smoothing),
        )
        ce_init, _ = factorized_cross_entropy(
            logits_init,
            targets[:, 0],
            mask=init_mask,
            label_smoothing=float(label_smoothing),
        )
        ce_next, _ = factorized_cross_entropy(
            logits_next,
            targets[:, 1:],
            mask=next_mask,
            label_smoothing=float(label_smoothing),
        )
        if include_ordinal and self.fsq_head.ordinal_loss_enabled:
            ordinal = self.fsq_head.ordinal_loss(logits, targets, mask=valid_token_mask)
            ordinal_init = self.fsq_head.ordinal_loss(
                logits_init, targets[:, 0], mask=init_mask
            )
            ordinal_next = self.fsq_head.ordinal_loss(
                logits_next, targets[:, 1:], mask=next_mask
            )
            ordinal_weight = self.fsq_head.ordinal_loss_weight
        else:
            ordinal = ce.new_zeros(())
            ordinal_init = ce_init.new_zeros(())
            ordinal_next = ce_next.new_zeros(())
            ordinal_weight = 0.0
        loss = ce + ordinal_weight * ordinal
        loss_init = ce_init + ordinal_weight * ordinal_init
        loss_next = ce_next + ordinal_weight * ordinal_next
        parts = {
            "loss": loss,
            "loss_ce": ce,
            "loss_ordinal": ordinal,
            "loss_init": loss_init,
            "loss_next": loss_next,
            "loss_init_ce": ce_init,
            "loss_next_ce": ce_next,
            "loss_init_ordinal": ordinal_init,
            "loss_next_ordinal": ordinal_next,
            "nll_token": nll_token,
        }
        if include_distribution_diagnostics:
            parts["entropy_token"] = factorized_entropy(logits, mask=valid_token_mask)
            parts["normalized_entropy_token"] = factorized_normalized_entropy(
                logits, mask=valid_token_mask
            )
        return parts

    def _modality_ids(
        self,
        batch_size: int,
        device,
        source_modality_id: torch.Tensor | None,
        target_modality_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_modality_id is None:
            source_modality_id = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
        else:
            source_modality_id = source_modality_id.to(device=device).long().view(-1)
        if target_modality_id is None:
            target_modality_id = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
        else:
            target_modality_id = target_modality_id.to(device=device).long().view(-1)
        return (source_modality_id, target_modality_id)

    def _ordered_block_layout(self, block_order, device):
        self._require_raster(block_order, device)
        indices = self.block_indices.to(device)
        return (
            indices,
            self.block_coords.to(device),
            self.block_token_coords.to(device),
            self.history_block_coords.to(device),
            None,
        )

    def _paired_endpoint_contexts(
        self,
        *,
        source_memory: torch.Tensor,
        source_tokens: torch.Tensor,
        conditioning_blocks: torch.Tensor,
        conditioning_valid_token_mask: torch.Tensor | None,
        coords: torch.Tensor,
        indices: torch.Tensor,
        token_coords: torch.Tensor,
        history_coords: torch.Tensor,
        history_layout,
        return_history_attention_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        if conditioning_valid_token_mask is None:
            full_mask = torch.ones(
                conditioning_blocks.shape[:3],
                device=conditioning_blocks.device,
                dtype=torch.bool,
            )
        else:
            if tuple(conditioning_valid_token_mask.shape) != tuple(
                conditioning_blocks.shape[:3]
            ):
                raise ValueError(
                    f"conditioning_valid_token_mask must match conditioning blocks [B,N,V]: {tuple(conditioning_valid_token_mask.shape)} vs {tuple(conditioning_blocks.shape[:3])}"
                )
            full_mask = conditioning_valid_token_mask.to(
                device=conditioning_blocks.device, dtype=torch.bool
            )
        null_mask = torch.zeros_like(full_mask)
        common = {
            "source_memory": source_memory,
            "source_tokens": source_tokens,
            "conditioning_blocks": conditioning_blocks,
            "coords": coords,
            "indices": indices,
            "token_coords": token_coords,
            "history_coords": history_coords,
            "history_layout": history_layout,
            "return_history_attention_stats": return_history_attention_stats,
        }
        share_rng = self.training and torch.is_grad_enabled()
        before = self._rng_state_for_device(source_memory.device) if share_rng else None
        full_context, history_attention_stats = self._unified_prediction_contexts(
            **common, conditioning_valid_token_mask=full_mask
        )
        after_full = (
            self._rng_state_for_device(source_memory.device) if share_rng else None
        )
        if before is not None:
            self._set_rng_state_for_device(source_memory.device, before)
        try:
            source_context, _ = self._unified_prediction_contexts(
                **common, conditioning_valid_token_mask=null_mask
            )
        except Exception:
            if after_full is not None:
                self._set_rng_state_for_device(source_memory.device, after_full)
            raise
        if after_full is not None:
            self._set_rng_state_for_device(source_memory.device, after_full)
        source_context = source_context.clone()
        source_context[:, 0] = full_context[:, 0]
        return (source_context, full_context, history_attention_stats)

    def _require_raster(
        self, block_order: torch.Tensor | None, device: torch.device
    ) -> None:
        if block_order is None:
            return
        order = torch.as_tensor(block_order, device=device, dtype=torch.long).reshape(
            -1
        )
        raster = torch.arange(self.num_blocks, device=device, dtype=torch.long)
        if order.shape != raster.shape or not torch.equal(order, raster):
            raise ValueError("SPIRA supports fixed raster_dhw traversal only")

    @staticmethod
    def _residual_norm_diagnostics(
        scaled_residual: torch.Tensor,
        source_context: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if scaled_residual.shape != source_context.shape:
            raise ValueError(
                f"scaled_residual/source_context shapes differ: {tuple(scaled_residual.shape)} vs {tuple(source_context.shape)}"
            )
        if scaled_residual.ndim != 4:
            raise ValueError(
                f"scaled_residual must be [B,N,V,C], got {tuple(scaled_residual.shape)}"
            )
        expected_mask = scaled_residual.shape[:3]
        if valid_token_mask is None:
            valid = torch.ones(
                expected_mask, device=scaled_residual.device, dtype=torch.bool
            )
        else:
            if tuple(valid_token_mask.shape) != tuple(expected_mask):
                raise ValueError(
                    f"valid_token_mask must match residual [B,N,V]: {tuple(valid_token_mask.shape)} vs {tuple(expected_mask)}"
                )
            valid = valid_token_mask.to(device=scaled_residual.device, dtype=torch.bool)
        residual_float = scaled_residual.detach().float()
        source_float = source_context.detach().float()
        residual_norm_token = residual_float.square().mean(dim=-1).sqrt()
        source_norm_token = source_float.square().mean(dim=-1).sqrt()
        ratio_token = residual_norm_token / source_norm_token.clamp_min(1e-12)
        residual_norm_token = residual_norm_token.masked_fill(~valid, 0.0)
        ratio_token = ratio_token.masked_fill(~valid, 0.0)
        weight = valid.to(dtype=torch.float32).unsqueeze(-1)
        residual_squared_sum = (residual_float.square() * weight).sum()
        source_squared_sum = (source_float.square() * weight).sum()
        valid_feature_count = weight.sum() * float(scaled_residual.shape[-1])
        denominator = valid_feature_count.clamp_min(1.0)
        residual_rms = (residual_squared_sum / denominator).sqrt()
        source_rms = (source_squared_sum / denominator).sqrt()
        return {
            "sufficiency_scaled_residual_norm_token": residual_norm_token,
            "sufficiency_scaled_residual_norm_block": block_reduce(
                residual_norm_token, valid
            ),
            "sufficiency_scaled_residual_to_source_norm_ratio_token": ratio_token,
            "sufficiency_scaled_residual_to_source_norm_ratio_block": block_reduce(
                ratio_token, valid
            ),
            "sufficiency_scaled_residual_to_source_norm_ratio": residual_rms
            / source_rms.clamp_min(1e-12),
            "sufficiency_scaled_residual_squared_sum": residual_squared_sum,
            "sufficiency_source_context_squared_sum": source_squared_sum,
            "sufficiency_valid_feature_count": valid_feature_count,
            "sufficiency_valid_token_mask": valid,
        }

    def _resolve_source_features(
        self,
        source_blocks: torch.Tensor,
        *,
        source_memory: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate prepared source features or encode them on demand."""
        if (source_memory is None) != (source_tokens is None):
            raise ValueError(
                "source_memory and source_tokens must be provided together"
            )
        if source_memory is None or source_tokens is None:
            return self.encode_source(source_blocks)
        if source_blocks.ndim != 4:
            raise ValueError(
                f"Expected source_blocks [B,N,V,d], got {tuple(source_blocks.shape)}"
            )
        batch_size, num_blocks, tokens_per_block, _ = source_blocks.shape
        expected_memory = (batch_size, num_blocks, self.hidden_dim)
        expected_tokens = (batch_size, num_blocks, tokens_per_block, self.hidden_dim)
        if tuple(source_memory.shape) != expected_memory:
            raise ValueError(
                f"Prepared source_memory must have shape {expected_memory}, got {tuple(source_memory.shape)}"
            )
        if tuple(source_tokens.shape) != expected_tokens:
            raise ValueError(
                f"Prepared source_tokens must have shape {expected_tokens}, got {tuple(source_tokens.shape)}"
            )
        if (
            source_memory.device != source_blocks.device
            or source_tokens.device != source_blocks.device
        ):
            raise ValueError(
                "Prepared source features must be on the same device as source_blocks"
            )
        if source_memory.dtype != source_tokens.dtype:
            raise ValueError(
                "Prepared source_memory and source_tokens must use the same dtype"
            )
        return (source_memory, source_tokens)

    @staticmethod
    def _rng_state_for_device(
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        return (cpu_state, cuda_state)

    def _run_inter_block(
        self,
        summaries: torch.Tensor,
        source_memory: torch.Tensor,
        coords: torch.Tensor,
        *,
        source_coords: torch.Tensor | None = None,
        source_query_coords: torch.Tensor | None = None,
        valid_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._checkpointed_call(
            self.inter_block,
            summaries,
            source_memory,
            coords,
            source_coords=source_coords,
            source_query_coords=source_query_coords,
            valid_block_mask=valid_block_mask,
        )

    def _run_next_block_predictor(
        self, block_context: torch.Tensor, next_coords: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return self._checkpointed_call(
            self.next_block_predictor, block_context, next_coords, **kwargs
        )

    @staticmethod
    def _set_rng_state_for_device(
        device: torch.device, state: tuple[torch.Tensor, torch.Tensor | None]
    ) -> None:
        cpu_state, cuda_state = state
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)

    def _should_compute_spira_diagnostics(self) -> bool:
        """Return whether this forward should materialize analysis-only tensors."""
        return bool(
            self._spira_evaluation_diagnostics
            or not self.training
            or (not torch.is_grad_enabled())
            or (not self._latest_spira_summary)
            or ((self._spira_training_step + 1) % 200 == 0)
        )

    def _unified_prediction_contexts(
        self,
        *,
        source_memory: torch.Tensor,
        source_tokens: torch.Tensor,
        conditioning_blocks: torch.Tensor,
        coords: torch.Tensor,
        indices: torch.Tensor,
        token_coords: torch.Tensor,
        history_coords: torch.Tensor,
        history_layout: Any | None,
        conditioning_valid_token_mask: torch.Tensor | None = None,
        return_history_attention_stats: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Predict all blocks through one ``BOV/Bg -> B0/B[g+1]`` path."""
        if not self.use_source_conditioned_bov:
            raise RuntimeError(
                "Unified BOV prediction was requested while source-conditioned BOV is disabled"
            )
        if conditioning_blocks.shape[1] != self.num_blocks - 1:
            raise ValueError(
                f"Expected {self.num_blocks - 1} conditioning blocks, got {conditioning_blocks.shape[1]}"
            )
        if conditioning_valid_token_mask is not None:
            expected = conditioning_blocks.shape[:3]
            if tuple(conditioning_valid_token_mask.shape) != tuple(expected):
                raise ValueError(
                    f"conditioning_valid_token_mask must match conditioning blocks [B,N,V]: {tuple(conditioning_valid_token_mask.shape)} vs {tuple(expected)}"
                )
            conditioning_valid_token_mask = conditioning_valid_token_mask.to(
                device=conditioning_blocks.device, dtype=torch.bool
            )
        bov_hidden = self.build_bov_hidden(
            source_memory,
            source_tokens,
            first_block_index=indices[0],
            first_token_coords=token_coords[0],
        )
        bov_valid = torch.ones(
            bov_hidden.shape[:2], device=bov_hidden.device, dtype=torch.bool
        )
        if conditioning_blocks.shape[1] > 0:
            target_hidden, _ = self.summarize_target_blocks(
                conditioning_blocks, conditioning_valid_token_mask
            )
            history_hidden = torch.cat([bov_hidden[:, None], target_hidden], dim=1)
            history_valid_token_mask = torch.cat(
                [
                    bov_valid[:, None],
                    torch.ones(
                        conditioning_blocks.shape[:3],
                        device=conditioning_blocks.device,
                        dtype=torch.bool,
                    )
                    if conditioning_valid_token_mask is None
                    else conditioning_valid_token_mask,
                ],
                dim=1,
            )
        else:
            history_hidden = bov_hidden[:, None]
            history_valid_token_mask = bov_valid[:, None]
        history_summaries = self.intra_block.pool(
            history_hidden, token_valid_mask=history_valid_token_mask
        )
        inter_context = self._run_inter_block(
            history_summaries,
            source_memory,
            history_coords,
            source_coords=self.block_coords.to(device=source_memory.device),
            source_query_coords=coords,
            valid_block_mask=history_valid_token_mask.any(dim=-1),
        )
        history_attention_stats = None
        prediction_context = self._run_next_block_predictor(
            inter_context,
            coords,
            current_block_hidden=history_hidden,
            current_coords=history_coords,
            source_memory=source_memory,
            source_tokens=source_tokens,
            next_block_indices=indices,
            coord_max=coords.max(dim=0).values,
        )
        return (prediction_context, history_attention_stats)

    def _update_spira_log_summary(
        self,
        *,
        full_loss: torch.Tensor,
        source_loss: torch.Tensor,
        mixed_loss: torch.Tensor,
        retraction_kl: torch.Tensor,
        total_loss: torch.Tensor,
        target: ZeroRegretRetractionTarget,
        predicted_retraction: torch.Tensor,
        endpoint_effect_kl: torch.Tensor,
        oracle_actual_retraction_kl: torch.Tensor,
        learned_actual_retraction_kl: torch.Tensor,
        oracle_to_learned_kl: torch.Tensor,
        regret: dict[str, torch.Tensor],
        identity_logit_max_abs_error: torch.Tensor,
        first_unit_endpoint_error: torch.Tensor,
    ) -> None:
        valid = target.valid_mask
        truth = target.s_star[valid].float()
        prediction = predicted_retraction[valid].float()
        zero = total_loss.detach() * 0.0
        if truth.numel():
            target_quantiles = torch.quantile(
                truth, truth.new_tensor([0.05, 0.5, 0.95])
            )
            prediction_quantiles = torch.quantile(
                prediction, prediction.new_tensor([0.05, 0.5, 0.95])
            )
            target_mean = truth.mean()
            target_std = truth.std(unbiased=False)
            prediction_mean = prediction.mean()
            prediction_std = prediction.std(unbiased=False)
        else:
            target_quantiles = prediction_quantiles = torch.stack([zero, zero, zero])
            target_mean = target_std = prediction_mean = prediction_std = zero

        def valid_mean(value: torch.Tensor) -> torch.Tensor:
            selected = value.masked_select(valid)
            return selected.mean() if selected.numel() else zero

        self._latest_spira_summary = {
            "training_mode": "spira_zero_regret_retraction",
            "gate_granularity": "latent_token",
            "gate_input": "stopgrad_source_and_full_minus_source",
            "gate_architecture": "mlp_2h_plus_2_to_hidden_to_1_sigmoid",
            "gate_parameter_count": sum(
                (parameter.numel() for parameter in self.spira_gate_head.parameters())
            ),
            "oracle": "maximal_zero_regret_fixed_10_step_bisection",
            "numeric_tolerance": NUMERIC_TOL,
            "gate_loss": "factorized_forward_kl_oracle_to_learned",
            "valid_gate_token_count": int(valid.sum().detach().cpu()),
            "training_step": self._spira_training_step,
            "loss_full": full_loss.detach(),
            "loss_source": source_loss.detach(),
            "loss_retraction_kl": retraction_kl.detach(),
            "loss_mix_log_only": mixed_loss.detach(),
            "loss_total": total_loss.detach(),
            "s_star_mean": target_mean.detach(),
            "s_star_std": target_std.detach(),
            "s_star_q05": target_quantiles[0].detach(),
            "s_star_q50": target_quantiles[1].detach(),
            "s_star_q95": target_quantiles[2].detach(),
            "s_hat_mean": prediction_mean.detach(),
            "s_hat_std": prediction_std.detach(),
            "s_hat_q05": prediction_quantiles[0].detach(),
            "s_hat_q50": prediction_quantiles[1].detach(),
            "s_hat_q95": prediction_quantiles[2].detach(),
            "endpoint_effect_kl_mean": valid_mean(endpoint_effect_kl).detach(),
            "oracle_actual_retraction_kl_mean": valid_mean(
                oracle_actual_retraction_kl
            ).detach(),
            "learned_actual_retraction_kl_mean": valid_mean(
                learned_actual_retraction_kl
            ).detach(),
            "oracle_to_learned_kl": valid_mean(oracle_to_learned_kl).detach(),
            "positive_regret_rate": regret["positive_regret_rate"].detach(),
            "mean_positive_regret": regret["mean_positive_regret"].detach(),
            "identity_logit_max_abs_error": identity_logit_max_abs_error.detach(),
            "first_unit_endpoint_error": first_unit_endpoint_error.detach(),
        }

    def build_bov_hidden(
        self,
        source_memory: torch.Tensor,
        source_tokens: torch.Tensor,
        *,
        first_block_index: torch.Tensor,
        first_token_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Build source-conditioned BOV tokens and run the shared intra-block encoder."""
        if self.bov_builder is None:
            raise RuntimeError("source-conditioned BOV is disabled")

        def _build_bov(memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
            embedded = self.bov_builder(
                memory, tokens, first_token_coords, source_block_index=first_block_index
            )
            return self.intra_block(embedded)

        return self._checkpointed_call(_build_bov, source_memory, source_tokens)

    def enable_spira_evaluation_diagnostics(self, enabled: bool = True) -> None:
        self._spira_evaluation_diagnostics = bool(enabled)

    def encode_source(self, source_blocks: torch.Tensor):
        source_memory, source_tokens = self._checkpointed_call(
            self.source_encoder,
            source_blocks,
            block_coords=self.block_coords,
            return_tokens=True,
        )
        return (source_memory, source_tokens)

    def forward(
        self,
        source_fsq_scalars: torch.Tensor,
        target_fsq_scalars: torch.Tensor,
        *,
        valid_token_mask: torch.Tensor | None = None,
        source_modality_id: torch.Tensor | None = None,
        target_modality_id: torch.Tensor | None = None,
        compute_source_only: bool = False,
        return_logits: bool = False,
        return_history_attention_stats: bool = False,
        block_order: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        source_blocks = self.patchifier.patchify(source_fsq_scalars.long())
        target_blocks = self.patchifier.patchify(target_fsq_scalars.long())
        return self.forward_from_blocks(
            source_blocks,
            target_blocks,
            valid_token_mask=valid_token_mask,
            source_modality_id=source_modality_id,
            target_modality_id=target_modality_id,
            compute_source_only=compute_source_only,
            return_logits=return_logits,
            return_history_attention_stats=return_history_attention_stats,
            block_order=block_order,
        )

    def forward_from_blocks(
        self,
        source_blocks: torch.Tensor,
        target_blocks: torch.Tensor,
        *,
        valid_token_mask: torch.Tensor | None = None,
        source_modality_id: torch.Tensor | None = None,
        target_modality_id: torch.Tensor | None = None,
        compute_source_only: bool = False,
        return_logits: bool = False,
        return_history_attention_stats: bool = False,
        block_order: torch.Tensor | None = None,
        source_memory: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del compute_source_only
        self._require_raster(block_order, target_blocks.device)
        if source_blocks.shape != target_blocks.shape:
            raise ValueError("source/target block shapes differ")
        b, n, v, scalar_dim = target_blocks.shape
        if (
            n != self.num_blocks
            or v != self.tokens_per_block
            or scalar_dim != len(self.levels)
        ):
            raise ValueError(
                f"Expected [B,{self.num_blocks},{self.tokens_per_block},{len(self.levels)}], got {tuple(target_blocks.shape)}"
            )
        device = target_blocks.device
        indices, coords, token_coords, history_coords, history_layout = (
            self._ordered_block_layout(None, device)
        )
        source_modality_id, target_modality_id = self._modality_ids(
            b, device, source_modality_id, target_modality_id
        )
        source_memory, source_tokens = self._resolve_source_features(
            source_blocks, source_memory=source_memory, source_tokens=source_tokens
        )
        conditioning_blocks = target_blocks[:, :-1]
        del source_modality_id, target_modality_id
        source_context, full_context, history_attention_stats = (
            self._paired_endpoint_contexts(
                source_memory=source_memory,
                source_tokens=source_tokens,
                conditioning_blocks=conditioning_blocks,
                conditioning_valid_token_mask=None
                if valid_token_mask is None
                else valid_token_mask[:, :-1],
                coords=coords,
                indices=indices,
                token_coords=token_coords,
                history_coords=history_coords,
                history_layout=history_layout,
                return_history_attention_stats=return_history_attention_stats,
            )
        )
        endpoint_residual = full_context.float() - source_context.float()
        endpoint_residual[:, 0] = 0.0
        probe_alpha = torch.ones(
            target_blocks.shape[:-1],
            device=source_context.device,
            dtype=source_context.dtype,
        )
        endpoint_rollout = paired_factorized_logits_and_predictions(
            self.fsq_head,
            source_context,
            full_context,
            probe_alpha,
            target_blocks,
            conditioning="teacher",
            greedy=True,
            temperature=1.0,
        )
        predicted_retraction, _active_retraction, active_alpha, gate_fields = (
            self._gate_from_source_context(
                source_context,
                endpoint_residual,
                valid_token_mask,
                source_logits=endpoint_rollout.source_logits,
                full_logits=endpoint_rollout.full_logits,
            )
        )
        mixed_logits = mix_factorized_logits(
            endpoint_rollout.source_logits, endpoint_rollout.full_logits, active_alpha
        )
        rollout = PairedFactorizedRollout(
            source_logits=endpoint_rollout.source_logits,
            full_logits=endpoint_rollout.full_logits,
            mixed_logits=mixed_logits,
            predictions=torch.stack(
                [value.argmax(dim=-1) for value in mixed_logits], dim=-1
            ),
        )
        smoothing = float(self.fsq_head.label_smoothing)
        include_ordinal = bool(self.fsq_head.ordinal_loss_enabled)
        source_parts = self._loss_parts(
            rollout.source_logits,
            target_blocks,
            valid_token_mask,
            label_smoothing=smoothing,
            include_ordinal=include_ordinal,
        )
        full_parts = self._loss_parts(
            rollout.full_logits,
            target_blocks,
            valid_token_mask,
            label_smoothing=smoothing,
            include_ordinal=include_ordinal,
        )
        retraction_target = solve_zero_regret_retraction(
            rollout.source_logits,
            rollout.full_logits,
            target_blocks,
            valid_token_mask=valid_token_mask,
        )
        oracle_logits = mix_retracted_factorized_logits(
            rollout.source_logits,
            rollout.full_logits,
            retraction_target.s_star,
            detach_endpoints=True,
            valid_token_mask=retraction_target.valid_mask,
        )
        learned_logits = mix_retracted_factorized_logits(
            rollout.source_logits,
            rollout.full_logits,
            predicted_retraction,
            detach_endpoints=True,
            valid_token_mask=retraction_target.valid_mask,
        )
        retraction_kl, oracle_to_learned_kl = factorized_retraction_kl(
            oracle_logits, learned_logits, retraction_target.valid_mask
        )
        need_diagnostics = self._should_compute_spira_diagnostics() or bool(
            return_logits
        )
        endpoint_loss = 0.5 * (full_parts["loss"] + source_parts["loss"])
        total_loss = endpoint_loss + retraction_kl
        source_nll = source_parts["nll_token"].detach()
        full_nll = full_parts["nll_token"].detach()
        out: dict[str, Any] = {
            "spira_gate_granularity": "latent_token",
            "loss": total_loss,
            "loss_ar": endpoint_loss,
            "loss_init": full_parts["loss_init"],
            "loss_next": full_parts["loss_next"],
            "loss_init_ce": full_parts["loss_init_ce"],
            "loss_next_ce": full_parts["loss_next_ce"],
            "loss_init_ordinal": full_parts["loss_init_ordinal"],
            "loss_next_ordinal": full_parts["loss_next_ordinal"],
            "loss_full": full_parts["loss"],
            "loss_source": source_parts["loss"],
            "loss_source_only": source_parts["loss"],
            "loss_source_only_ce": source_parts["loss_ce"],
            "loss_source_only_ordinal": source_parts["loss_ordinal"],
            "loss_retraction_kl": retraction_kl,
            "nll_source_only_token": source_nll,
            "nll_source_only_block": block_reduce(source_nll, valid_token_mask),
            "nll_full_history_token": full_nll,
            "nll_full_history_block": block_reduce(full_nll, valid_token_mask),
            "delta_nll_token": source_nll - full_nll,
            "delta_nll_block": block_reduce(source_nll, valid_token_mask)
            - block_reduce(full_nll, valid_token_mask),
            "oracle_to_learned_kl_token": oracle_to_learned_kl.detach(),
            "s_star": retraction_target.s_star,
            "s_hat": predicted_retraction,
            "spira_valid_mask": retraction_target.valid_mask,
            "spira_valid_block_mask": retraction_target.valid_block_mask,
            "spira_valid_token_mask": retraction_target.valid_mask,
            "spira_full_nll_token": retraction_target.full_nll_token,
            "spira_oracle_nll_token": retraction_target.selected_nll_token,
            **gate_fields,
            "source_memory": source_memory,
            "current_token_gate": self.next_block_predictor.current_token_gate_value().detach(),
        }
        if need_diagnostics:
            endpoint_effect_kl = factorized_forward_kl(
                rollout.full_logits, rollout.source_logits, retraction_target.valid_mask
            ).detach()
            oracle_actual_retraction_kl = factorized_forward_kl(
                rollout.full_logits, oracle_logits, retraction_target.valid_mask
            ).detach()
            learned_actual_retraction_kl = factorized_forward_kl(
                rollout.full_logits, learned_logits, retraction_target.valid_mask
            ).detach()
            mixed_parts = self._loss_parts(
                [value.detach() for value in learned_logits],
                target_blocks,
                valid_token_mask,
                label_smoothing=0.0,
                include_ordinal=False,
                include_distribution_diagnostics=True,
            )
            regret = positive_regret_diagnostics(
                learned_logits,
                target_blocks,
                retraction_target.full_nll_token,
                retraction_target.valid_mask,
            )
            identity_logits = mix_retracted_factorized_logits(
                rollout.source_logits,
                rollout.full_logits,
                torch.zeros_like(predicted_retraction),
                detach_endpoints=True,
                valid_token_mask=retraction_target.valid_mask,
            )
            identity_logit_max_abs_error = max(
                (
                    (identity.float() - full.detach().float()).abs().max()
                    for identity, full in zip(identity_logits, rollout.full_logits)
                )
            )
            first_unit_endpoint_error = max(
                (
                    (source[:, 0].detach().float() - full[:, 0].detach().float())
                    .abs()
                    .max()
                    for source, full in zip(rollout.source_logits, rollout.full_logits)
                )
            )
            residual_diagnostics = self._residual_norm_diagnostics(
                endpoint_residual, source_context, valid_token_mask
            )
            mixed_nll = mixed_parts["nll_token"].detach()
            mixed_entropy = mixed_parts["entropy_token"].detach()
            normalized_entropy = mixed_parts["normalized_entropy_token"].detach()
            out.update(
                {
                    "loss_mix": mixed_parts["loss"].detach(),
                    "nll_token": mixed_nll,
                    "nll_block": block_reduce(mixed_nll, valid_token_mask),
                    "entropy_token": mixed_entropy,
                    "entropy_block": block_reduce(mixed_entropy, valid_token_mask),
                    "normalized_entropy_token": normalized_entropy,
                    "normalized_entropy_block": block_reduce(
                        normalized_entropy, valid_token_mask
                    ),
                    "prediction_risk_token": normalized_entropy,
                    "prediction_risk_block": block_reduce(
                        normalized_entropy, valid_token_mask
                    ),
                    "endpoint_effect_kl_token": endpoint_effect_kl,
                    "endpoint_effect_kl_block": block_reduce(
                        endpoint_effect_kl, retraction_target.valid_mask
                    ),
                    "oracle_actual_retraction_token": oracle_actual_retraction_kl,
                    "learned_actual_retraction_token": learned_actual_retraction_kl,
                    "actual_retraction_token": learned_actual_retraction_kl,
                    "actual_retraction_block": block_reduce(
                        learned_actual_retraction_kl, retraction_target.valid_mask
                    ),
                    "positive_regret_rate": regret["positive_regret_rate"],
                    "mean_positive_regret": regret["mean_positive_regret"],
                    "positive_regret_token": regret["regret_token"].clamp_min(0.0),
                    "identity_logit_max_abs_error": identity_logit_max_abs_error,
                    "first_unit_endpoint_error": first_unit_endpoint_error,
                    "prefix_correction_hidden": endpoint_residual.detach(),
                    **residual_diagnostics,
                }
            )
            self._update_spira_log_summary(
                full_loss=full_parts["loss"],
                source_loss=source_parts["loss"],
                mixed_loss=mixed_parts["loss"],
                retraction_kl=retraction_kl,
                total_loss=total_loss,
                target=retraction_target,
                predicted_retraction=predicted_retraction,
                endpoint_effect_kl=endpoint_effect_kl,
                oracle_actual_retraction_kl=oracle_actual_retraction_kl,
                learned_actual_retraction_kl=learned_actual_retraction_kl,
                oracle_to_learned_kl=oracle_to_learned_kl,
                regret=regret,
                identity_logit_max_abs_error=identity_logit_max_abs_error,
                first_unit_endpoint_error=first_unit_endpoint_error,
            )
        if history_attention_stats is not None:
            out["target_history_attention_mass"] = history_attention_stats
        if return_logits:
            out["logits_b0"] = [item[:, 0] for item in learned_logits]
            out["logits_next"] = [item[:, 1:] for item in learned_logits]
            out["logits_source"] = rollout.source_logits
            out["logits_full"] = rollout.full_logits
            out["logits_mixed"] = [item.detach() for item in learned_logits]
        if self._spira_evaluation_diagnostics:
            self._attach_target_diagnostics(out, retraction_target)
        return out

    @torch.no_grad()
    def generate(
        self,
        source_fsq_scalars: torch.Tensor,
        *,
        target_fsq_scalars: torch.Tensor | None = None,
        source_modality_id: torch.Tensor | None = None,
        target_modality_id: torch.Tensor | None = None,
        greedy: bool = True,
        temperature: float = 1.0,
        use_kv_cache: bool = True,
        compute_source_only: bool = False,
        block_order: torch.Tensor | None = None,
        valid_token_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        source_blocks = self.patchifier.patchify(source_fsq_scalars.long())
        target_blocks_gt = (
            None
            if target_fsq_scalars is None
            else self.patchifier.patchify(target_fsq_scalars.long())
        )
        target_blocks, stats = self.generate_from_source_blocks(
            source_blocks,
            target_blocks_gt=target_blocks_gt,
            source_modality_id=source_modality_id,
            target_modality_id=target_modality_id,
            greedy=greedy,
            temperature=temperature,
            use_kv_cache=use_kv_cache,
            compute_source_only=compute_source_only,
            block_order=block_order,
            valid_token_mask=valid_token_mask,
        )
        return {
            "target_fsq_scalars": self.patchifier.unpatchify(target_blocks),
            "target_blocks": target_blocks,
            **stats,
        }

    @torch.no_grad()
    def generate_from_source_blocks(
        self,
        source_blocks: torch.Tensor,
        *,
        target_blocks_gt: torch.Tensor | None = None,
        source_modality_id: torch.Tensor | None = None,
        target_modality_id: torch.Tensor | None = None,
        greedy: bool = True,
        temperature: float = 1.0,
        use_kv_cache: bool = True,
        compute_source_only: bool = False,
        source_memory: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
        block_order: torch.Tensor | None = None,
        valid_token_mask=None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self._require_raster(block_order, source_blocks.device)
        if valid_token_mask is not None:
            if tuple(valid_token_mask.shape) != tuple(source_blocks.shape[:3]):
                raise ValueError("valid_token_mask must match source blocks [B,N,V]")
            valid_token_mask = valid_token_mask.to(
                device=source_blocks.device, dtype=torch.bool
            )
        previous_mask = self._spira_generation_valid_token_mask
        self._spira_generation_valid_token_mask = valid_token_mask
        try:
            if (
                target_blocks_gt is not None
                and target_blocks_gt.shape != source_blocks.shape
            ):
                raise ValueError(
                    f"source_blocks and target_blocks_gt shapes differ: {source_blocks.shape} vs {target_blocks_gt.shape}"
                )
            b = source_blocks.shape[0]
            device = source_blocks.device
            indices, coords, token_coords, history_coords, history_layout = (
                self._ordered_block_layout(block_order, device)
            )
            source_modality_id, target_modality_id = self._modality_ids(
                b, device, source_modality_id, target_modality_id
            )
            source_memory, source_tokens = self._resolve_source_features(
                source_blocks, source_memory=source_memory, source_tokens=source_tokens
            )
            target_blocks, stats = self._generate_with_unified_bov(
                source_memory=source_memory,
                source_tokens=source_tokens,
                target_blocks_gt=target_blocks_gt,
                coords=coords,
                indices=indices,
                token_coords=token_coords,
                history_coords=history_coords,
                history_layout=history_layout,
                source_modality_id=source_modality_id,
                target_modality_id=target_modality_id,
                greedy=greedy,
                temperature=temperature,
                use_kv_cache=use_kv_cache,
                compute_source_only=compute_source_only,
            )
            return (target_blocks, stats)
        finally:
            self._spira_generation_valid_token_mask = previous_mask

    def record_spira_gradient_norms(self, summary: dict[str, float]) -> None:
        self._latest_spira_gradient_summary = dict(summary)

    def set_spira_training_step(self, step: int) -> None:
        """Set continuous coefficient-ramp progress without changing trainability."""
        self._spira_training_step = max(int(step), 0)

    def spira_log_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            **self._latest_spira_summary,
            **self._latest_spira_gradient_summary,
        }
        serializable: dict[str, Any] = {}
        for key, value in summary.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    serializable[key] = float(value.detach().cpu())
            elif isinstance(value, (str, int, float, bool)) or value is None:
                serializable[key] = value
        return serializable

    def summarize_target_blocks(
        self, target_blocks: torch.Tensor, valid_token_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        def _summarize(
            blocks: torch.Tensor, token_mask: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            embedded = self.target_embedder(blocks)
            hidden = self.intra_block(embedded, token_valid_mask=token_mask)
            return (hidden, self.intra_block.pool(hidden, token_valid_mask=token_mask))

        return self._checkpointed_call(_summarize, target_blocks, valid_token_mask)


def build_spira_from_config(cfg: dict[str, Any], levels: Sequence[int]) -> SPIRA:
    resolved = resolve_spira_config(
        cfg, max_steps=int(cfg.get("train", {}).get("max_steps", 0)) or None
    )
    if not bool(resolved.get("enabled", False)):
        raise ValueError("The SPIRA builder requires spira.enabled=true")
    model_cfg = cfg.get("model", {})
    intra_cfg = model_cfg.get("intra_block_biattn", {})
    source_cfg = model_cfg.get("source_encoder", {})
    next_cfg = model_cfg.get("next_block_predictor", {})
    fsq_cfg = model_cfg.get("fsq_factorized_head", {})
    train_cfg = cfg.get("train", {})
    dropout_cfg = train_cfg.get("dropout", {})
    gradient_checkpointing = train_cfg.get("gradient_checkpointing", False)
    if not isinstance(gradient_checkpointing, bool):
        raise ValueError("train.gradient_checkpointing must be a boolean")
    model = SPIRA(
        levels=levels,
        hidden_dim=int(model_cfg.get("hidden_dim", 768)),
        num_layers=int(model_cfg.get("num_layers", 12)),
        num_heads=int(model_cfg.get("num_heads", 12)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        latent_shape_dhw=tuple(
            model_cfg.get(
                "latent_shape",
                cfg.get("data", {}).get("latent_crop_size", (24, 24, 24)),
            )
        ),
        block_size_dhw=tuple(model_cfg.get("block_size", (4, 4, 4))),
        intra_num_layers=int(intra_cfg.get("num_layers", 1)),
        intra_num_heads=int(intra_cfg.get("num_heads", 8)),
        source_local_num_layers=int(source_cfg.get("local_num_layers", 1)),
        source_local_num_heads=int(
            source_cfg.get("local_num_heads", intra_cfg.get("num_heads", 8))
        ),
        source_global_num_layers=int(source_cfg.get("global_num_layers", 4)),
        source_global_num_heads=int(
            source_cfg.get("global_num_heads", model_cfg.get("num_heads", 12))
        ),
        source_relative_position_bias=bool(
            source_cfg.get("relative_3d_position_bias", True)
        ),
        next_block_local_num_layers=int(next_cfg.get("local_decoder_num_layers", 1)),
        next_block_local_num_heads=int(
            next_cfg.get("local_decoder_num_heads", intra_cfg.get("num_heads", 8))
        ),
        dropout=float(model_cfg.get("dropout", dropout_cfg.get("resid", 0.05))),
        label_smoothing=float(fsq_cfg.get("label_smoothing", 0.0)),
        current_token_gate_cfg=next_cfg.get("current_token_gate"),
        ordinal_loss_cfg=fsq_cfg.get("ordinal_loss", {}),
        bov_init_cfg=model_cfg.get("source_conditioned_bov", {}),
        target_history_cfg=model_cfg.get("target_history", {}),
        gradient_checkpointing=gradient_checkpointing,
        spira_cfg=resolved,
    )
    model.spira_model_config = copy.deepcopy(model_cfg)
    return model
