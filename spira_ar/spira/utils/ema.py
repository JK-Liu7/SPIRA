from __future__ import annotations
import copy
from collections import OrderedDict
from typing import Any
import torch


class ModelEma:
    """Frozen model exponential moving average for validation and inference."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {self.decay}")
        self.num_updates = 0
        self.module = copy.deepcopy(model)
        self._freeze_module()

    def _freeze_module(self) -> None:
        self.module.eval()
        self.module.requires_grad_(False)

    @property
    def averaged_model(self) -> torch.nn.Module:
        """Return the frozen forward-capable averaged model."""
        self._freeze_module()
        return self.module

    @torch.no_grad()
    def copy_from(self, model: torch.nn.Module) -> None:
        self.module.load_state_dict(model.state_dict(), strict=True)
        self._freeze_module()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        student_state = model.state_dict()
        teacher_state = self.module.state_dict()
        if student_state.keys() != teacher_state.keys():
            missing = sorted(set(student_state) - set(teacher_state))
            unexpected = sorted(set(teacher_state) - set(student_state))
            raise RuntimeError(
                f"EMA student/teacher state keys differ: missing_in_teacher={missing}, unexpected_in_teacher={unexpected}"
            )
        for name, value in student_state.items():
            teacher_value = teacher_state[name]
            if teacher_value.shape != value.shape:
                raise RuntimeError(
                    f"EMA tensor shape mismatch for {name}: teacher={tuple(teacher_value.shape)} student={tuple(value.shape)}"
                )
            if teacher_value.device != value.device:
                raise RuntimeError(
                    f"EMA teacher/student devices differ for {name}: teacher={teacher_value.device} student={value.device}"
                )
            if not torch.is_floating_point(value):
                teacher_value.copy_(value.detach())
                continue
            if teacher_value.dtype != value.dtype:
                raise RuntimeError(
                    f"EMA teacher/student dtypes differ for {name}: teacher={teacher_value.dtype} student={value.dtype}"
                )
            teacher_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
        self.num_updates += 1
        self._freeze_module()

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (
                (name, tensor.detach().cpu().clone())
                for name, tensor in self.module.state_dict().items()
            )
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            isinstance(state, dict)
            and "shadow" in state
            and isinstance(state["shadow"], dict)
        ):
            self.decay = float(state.get("decay", self.decay))
            self.num_updates = int(state.get("num_updates", self.num_updates))
            state = state["shadow"]
        if not isinstance(state, dict):
            raise TypeError(f"EMA state must be a mapping, got {type(state).__name__}")
        self.module.load_state_dict(state, strict=True)
        self._freeze_module()


def build_ema_from_config(
    model: torch.nn.Module, cfg: dict[str, Any]
) -> tuple[ModelEma | None, dict[str, Any]]:
    ema_cfg = cfg.get("train", {}).get("ema", {}) if isinstance(cfg, dict) else {}
    if ema_cfg is None:
        ema_cfg = {}
    if not isinstance(ema_cfg, dict):
        raise TypeError(f"train.ema must be a mapping, got {type(ema_cfg).__name__}")
    enabled = bool(ema_cfg.get("enabled", False))
    if not enabled:
        return (
            None,
            {
                "enabled": False,
                "decay": None,
                "start_step": None,
                "use_for_validation": False,
            },
        )
    decay = float(ema_cfg.get("decay", 0.999))
    start_step = int(ema_cfg.get("start_step", ema_cfg.get("start_after_step", 0)))
    if start_step < 0:
        raise ValueError(f"EMA start_step must be non-negative, got {start_step}")
    use_for_validation = bool(ema_cfg.get("use_for_validation", False))
    ema = ModelEma(model, decay=decay)
    return (
        ema,
        {
            "enabled": True,
            "decay": decay,
            "start_step": start_step,
            "use_for_validation": use_for_validation,
        },
    )


def validation_model_from_ema(
    model: torch.nn.Module,
    ema: ModelEma | None,
    runtime_cfg: dict[str, Any],
    *,
    step: int,
) -> tuple[torch.nn.Module, bool]:
    """Select EMA only after it has started receiving real updates."""
    use_ema = (
        ema is not None
        and bool(runtime_cfg.get("use_for_validation", False))
        and (int(step) >= int(runtime_cfg.get("start_step") or 0))
        and (int(ema.num_updates) > 0)
    )
    selected = ema.averaged_model if use_ema else model
    set_training_step = getattr(selected, "set_spira_training_step", None)
    if callable(set_training_step):
        set_training_step(int(step))
    return (selected, bool(use_ema))
