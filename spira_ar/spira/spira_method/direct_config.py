"""Configuration of the fixed zero-regret retraction objective."""

from __future__ import annotations

CONTRACT = "spira_zero_regret_retraction_v2"


def resolve_spira_config(cfg, *, max_steps=None):
    method = cfg.get("spira", {})
    if set(method) - {"enabled", "gate"}:
        raise ValueError("SPIRA accepts only enabled and gate configuration fields")
    if not method.get("enabled", False):
        raise ValueError("spira.enabled must be true")
    gate = dict(method.get("gate", {}))
    if set(gate) != {"version", "hidden_dim"} or gate["version"] != "v2":
        raise ValueError("spira.gate requires version: v2 and hidden_dim")
    gate["hidden_dim"] = int(gate["hidden_dim"])
    if gate["hidden_dim"] < 1:
        raise ValueError("Gate hidden_dim must be positive")
    steps = int(max_steps or cfg["train"]["max_steps"])
    if steps < 1:
        raise ValueError("train.max_steps must be positive")
    if cfg.get("model", {}).get("order", "raster_dhw") != "raster_dhw":
        raise ValueError("SPIRA requires raster_dhw order")
    return {"enabled": True, "contract": CONTRACT, "gate": gate, "max_steps": steps}
