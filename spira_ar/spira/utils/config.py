"""Portable configuration: relative paths are rooted at this release directory."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATH_KEYS = {
    "output_dir",
    "data_root",
    "latent_root",
    "train_manifest",
    "val_manifest",
    "persistent_cache_dir",
    "train_datalist",
    "val_datalist",
    "levels_json",
    "repo_root",
    "config_path",
    "ckpt_path",
}


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    data = cfg.get("data", {})
    fields = {
        k: str(data.get(k, k)).lower() if k == "dataset" else str(data.get(k, k))
        for k in ("dataset", "source_modality", "target_modality")
    }

    def resolve(value, key=""):
        if isinstance(value, dict):
            return {k: resolve(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, str) and key in PATH_KEYS:
            for name, replacement in fields.items():
                value = value.replace("{" + name + "}", replacement)
            p = Path(value).expanduser()
            return str(p if p.is_absolute() else PROJECT_ROOT / p)
        return value

    cfg = resolve(cfg)
    cfg["_config_path"] = str(path)
    return cfg


def training_contract(cfg):
    """Settings that must remain unchanged when resuming an interrupted run."""
    result = copy.deepcopy(
        {
            k: cfg.get(k)
            for k in ("data", "augmentation", "model", "train", "spira", "vidtok")
        }
    )
    result["seed"] = cfg.get("experiment", {}).get("seed", 42)
    return result
