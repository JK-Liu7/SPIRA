"""Validate a release checkpoint on deterministic latent crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spira_ar.train_spira import evaluate, make_loader
from spira_ar.spira.models.spira import build_spira_from_config
from spira_ar.spira.utils.config import load_config
from spira_ar.spira.utils.runtime import build_dataset_from_config, levels_from_sample


def load_checkpoint_model(path, cfg, device, use_ema=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    levels = list(checkpoint["fsq_levels"])
    # Architecture comes from the checkpoint; the CLI YAML supplies data/execution settings.
    model = build_spira_from_config(checkpoint["config"], levels).to(device)
    state = checkpoint.get("ema") if use_ema else checkpoint["model"]
    if state is None:
        raise ValueError("Checkpoint has no EMA weights")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, levels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--free-running", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model, levels = load_checkpoint_model(args.ckpt, cfg, device, args.use_ema)
    data = build_dataset_from_config(cfg, split="val", training=False)
    if not len(data) or levels_from_sample(data[0]) != levels:
        raise ValueError("Validation data is empty or has incompatible FSQ levels")
    result = evaluate(
        model,
        make_loader(data, cfg, training=False),
        cfg,
        device,
        max_batches=args.max_batches,
        free_running=args.free_running,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
