"""Frozen VidTok-FSQ runtime with strict encoder/decoder weight loading."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import nn


class FrozenVidTok(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        from vidtok.modules.util import instantiate_from_config

        params = architecture.model.params
        self.encoder = instantiate_from_config(params.encoder_config)
        self.decoder = instantiate_from_config(params.decoder_config)
        self.regularization = instantiate_from_config(params.regularizer_config)

    def encode(self, x, return_reg_log=False):
        z, log = self.regularization(self.encoder(x), n_steps=0)
        return (z, log) if return_reg_log else z

    def decode(self, z, decode_from_indices=False):
        if decode_from_indices:
            from einops import rearrange
            from vidtok.modules.regularizers import pack_one, unpack_one

            indices, shape = pack_one(rearrange(z, "... -> ... 1"), "b * d")
            codes = self.regularization.indices_to_codes(indices)
            codes = rearrange(codes, "b d n c -> b n (c d)")
            z = unpack_one(self.regularization.project_out(codes), shape, "b * d")
            z = rearrange(z, "b ... d -> b d ...")
        return self.decoder(z)


class MedicalVidTokWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


def load_medical_wrapper(cfg, ckpt_override=None, lightweight_loss=True):
    del lightweight_loss
    from omegaconf import OmegaConf

    settings = cfg["vidtok"]
    repo = Path(settings["repo_root"]).expanduser().resolve()
    if not (repo / "vidtok").is_dir():
        raise FileNotFoundError(
            "Set vidtok.repo_root to the public VidTok source checkout"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    importlib.invalidate_caches()
    checkpoint = Path(ckpt_override or settings["ckpt_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Tokenizer checkpoint does not exist: {checkpoint}")
    architecture = OmegaConf.load(settings["config_path"])
    model = FrozenVidTok(architecture)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint))
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state.get("model", state))
    expected = model.state_dict()
    selected = {}
    for name, value in state.items():
        for prefix in ("module.model.", "module.", "model.", ""):
            if name.startswith(prefix) and name[len(prefix) :] in expected:
                selected[name[len(prefix) :]] = value
                break
    # Missing or incompatible tokenizer tensors are errors, never random weights.
    model.load_state_dict(selected, strict=True)
    return MedicalVidTokWrapper(model).eval().requires_grad_(False)
