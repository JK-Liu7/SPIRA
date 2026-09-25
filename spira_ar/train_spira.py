"""Train SPIRA's shared endpoints and Gate V2 jointly on cached FSQ latents."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spira_ar.spira.models.spira import build_spira_from_config
from spira_ar.spira.utils.config import PROJECT_ROOT, load_config, training_contract
from spira_ar.spira.utils.ema import build_ema_from_config, validation_model_from_ema
from spira_ar.spira.utils.runtime import (
    autocast_context,
    build_dataset_from_config,
    configure_optimizer,
    levels_from_sample,
    lr_at_step,
    move_batch_to_device,
    seed_all,
    read_levels_from_config,
    set_optimizer_lr,
    valid_token_mask_from_batch,
)


def make_loader(dataset, cfg, *, training, epoch=0):
    settings = cfg["train"] if training else cfg.get("validation", {})
    generator = torch.Generator().manual_seed(int(cfg["experiment"]["seed"]) + epoch)
    return DataLoader(
        dataset,
        batch_size=int(settings.get("batch_size_per_gpu", 1)),
        shuffle=training,
        num_workers=int(settings.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


@torch.no_grad()
def evaluate(model, loader, cfg, device, *, max_batches=None, free_running=False):
    was_training = model.training
    model.eval()
    totals = {}
    count = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        mask = valid_token_mask_from_batch(batch, model.patchifier.block_size_dhw)
        weight = int(mask.sum())
        if not weight:
            continue
        with autocast_context(device, cfg["train"].get("precision", "fp32")):
            out = model(
                batch["source_fsq_scalars"],
                batch["target_fsq_scalars"],
                valid_token_mask=mask,
            )
        metrics = {k: float(out[k]) for k in ("loss", "loss_ar", "loss_retraction_kl")}
        for key in ("nll_full_history_token", "nll_source_only_token"):
            metrics[key] = float(out[key][mask].float().mean())
        if free_running:
            with autocast_context(device, cfg["train"].get("precision", "fp32")):
                prediction = model.generate(
                    batch["source_fsq_scalars"],
                    valid_token_mask=mask,
                    greedy=True,
                    use_kv_cache=True,
                )["target_fsq_scalars"]
            pred_blocks = model.patchifier.patchify(prediction)
            truth = model.patchifier.patchify(batch["target_fsq_scalars"])
            metrics["scalar_accuracy"] = float(
                (pred_blocks == truth)[mask].float().mean()
            )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * weight
        count += weight
    model.train(was_training)
    if not count:
        raise RuntimeError("Validation contains no valid latent tokens")
    return {**{k: v / count for k, v in totals.items()}, "valid_tokens": count}


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    ema,
    scaler,
    cfg,
    levels,
    step,
    epoch,
    batch_in_epoch,
    best_loss,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "ema_updates": 0 if ema is None else ema.num_updates,
        "scaler": scaler.state_dict(),
        "config": cfg,
        "fsq_levels": levels,
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "best_loss": best_loss,
        "rng_state": rng_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/brats24.yaml"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop at this optimizer step without changing the configured schedule.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    settings = cfg["train"]
    seed_all(int(cfg["experiment"]["seed"]))
    np.random.seed(int(cfg["experiment"]["seed"]))
    device = torch.device(args.device)
    train_data = build_dataset_from_config(cfg, split="train", training=True)
    val_data = build_dataset_from_config(cfg, split="val", training=False)
    if not len(train_data) or not len(val_data):
        raise RuntimeError(
            "Training and validation manifests must both contain usable modality pairs"
        )
    levels = levels_from_sample(train_data[0])
    configured_levels = read_levels_from_config(cfg)
    if configured_levels is not None and configured_levels != levels:
        raise ValueError("Configured FSQ levels differ from the latent caches")
    model = build_spira_from_config(cfg, levels).to(device)
    optimizer = configure_optimizer(model, cfg)
    ema, ema_settings = build_ema_from_config(model, cfg)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and settings.get("precision") == "fp16"
    )
    output = Path(cfg["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "last.pt"
    if checkpoint_path.exists() and args.resume is None:
        raise FileExistsError(
            "Output already contains last.pt; use --resume or choose a new output_dir"
        )
    step = epoch = batch_in_epoch = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if training_contract(checkpoint["config"]) != training_contract(cfg):
            raise ValueError(
                "Resume requires the same data, model, objective and training configuration"
            )
        if checkpoint["fsq_levels"] != levels:
            raise ValueError("Checkpoint and data FSQ levels differ")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None:
            if checkpoint["ema"] is None:
                raise ValueError("Resume checkpoint has no required EMA state")
            ema.load_state_dict(checkpoint["ema"])
            ema.num_updates = int(checkpoint["ema_updates"])
        step, epoch = int(checkpoint["step"]), int(checkpoint["epoch"])
        batch_in_epoch, best_loss = (
            int(checkpoint["batch_in_epoch"]),
            float(checkpoint["best_loss"]),
        )
        restore_rng(checkpoint["rng_state"])
    (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    endpoint_parameters = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("spira_gate_head.") and p.requires_grad
    ]
    gate_parameters = list(model.spira_gate_head.parameters())
    accum = max(1, int(settings.get("grad_accum_steps", 1)))
    limit = min(
        int(settings["max_steps"]), args.stop_after or int(settings["max_steps"])
    )
    val_loader = make_loader(val_data, cfg, training=False)
    train_data.set_epoch(epoch)
    train_loader = make_loader(train_data, cfg, training=True, epoch=epoch)
    iterator = iter(train_loader)
    for _ in range(batch_in_epoch):
        next(iterator)
    bar = tqdm(total=limit, initial=min(step, limit), desc="SPIRA training")

    def save(path):
        save_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            ema=ema,
            scaler=scaler,
            cfg=cfg,
            levels=levels,
            step=step,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            best_loss=best_loss,
        )

    with (output / "train.jsonl").open("a", encoding="utf-8") as log:
        while step < limit:
            model.train()
            model.set_spira_training_step(step)
            lr = lr_at_step(step, cfg)
            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    batch_in_epoch = 0
                    train_data.set_epoch(epoch)
                    train_loader = make_loader(
                        train_data, cfg, training=True, epoch=epoch
                    )
                    iterator = iter(train_loader)
                    batch = next(iterator)
                batch_in_epoch += 1
                batch = move_batch_to_device(batch, device)
                mask = valid_token_mask_from_batch(
                    batch, model.patchifier.block_size_dhw
                )
                with autocast_context(device, settings.get("precision", "fp32")):
                    result = model(
                        batch["source_fsq_scalars"],
                        batch["target_fsq_scalars"],
                        valid_token_mask=mask,
                    )
                    loss = result["loss"] / accum
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite loss at optimizer step {step}"
                    )
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
            scaler.unscale_(optimizer)
            norms = [
                torch.nn.utils.clip_grad_norm_(
                    params, float(settings["grad_clip"]), error_if_nonfinite=True
                )
                for params in (endpoint_parameters, gate_parameters)
            ]
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if ema is not None:
                if step <= int(ema_settings["start_step"]):
                    ema.copy_from(model)
                else:
                    ema.update(model)
            row = {
                "step": step,
                "loss": loss_sum,
                "lr": lr,
                "gradient_norm_endpoint": float(norms[0]),
                "gradient_norm_gate": float(norms[1]),
            }
            validation = cfg.get("validation", {})
            every = int(validation.get("val_every_steps", 0))
            if every > 0 and step % every == 0:
                selected, used_ema = validation_model_from_ema(
                    model, ema, ema_settings, step=step
                )
                free = validation.get("free_running", {})
                run_free = (
                    bool(free.get("enabled", False))
                    and step % int(free.get("every_steps", every)) == 0
                )
                row["validation"] = evaluate(
                    selected, val_loader, cfg, device, free_running=run_free
                )
                row["validation_ema"] = used_ema
                if row["validation"]["loss"] < best_loss:
                    best_loss = row["validation"]["loss"]
                    save(output / "best.pt")
            save_every = int(validation.get("save_every_steps", 0))
            if save_every > 0 and step % save_every == 0:
                save(checkpoint_path)
            log.write(json.dumps(row) + "\n")
            log.flush()
            bar.update(1)
            bar.set_postfix(loss=f"{loss_sum:.4f}")
        save(checkpoint_path)
    bar.close()
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
