"""Length-aware, token-normalized training of complete audio-score pairs."""
import contextlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ScoreDataset, FeatureDataset, LengthBatchSampler, collate_features, read_manifest
from .runtime import configuration, load_checkpoint, save_checkpoint
from .tokenizer import tokenizer_from_metadata
from model.T5 import Transformer


def precision_mode(settings, device):
    mode = settings.get("precision", "fp32")
    if mode == "auto":
        mode = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if str(device).startswith("cuda") else "fp32"
    if mode not in {"fp32", "fp16", "bf16"} or (mode == "fp16" and not str(device).startswith("cuda")):
        raise ValueError("Use fp32/bf16, or fp16 on CUDA")
    return mode


def autocast(mode, device):
    if mode == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16 if mode == "bf16" else torch.float16)


def make_loader(features, sampler, settings, device, epoch):
    workers = settings.get("num_workers", 0)
    return DataLoader(features, batch_sampler=sampler, collate_fn=collate_features,
                      num_workers=workers, pin_memory=str(device).startswith("cuda"),
                      persistent_workers=False,
                      generator=torch.Generator().manual_seed(settings["seed"] + epoch),
                      **({"prefetch_factor": 2} if workers else {}))


@torch.no_grad()
def validate_loss(model, dataset, feature_dataset, cfg, device, mode):
    model.eval()
    loss_sum, count = 0.0, 0
    sampler = LengthBatchSampler(dataset, cfg["training"], shuffle=False)
    for x, y, _, lengths in make_loader(feature_dataset, sampler, cfg["training"], device, 0):
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        with autocast(mode, device):
            logits = model(x, y, encoder_lengths=lengths, enable_dropout=False)["decoder_outputs"]
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=0, reduction="sum")
        loss_sum += loss.item()
        count += (y != 0).sum().item()
    model.train()
    return loss_sum / count


def train(args):
    checkpoint = None
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, args.device)
        cfg = checkpoint["config"]
        if (args.config or getattr(args, "tokenizer", None) or getattr(args, "set", None)) and configuration(args.config, getattr(args, "tokenizer", None), getattr(args, "set", None)) != cfg:
            raise ValueError("Resume configuration differs; use checkpoint config and --steps")
    else:
        cfg = configuration(args.config, getattr(args, "tokenizer", None), getattr(args, "set", None))
        torch.manual_seed(cfg["training"]["seed"])
        random.seed(cfg["training"]["seed"])
        model = Transformer(cfg["model"]).to(args.device)
    settings = cfg["training"]
    tokenizer = tokenizer_from_metadata(cfg.get("tokenizer"))
    dataset = ScoreDataset(args.manifest, cfg["data"], "train", tokenizer=tokenizer)
    cache_dir = getattr(args, "cache_dir", None)
    feature_dataset = FeatureDataset(dataset, cache_dir, settings.get("cache_features", False))
    validation = validation_features = None
    if any(r["split"] == "validation" for r in read_manifest(args.manifest)):
        validation = ScoreDataset(args.manifest, cfg["data"], "validation", tokenizer=tokenizer)
        validation_features = FeatureDataset(validation, cache_dir)
    fingerprint = dataset.fingerprint()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    mode = precision_mode(settings, args.device)
    scaler = torch.amp.GradScaler("cuda", enabled=mode == "fp16")
    warmup, schedule_steps = settings.get("warmup_steps", 0), settings["steps"]
    def lr_multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        if not settings.get("cosine_decay", False):
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / max(1, schedule_steps - warmup)))
        floor = settings.get("min_lr_ratio", 0.1)
        return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    state = {"epoch": 0, "batch_cursor": 0, "best_validation_loss": float("inf"), "bad_validation_checks": 0}
    step = 0
    if checkpoint:
        if fingerprint != checkpoint["manifest_fingerprint"]:
            raise ValueError("Training manifest/targets/audio metadata changed since checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        state.update(checkpoint.get("training_state", {}))
        if not checkpoint.get("training_state"):
            state["epoch"], state["batch_cursor"] = divmod(step, len(LengthBatchSampler(dataset, settings)))
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if checkpoint["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    total = args.steps if args.steps is not None else settings["steps"]
    if total <= step:
        raise ValueError("Total steps must exceed checkpoint step")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    started = time.monotonic()
    model.train()
    accumulation = settings.get("gradient_accumulation", 1)
    while step < total:
        sampler = LengthBatchSampler(dataset, settings, state["epoch"], state["batch_cursor"])
        loader = iter(make_loader(feature_dataset, sampler, settings, args.device, state["epoch"]))
        cursor = state["batch_cursor"]
        while cursor < len(sampler.batches) and step < total:
            microbatches = min(accumulation, len(sampler.batches) - cursor)
            # Normalize by total non-PAD tokens, including unequal final batches.
            tokens = sum(dataset.token_lengths[i] for b in sampler.batches[cursor:cursor+microbatches] for i in b)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(microbatches):
                x, y, _, lengths = next(loader)
                x, y, lengths = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True), lengths.to(args.device)
                with autocast(mode, args.device):
                    logits = model(x, y, encoder_lengths=lengths)["decoder_outputs"]
                    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=0, reduction="sum",
                                           label_smoothing=settings.get("label_smoothing", 0.0))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss_sum += loss.item()
                scaler.scale(loss / tokens).backward()
                del logits, loss, x, y
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("gradient_clip", 1.0))
            before_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            cursor += microbatches
            state["batch_cursor"] = cursor
            if cursor == len(sampler.batches):
                state["epoch"] += 1
                state["batch_cursor"] = 0
            if scaler.get_scale() < before_scale:
                # Overflow skipped the update; consume data but don't claim a step.
                continue
            scheduler.step()
            step += 1
            row = {"step": step, "loss": loss_sum / tokens, "tokens": tokens,
                   "microbatches": microbatches, "lr": optimizer.param_groups[0]["lr"],
                   "seconds": round(time.monotonic() - started, 2), "precision": mode}
            if step == 1 or step % 25 == 0 or step == total:
                print(json.dumps(row), flush=True)
            if step % settings["save_every"] == 0 or step == total:
                if validation is not None:
                    row["validation_loss"] = validate_loss(model, validation, validation_features, cfg, args.device, mode)
                    if row["validation_loss"] < state["best_validation_loss"] - settings.get("early_stopping_min_delta", 0.0):
                        state["best_validation_loss"] = row["validation_loss"]
                        state["bad_validation_checks"] = 0
                        save_checkpoint(output / "best.pt", model, optimizer, cfg, step, fingerprint, state, scheduler, scaler)
                    else:
                        state["bad_validation_checks"] += 1
                    patience = settings.get("early_stopping_patience", 0)
                    row["early_stopped"] = bool(patience and state["bad_validation_checks"] >= patience)
                save_checkpoint(output / "latest.pt", model, optimizer, cfg, step, fingerprint, state, scheduler, scaler)
                print(json.dumps({"checkpoint": str(output / "latest.pt"), **row}), flush=True)
            with (output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if row.get("early_stopped"):
                return
