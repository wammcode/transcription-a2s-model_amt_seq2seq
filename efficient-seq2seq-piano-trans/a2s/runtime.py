"""Configuration, unchanged acoustic frontend, strict task-specific checkpoints."""
from pathlib import Path
import random
import json
import torch
from omegaconf import OmegaConf

from data.mel import MelSpectrogram
from model.T5 import Transformer
from .tokenizer import KernTokenizer, tokenizer_from_metadata


def configuration(path=None, tokenizer_path=None, overrides=None):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "config/score.yaml")
    if path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if tokenizer_path:
        cfg["tokenizer"] = tokenizer_from_metadata(json.loads(Path(tokenizer_path).read_text())).metadata()
        cfg["model"]["vocab_size"] = cfg["tokenizer"]["vocab_size"]
    check_config(cfg)
    return cfg


def check_config(cfg):
    model, data = cfg["model"], cfg["data"]
    if cfg["task"] != "audio_to_score" or model["task"] != "audio_to_score":
        raise ValueError("Expected audio_to_score configuration")
    tok = tokenizer_from_metadata(cfg.get("tokenizer"))
    if (model["vocab_size"], model["pad_token"], model["bos_token"], model["eos_token"]) != (tok.vocab_size, tok.PAD, tok.BOS, tok.EOS):
        raise ValueError("Model vocabulary does not match tokenizer")
    if model["encoder_name"] != "TransformerEncoder" or model["decoder_name"] != "TransformerDecoder":
        raise ValueError("A2S retains the repository's Transformer encoder/decoder")
    if model["froze_encoder"] or model.get("checkpoint_path"):
        raise ValueError("A2S trains from scratch; only --resume accepts an A2S checkpoint")
    if model.get("hybrid_global_local_cross_attn") or model.get("encoder_decoder_slide_window_size", 0):
        raise ValueError("MIDI triplet scheduling is incompatible with **kern tokens")
    if data["features"] != "mel" or data["amplitude_to_db"]:
        raise ValueError("Use the original experiment log-mel frontend")
    if model["encoder_input_dim"] != data["num_mel_bins"]:
        raise ValueError("Encoder input dimension differs from mel bins")
    if model["emb_dim"] != model["num_heads"] * model["head_dim"]:
        raise ValueError("emb_dim must equal num_heads * head_dim")
    if not 0 < data["max_token_length"] <= model["max_decode_length"] or not 0 < data["n_frames"]:
        raise ValueError("Sequence length exceeds positional capacity")
    if data.get("variable_length") and data.get("max_audio_seconds", 120) <= 0:
        raise ValueError("max_audio_seconds must be positive")
    if model.get("cross_attention_hierarchy_pooling"):
        sizes = model["pooling_sizes"]
        if len(sizes) != model["num_decoder_layers"] or any(p < 1 for p in sizes):
            raise ValueError("Pooling requires one positive group size per decoder layer")
    if model.get("attention_chunk_size", 256) < 1:
        raise ValueError("attention_chunk_size must be positive")
    settings = cfg["training"]
    if any(settings.get(k, 1) < 1 for k in ["batch", "gradient_accumulation", "save_every"]):
        raise ValueError("Batch size, gradient accumulation and save_every must be positive")


def frontend(config, device):
    d = config["data"]
    return MelSpectrogram(d["num_mel_bins"], d["sample_rate"], d["fft_size"],
                          d["hop_length"], mel_fmin=d["mel_fmin"], mel_fmax=d["mel_fmax"],
                          stft_backend=d.get("stft_backend", "conv")).to(device)


@torch.no_grad()
def features(extractor, audio):
    # Identical to MT3Trainer.forward_step: drop final sample, then transpose.
    return extractor(audio[:, :-1]).transpose(-1, -2).detach()


def save_checkpoint(path, model, optimizer, cfg, step, fingerprint, training_state=None, scheduler=None, scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 2, "task": "audio_to_score", "tokenizer": tokenizer_from_metadata(cfg.get("tokenizer")).metadata(),
               "config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "step": step, "manifest_fingerprint": fingerprint,
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    payload["training_state"] = training_state or {}
    payload["scheduler"] = scheduler.state_dict() if scheduler else None
    payload["scaler"] = scaler.state_dict() if scaler else None
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != "audio_to_score" or checkpoint.get("format_version") not in (1, 2):
        raise ValueError("Not an A2S checkpoint; MIDI weights cannot initialize this task")
    if checkpoint.get("tokenizer") != tokenizer_from_metadata(checkpoint.get("config", {}).get("tokenizer")).metadata():
        raise ValueError("Tokenizer version/vocabulary mismatch")
    check_config(checkpoint["config"])
    model = Transformer(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint
