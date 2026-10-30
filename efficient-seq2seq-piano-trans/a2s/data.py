"""Complete recordings or explicitly aligned intervals paired with **kern scores."""
from pathlib import Path
import os
import hashlib
import json
import math
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .kern import validate
from .tokenizer import KernTokenizer


def read_manifest(path):
    path = Path(path).resolve()
    entries, ids, pieces, audio_splits = [], set(), {}, {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = {"id", "piece_id", "split", "audio", "score", "start", "end"}
        if required - item.keys():
            raise ValueError(f"Manifest line {number} missing {required - item.keys()}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]) or item["id"] in ids:
            raise ValueError("IDs must be unique safe filenames")
        ids.add(item["id"])
        if item["split"] not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        for field in ("audio", "score"):
            item[field] = str((path.parent / item[field]).resolve())
            if not Path(item[field]).is_file():
                raise FileNotFoundError(item[field])
        for key, mapping in ((item["piece_id"], pieces), (item["audio"], audio_splits)):
            if key in mapping and mapping[key] != item["split"]:
                raise ValueError("Data leakage: a piece or audio file occurs in different splits")
            mapping[key] = item["split"]
        start, end = float(item["start"]), float(item["end"])
        if not math.isfinite(start + end) or not 0 <= start < end:
            raise ValueError("Require finite 0 <= start < end in seconds")
        info = sf.info(item["audio"])
        if end > info.duration + 1 / info.samplerate:
            raise ValueError(f"{item['id']}: interval extends past audio duration")
        item.update(start=start, end=end)
        entries.append(item)
    if not entries:
        raise ValueError("Empty manifest")
    return entries


def load_audio(path, start, end, config):
    info = sf.info(path)
    if not 0 <= start < end <= info.duration + 1 / info.samplerate:
        raise ValueError("Audio interval out of bounds")
    sr, hop, frames = config["sample_rate"], config["hop_length"], config["n_frames"]
    limit = config.get("max_audio_seconds", 120) if config.get("variable_length", False) else frames * hop / sr
    if end - start > limit + 1 / sr:
        raise ValueError(f"Audio exceeds configured limit ({limit}s); increase max_audio_seconds, never truncate labels")
    audio, original_sr = sf.read(path, start=round(start * info.samplerate),
                                 stop=round(end * info.samplerate), always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if original_sr != sr:
        # Same resampling defaults as the original inference.py.
        audio = torchaudio.transforms.Resample(original_sr, sr)(torch.from_numpy(audio)).numpy()
    if not np.isfinite(audio).all() or np.max(np.abs(audio), initial=0) > 1.0001:
        raise ValueError("Audio must be finite and normalized to [-1, 1]")
    size = frames * hop
    if config.get("variable_length", False):
        size = max(math.ceil(len(audio) / hop), math.ceil((config["fft_size"] / 2 + 2) / hop)) * hop
    if len(audio) > size + 1:
        raise ValueError("Resampled interval exceeds configured size")
    audio = np.pad(audio[:size], (0, max(0, size - len(audio))))
    return torch.from_numpy(audio.astype(np.float32))


class ScoreDataset(Dataset):
    def __init__(self, manifest, config, split="train", external=False, tokenizer=None):
        self.entries = [r for r in read_manifest(manifest) if r["split"] == split]
        if not self.entries:
            raise ValueError(f"No samples in split {split}")
        self.config = config
        self.tokenizer = tokenizer or KernTokenizer()
        self.reports, self.token_lengths, self.frame_lengths, self.score_hashes = [], [], [], []
        for item in self.entries:
            text = Path(item["score"]).read_text(encoding="utf-8")
            report = validate(text, external=external)
            tokens = self.tokenizer.encode(text)
            if len(tokens) > config["max_token_length"]:
                raise ValueError(f"{item['id']}: {len(tokens)} tokens exceed max_token_length; do not truncate scores")
            limit = config.get("max_audio_seconds", 120) if config.get("variable_length", False) else config["n_frames"] * config["hop_length"] / config["sample_rate"]
            if item["end"] - item["start"] > limit:
                raise ValueError(f"{item['id']}: audio interval too long")
            self.token_lengths.append(len(tokens))
            self.reports.append(report)
            self.score_hashes.append(hashlib.sha256(text.encode()).hexdigest())
            frames = math.ceil((item["end"] - item["start"]) * config["sample_rate"] / config["hop_length"])
            minimum = math.ceil((config["fft_size"] / 2 + 2) / config["hop_length"])
            self.frame_lengths.append(max(minimum, frames) if config.get("variable_length", False) else config["n_frames"])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        item = self.entries[index]
        audio = load_audio(item["audio"], item["start"], item["end"], self.config)
        return audio, self.target(index), item["id"]

    def target(self, index):
        text = Path(self.entries[index]["score"]).read_text(encoding="utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != self.score_hashes[index]:
            raise ValueError("Score changed after dataset indexing")
        return torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

    @property
    def texts(self):
        return [Path(item["score"]).read_text(encoding="utf-8") for item in self.entries]

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(json.dumps(self.tokenizer.metadata(), sort_keys=True).encode())
        for item, score_hash in zip(self.entries, self.score_hashes):
            digest.update(json.dumps(item, sort_keys=True).encode())
            digest.update(score_hash.encode())
            stat = Path(item["audio"]).stat()
            digest.update(str((stat.st_size, stat.st_mtime_ns)).encode())
        return digest.hexdigest()


def collate(items):
    audio, targets, names = zip(*items)
    return torch.stack(audio), pad_sequence(targets, batch_first=True, padding_value=KernTokenizer.PAD), names


def collate_features(items):
    acoustic, targets, names = zip(*items)
    lengths = torch.tensor([x.size(0) for x in acoustic], dtype=torch.long)
    return (pad_sequence(acoustic, batch_first=True),
            pad_sequence(targets, batch_first=True, padding_value=0), names, lengths)


class FeatureDataset(Dataset):
    """Lazy disk cache keyed by file metadata and frontend settings."""
    def __init__(self, dataset, cache_dir=None, memory_cache=False):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory = {} if memory_cache else None
        self.extractor = None

    def __len__(self):
        return len(self.dataset)

    def cache_path(self, index):
        if self.cache_dir is None:
            return None
        item = self.dataset.entries[index]
        stat = Path(item["audio"]).stat()
        identity = {"version": "original-logmel-v1", "audio": item["audio"],
                    "stat": (stat.st_size, stat.st_mtime_ns), "start": item["start"], "end": item["end"],
                    "config": self.dataset.config}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / (key + ".npy")

    def get_features(self, index, extractor=None, device="cpu"):
        from .runtime import features, frontend
        if self.memory is not None and index in self.memory:
            return self.memory[index]
        path = self.cache_path(index)
        if path is not None and path.exists():
            result = torch.from_numpy(np.load(path, allow_pickle=False))
        else:
            if extractor is None:
                if self.extractor is None:
                    self.extractor = frontend({"data": self.dataset.config}, "cpu")
                extractor = self.extractor
            audio, _, _ = self.dataset[index]
            result = features(extractor, audio[None].to(device))[0].cpu()
            if path is not None:
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, result.numpy(), allow_pickle=False)
                temporary.replace(path)
        if (result.ndim != 2 or result.shape[1] != self.dataset.config["num_mel_bins"] or
                result.dtype != torch.float32 or result.shape[0] < 1 or not torch.isfinite(result).all()):
            raise ValueError(f"Malformed feature cache: {path}")
        if self.memory is not None:
            self.memory[index] = result
        return result

    def __getitem__(self, index):
        return self.get_features(index), self.dataset.target(index), self.dataset.entries[index]["id"]


class LengthBatchSampler:
    """Deterministic length buckets with padded-frame/token/cross-attention budgets."""
    def __init__(self, dataset, settings, epoch=0, start_batch=0, shuffle=True):
        generator = torch.Generator().manual_seed(settings["seed"] + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist() if shuffle else list(range(len(dataset)))
        bucket = settings.get("bucket_size", 1)
        if bucket > 1:
            order = [index for start in range(0, len(order), bucket)
                     for index in sorted(order[start:start+bucket], key=lambda i: (dataset.frame_lengths[i], dataset.token_lengths[i]))]
        batches, current = [], []
        def fits(indices):
            frames = max(dataset.frame_lengths[i] for i in indices)
            tokens = max(dataset.token_lengths[i] for i in indices)
            return (len(indices) <= settings["batch"] and
                    len(indices) * frames <= settings.get("max_batch_frames", float("inf")) and
                    len(indices) * tokens <= settings.get("max_batch_tokens", float("inf")) and
                    len(indices) * frames * tokens <= settings.get("max_cross_product", float("inf")))
        for index in order:
            if not fits([index]):
                raise ValueError(f"{dataset.entries[index]['id']}: exceeds single-example batch budget; increase budgets explicitly")
            if current and not fits(current + [index]):
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)
        self.batches = batches
        self.start_batch = start_batch

    def __iter__(self):
        return iter(self.batches[self.start_batch:])

    def __len__(self):
        return len(self.batches) - self.start_batch
