"""Quartets/Saxophone adapters for the repository's original Audio to MIDI model.

All corpus caches and predictions live in the run directory. Source data is read-only.
"""
import argparse
from bisect import bisect_right
from collections import Counter, OrderedDict
import hashlib
import json
import math
from pathlib import Path
import shutil


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_partitions(root, partitions):
    """Accept bare filenames (Quartets) and wav/midi relative paths (Saxophone)."""
    root = Path(root).resolve()
    rows, audio_seen, midi_seen = [], set(), set()
    for split in ("train", "val", "test"):
        path = root / partitions / f"{split}.txt"
        lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Empty partition: {path}")
        for fields in lines:
            if len(fields) != 2:
                raise ValueError(f"Expected WAV and MIDI columns in {path}: {fields}")
            resolved = []
            for name, directory in zip(fields, ("wav", "midi")):
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe corpus path: {name}")
                if relative.parts[0] != directory:
                    relative = Path(directory) / relative
                source = (root / relative).resolve()
                if not source.is_relative_to(root) or not source.is_file() or not source.stat().st_size:
                    raise ValueError(f"Missing or invalid corpus file: {source}")
                resolved.append(source)
            audio, midi = resolved
            if audio in audio_seen or midi in midi_seen:
                raise ValueError(f"Duplicate pair or split overlap: {audio}, {midi}")
            audio_seen.add(audio)
            midi_seen.add(midi)
            rows.append({"id": len(rows), "split": split, "audio": str(audio),
                         "midi": str(midi), "audio_relative": str(audio.relative_to(root)),
                         "midi_relative": str(midi.relative_to(root))})
    stems = [Path(row["audio_relative"]).stem for row in rows]
    if len(stems) != len(set(stems)):
        raise ValueError("Audio basenames must be unique to name predicted MIDI files")
    return rows


def prepare(args):
    import h5py
    import numpy as np
    import pandas as pd
    import partitura
    import soundfile as sf
    from scipy.signal import resample_poly
    from tqdm import tqdm

    rows = read_partitions(args.root, args.partitions)
    target = args.output / "dataset"
    target.mkdir()  # Never overwrite an earlier or partial preparation.
    (target / "cache").mkdir()
    for split in ("train", "val", "test"):
        shutil.copyfile(args.root / args.partitions / f"{split}.txt",
                        target / f"original_{split}.txt")
    subsets = {"train": [], "validation": [], "test": []}
    with h5py.File(target / "audio.h5", "w") as h5:
        for row in tqdm(rows, desc="Preparing WAV/MIDI pairs"):
            audio, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sr != 16000:
                divisor = math.gcd(sr, 16000)
                audio = resample_poly(audio, 16000 // divisor, sr // divisor)
            if not len(audio) or not np.isfinite(audio).all() or len(audio) > 16000 * 3000:
                raise ValueError(f"Invalid audio: {row['audio']}")
            # Match the original MelSpectrogram's required [-1, 1] input range.
            peak = float(np.abs(audio).max())
            if peak > 1:
                audio = audio / peak
            audio = audio.astype(np.float32)
            relative = row["audio_relative"]
            cache_stem = str(Path(relative).with_suffix("")).replace("/", ">")
            cache_midi = target / "cache" / f"{cache_stem}.mid"
            shutil.copyfile(row["midi"], cache_midi)
            notes = partitura.load_performance(row["midi"], merge_tracks=True,
                                               pedal_threshold=64).note_array()
            frame = pd.DataFrame({"type": "note", "type_id": 1,
                "onset_sec": notes["onset_sec"], "duration_sec": notes["duration_sec"],
                "offset_sec": notes["onset_sec"] + notes["duration_sec"],
                "pitch": notes["pitch"], "velocity": notes["velocity"]})
            if frame.empty or not np.isfinite(frame.select_dtypes("number")).all().all():
                raise ValueError(f"Empty or invalid MIDI: {row['midi']}")
            if (frame.duration_sec <= 0).any():
                raise ValueError(f"Non-positive MIDI duration: {row['midi']}")
            frame.sort_values(["onset_sec", "type_id", "pitch"]).to_csv(
                cache_midi.with_suffix(".midi-notes.tsv"), sep="\t", index=False)
            row.update({"samples": len(audio), "duration": len(audio) / 16000,
                        "notes": len(frame), "input_peak": peak})
            group = h5.create_group(str(row["id"]))
            group.create_dataset("audio", data=audio)
            group["path"] = row["audio"]
            group["sample_rate"] = 16000
            group["duration"] = row["duration"]
            split = "validation" if row["split"] == "val" else row["split"]
            subsets[split].append(f"{row['id']}\t{relative}\n")
    for split, lines in subsets.items():
        (target / f"subset_{split}.tsv").write_text("".join(lines))
    (target / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(target / "complete.json", {"task": "audio_to_midi",
        "splits": dict(Counter(row["split"] for row in rows)), "sample_rate": 16000})


def load_config(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(Path.cwd() / "config"), version_base=None):
        config = compose(config_name="main_config")
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(args.config))
    config.data.dataset_dirs = [str(args.output / "dataset")]
    config.model.task = "audio_to_midi"
    if config.model.checkpoint_path is not None or config.model.froze_encoder:
        raise ValueError("These jobs train from random initialization")
    return config


def make_dataset(config, rows, split):
    import numpy as np
    from torch.utils.data import Dataset
    from data.dataset_Audio2Midi import SingleWavDataset

    class PartitionDataset(Dataset):
        """Use original MIDI tokenization with a bounded per-worker HDF5 cache.

        The original corpus loader opens one HDF5 handle per piece at construction.
        Quartets has 38k pieces; lazy loading avoids exhausting file descriptors.
        """
        def __init__(self):
            self.rows = [row for row in rows if row["split"] == split]
            self.ends = []
            self.cache = OrderedDict()
            count = 0
            for row in self.rows:
                count += math.ceil(row["samples"] / (config.data.hop_length * config.data.n_frames))
                self.ends.append(count)

        def __len__(self):
            return self.ends[-1]

        def __getitem__(self, index):
            piece = bisect_right(self.ends, index)
            row = self.rows[piece]
            if piece not in self.cache:
                root = Path(config.data.dataset_dirs[0])
                stem = str(Path(row["audio_relative"]).with_suffix("")).replace("/", ">")
                self.cache[piece] = SingleWavDataset(config, str(root), 0, row["id"],
                    str(root / "cache" / f"{stem}.mid"), str(root / "audio.h5"), random_clip=False)
                if len(self.cache) > 32:
                    _, old = self.cache.popitem(last=False)
                    old.audio_h5.close()
            source = self.cache.pop(piece)
            self.cache[piece] = source
            local = index - (self.ends[piece - 1] if piece else 0)
            begin = local * config.data.n_frames
            if split == "train":
                # The legacy randint fails on clips shorter than n_frames / 4.
                begin = int(np.random.randint(max(1, source.total_frames - config.data.n_frames // 4)))
            return source.__getitem__(local, begin=begin, end=begin + config.data.n_frames)

    return PartitionDataset()


def read_manifest(output):
    return [json.loads(line) for line in (output / "dataset/manifest.jsonl").read_text().splitlines()]


def make_frontend(config):
    from data.mel import MelSpectrogram
    return MelSpectrogram(config.data.num_mel_bins, config.data.sample_rate,
        config.data.fft_size, config.data.hop_length, mel_fmin=config.data.mel_fmin,
        mel_fmax=config.data.mel_fmax, stft_backend="fft")


def train(args):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    from model.T5 import Transformer
    from loss.CrossEntropyLoss import CrossEntropyLoss

    config = load_config(args)
    rows = read_manifest(args.output)
    pl.seed_everything(config.training.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    class MidiModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = Transformer(config.model)
            self.frontend = make_frontend(config)
            self.criterion = CrossEntropyLoss(config)

        def loss(self, batch):
            with torch.no_grad():
                features = self.frontend(batch["inputs"][:, :-1]).transpose(-1, -2)
            result = self.model(encoder_input_tokens=features,
                decoder_target_tokens=batch["decoder_targets"], decode=False,
                decoder_targets_frame_index=batch["decoder_targets_frame_index"],
                encoder_decoder_mask=batch["encoder_decoder_mask"])
            loss = self.criterion(result["decoder_outputs"], batch["decoder_targets"],
                                  batch["decoder_targets_mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Audio to MIDI loss")
            return loss

        def training_step(self, batch, batch_idx):
            loss = self.loss(batch)
            self.log("train_loss", loss, batch_size=len(batch["inputs"]))
            return loss

        def validation_step(self, batch, batch_idx):
            self.log("val_loss", self.loss(batch), on_epoch=True,
                     batch_size=len(batch["inputs"]))

        def configure_optimizers(self):
            return torch.optim.AdamW(self.model.parameters(), lr=config.training.learning_rate)

    run = args.output / "run"
    run.mkdir()
    OmegaConf.save(config, run / "config.yaml", resolve=True)
    loaders = []
    for split in ("train", "val"):
        dataset = make_dataset(config, rows, split)
        loaders.append(DataLoader(dataset, batch_size=config.training.batch,
            shuffle=split == "train", num_workers=config.training.num_workers,
            pin_memory=args.device == "cuda"))
    checkpoints = ModelCheckpoint(dirpath=run / "checkpoints", filename="best",
        monitor="val_loss", mode="min", save_top_k=1, save_last=True)
    stop = EarlyStopping(monitor="val_loss", patience=config.training.patience, mode="min")
    trainer = pl.Trainer(accelerator="gpu" if args.device == "cuda" else "cpu", devices=1,
        max_steps=config.training.training_steps, max_epochs=-1,
        max_time=config.training.max_time, precision="32-true",
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        val_check_interval=config.training.validate_every, check_val_every_n_epoch=None,
        gradient_clip_val=0.5, gradient_clip_algorithm="value",
        callbacks=[checkpoints, stop], num_sanity_val_steps=0,
        logger=[CSVLogger(str(run), name="csv"), TensorBoardLogger(str(run), name="tensorboard")],
        log_every_n_steps=10, enable_progress_bar=False)
    module = MidiModule()
    trainer.fit(module, train_dataloaders=loaders[0], val_dataloaders=loaders[1])
    # A timer/short test may stop before the first scheduled validation.
    best_path = checkpoints.best_model_path
    best_score = checkpoints.best_model_score
    if not best_path:
        metrics = trainer.validate(module, dataloaders=loaders[1])
        best_score = metrics[0]["val_loss"]
        if not math.isfinite(best_score):
            raise RuntimeError("No finite validation loss to select a checkpoint")
        best_path = str(run / "checkpoints/best.ckpt")
        trainer.save_checkpoint(best_path)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    state = {name.removeprefix("model."): tensor for name, tensor in best["state_dict"].items()
             if name.startswith("model.")}
    torch.save(state, run / "best.pt")
    torch.save({name: tensor.detach().cpu() for name, tensor in module.model.state_dict().items()},
               run / "latest.pt")
    write_json(args.output / "training_complete.json", {"task": "audio_to_midi",
        "initialization": "random", "step": trainer.global_step,
        "best_validation_loss": float(best_score),
        "best_checkpoint": "run/best.pt", "early_stopped": bool(stop.stopped_epoch)})


def predict_audio(model, frontend, waveform, config, device):
    """Generate from audio alone; merge events before reconstructing note offsets."""
    import torch
    import torch.nn.functional as F
    from data.constants import sm_tokenizer

    clip_samples = config.data.n_frames * config.data.hop_length
    batch_size = config.training.batch_inference
    events, truncated = [], 0
    for start in range(0, len(waveform), clip_samples * batch_size):
        waves = []
        for offset in range(start, min(len(waveform), start + clip_samples * batch_size), clip_samples):
            wave = torch.as_tensor(waveform[offset:offset + clip_samples], device=device)
            waves.append(F.pad(wave, (0, clip_samples - len(wave))))
        with torch.inference_mode():
            features = frontend(torch.stack(waves)[:, :-1]).transpose(-1, -2)
            tokens = model.generate(features, target_seq_length=config.data.max_token_length,
                                    berak_on_eos=True)
        for index, sequence in enumerate(tokens.cpu().tolist()):
            if sm_tokenizer.EOS in sequence:
                sequence = sequence[:sequence.index(sm_tokenizer.EOS)]
            else:
                truncated += 1
            # Keep predictions even when generation reaches its token budget.
            events.extend(sm_tokenizer.detokenize(sequence,
                offsets_sec=(start + index * clip_samples) / config.data.sample_rate))
    notes = sm_tokenizer.midi_events_to_notes(events) if events else []
    return notes, truncated


def predict(args):
    import h5py
    import torch
    from data.constants import sm_tokenizer
    from model.T5 import Transformer
    from omegaconf import OmegaConf
    from tqdm import tqdm

    config = OmegaConf.load(args.output / "run/config.yaml")
    checkpoint = args.output / "run/best.pt"
    model = Transformer(config.model).to(args.device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True), strict=True)
    frontend = make_frontend(config).to(args.device).eval()
    rows = [row for row in read_manifest(args.output) if row["split"] == "test"]
    # Recheck against the original partition, including every test pair.
    original = read_partitions(args.root, args.partitions)
    expected = [(r["audio_relative"], r["midi_relative"]) for r in original if r["split"] == "test"]
    actual = [(r["audio_relative"], r["midi_relative"]) for r in rows]
    if not rows or actual != expected:
        raise ValueError("Prepared test pairs do not match the original test partition")
    destination = args.prediction_output or args.output / "predictions/test"
    destination.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(destination / "prediction_config.json", {"task": "audio_to_midi",
        "checkpoint": "run/best.pt", "checkpoint_sha256": digest,
        "samples": len(rows), "decoding": "greedy", "reference_conditioning": False})
    predicted, total_truncated = 0, 0
    with h5py.File(args.output / "dataset/audio.h5", "r") as h5, \
            (destination / "examples.jsonl").open("w") as reports:
        for row in tqdm(rows, desc="Audio to MIDI: test"):
            output = destination / (Path(row["audio_relative"]).stem + ".mid")
            notes, truncated = predict_audio(model, frontend, h5[str(row["id"])]["audio"][:],
                                              config, args.device)
            sm_tokenizer.save_midi(notes, str(output))
            if not output.is_file() or not output.stat().st_size:
                raise RuntimeError(f"Missing predicted MIDI: {output}")
            predicted += 1
            total_truncated += truncated
            reports.write(json.dumps({"id": row["id"], "audio": row["audio_relative"],
                "reference": row["midi_relative"], "prediction": output.name,
                "notes": len(notes), "truncated_segments": truncated}) + "\n")
            reports.flush()
            write_json(destination / "progress.json", {"completed": predicted, "total": len(rows)})
    if len(list(destination.glob("*.mid"))) != len(rows):
        raise RuntimeError("The predicted MIDI count does not match test")
    write_json(destination / "complete.json", {"samples": predicted, "truncated_segments": total_truncated})
    if args.prediction_output is None:
        write_json(args.output / "complete.json", {"task": "audio_to_midi",
            "training": "training_complete.json", "predicted_test_midis": predicted,
            "predictions": "predictions/test", "truncated_segments": total_truncated})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "prepare", "train", "predict"])
    parser.add_argument("--task", choices=["audio_to_midi"], default="audio_to_midi")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--partitions", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage == "check":
        rows = read_partitions(args.root, args.partitions)
        print(json.dumps(dict(Counter(row["split"] for row in rows))))
        return
    if args.stage != "prepare":
        import torch
        torch.set_num_threads(args.threads)
    {"prepare": prepare, "train": train, "predict": predict}[args.stage](args)


if __name__ == "__main__":
    main()
