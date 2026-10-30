"""Select short/median/long real TRAIN pairs for a reproducible memorization check."""
import argparse
import json
from pathlib import Path

from a2s.data import read_manifest
from a2s.runtime import configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/saxophone/prepared.jsonl")
    parser.add_argument("--tokenizer", default="outputs/saxophone/vocab.json")
    parser.add_argument("--output", default="outputs/saxophone/smoke")
    args = parser.parse_args()
    rows = sorted((r for r in read_manifest(args.manifest) if r["split"] == "train"), key=lambda r: r["end"] - r["start"])
    if len(rows) < 3:
        raise ValueError("At least three training examples are required")
    selected = [rows[0], rows[len(rows) // 2], rows[-1]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    cfg = configuration("config/score_smoke.yaml", args.tokenizer)
    cfg["model"].update(gradient_checkpointing=False, attention_chunk_size=128)
    cfg["data"].update(variable_length=True, max_audio_seconds=120, stft_backend="fft")
    cfg["training"].update(steps=600, save_every=600)
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps({"output": str(output), "pairs": [(r["id"], r["end"] - r["start"]) for r in selected]}, indent=2))


if __name__ == "__main__":
    main()
