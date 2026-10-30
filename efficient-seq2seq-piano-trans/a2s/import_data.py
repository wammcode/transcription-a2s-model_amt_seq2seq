"""Import complete paired local Saxophone folders.

    python -m a2s.import_data saxophone --root ... --output outputs/saxophone

Sources are never modified; partitions are retained. No MIDI is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import soundfile as sf


def write_manifest(entries, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in entries))
    print(json.dumps({"manifest": str(path), "samples": len(entries),
                      "max_seconds": max(r["end"] for r in entries),
                      "splits": {s: sum(r["split"] == s for r in entries) for s in sorted({r["split"] for r in entries})}}, indent=2))
    return path


def saxophone(root, output, score_dir="krn_clean", audio_dir="wav", partitions_dir="a2s_partitions"):
    root = Path(root).resolve()
    entries = []
    for source_split, split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        partition = root / partitions_dir / (source_split + ".txt")
        for line in partition.read_text().splitlines():
            if not line.strip():
                continue
            audio_name, score_name = line.split()
            audio, score = root / audio_dir / audio_name, root / score_dir / score_name
            if not score.is_file():
                raise FileNotFoundError(score)
            info = sf.info(audio)
            ident = re.sub(r"[^A-Za-z0-9_-]", "_", Path(audio_name).stem) + "-" + hashlib.sha256(audio_name.encode()).hexdigest()[:8]
            entries.append({"id": ident, "piece_id": Path(score_name).stem, "split": split,
                            "category": "mono", "audio": str(audio), "score": str(score),
                            "start": 0, "end": info.duration})
    return write_manifest(entries, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("saxophone")
    local.add_argument("--root", required=True)
    local.add_argument("--output", required=True)
    local.add_argument("--score-dir", default="krn_clean")
    local.add_argument("--audio-dir", default="wav")
    local.add_argument("--partitions-dir", default="a2s_partitions")
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"saxophone": saxophone}[command](**args)


if __name__ == "__main__":
    main()
