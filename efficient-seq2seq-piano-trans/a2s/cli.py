"""A2S preparation, teacher-forced training, free-running inference/evaluation."""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from .data import ScoreDataset, FeatureDataset, load_audio, read_manifest
from .kern import validate
from .runtime import configuration, features, frontend, load_checkpoint
from .tokenizer import KernTokenizer, KernFieldTokenizer, tokenizer_from_metadata


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parser(command):
    p = argparse.ArgumentParser(description=f"Direct audio_to_score: {command}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    if command in {"prepare", "train"}:
        p.add_argument("--config", help="YAML overrides of config/score.yaml")
        p.add_argument("--tokenizer", help="Vocabulary JSON from prepare --fit-tokenizer")
        p.add_argument("--set", action="append", default=[], help="YAML override key=value (repeatable)")
        p.add_argument("--cache-dir", help="Persistent log-mel cache directory")
    if command in {"prepare", "train", "evaluate"}:
        p.add_argument("--manifest", required=True, help="Aligned audio-score JSONL")
    if command == "prepare":
        p.add_argument("--fit-tokenizer", help="Fit compact vocabulary on train only and save JSON")
        p.add_argument("--vocab-size", type=int, default=2048)
        p.add_argument("--output", required=True, help="Resolved, validated manifest JSONL")
    elif command == "train":
        p.add_argument("--output", required=True, help="Run directory")
        p.add_argument("--resume", help="Resume an A2S checkpoint, including optimizer and RNG")
        p.add_argument("--steps", type=int, help="Total optimizer steps (including resumed steps)")
    elif command == "inference":
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--audio", required=True)
        p.add_argument("--output", required=True, help="Output .krn (only written if valid)")
        p.add_argument("--start", type=float, default=0.0)
        p.add_argument("--end", type=float)
        p.add_argument("--max-tokens", type=int)
    elif command == "evaluate":
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--split", default="test", choices=["train", "validation", "test"])
        p.add_argument("--output", required=True, help="Evaluation directory")
    return p


def prepare(args):
    entries = read_manifest(args.manifest)
    if args.fit_tokenizer:
        if args.tokenizer:
            raise ValueError("Use --fit-tokenizer or --tokenizer, not both")
        tok = KernFieldTokenizer.fit((Path(e["score"]).read_text() for e in entries if e["split"] == "train"), args.vocab_size)
        write_json(args.fit_tokenizer, tok.metadata())
        args.tokenizer = args.fit_tokenizer
    cfg = configuration(args.config, args.tokenizer, args.set)
    tokenizer = tokenizer_from_metadata(cfg.get("tokenizer"))
    report = {}
    extractor = frontend(cfg, args.device) if args.cache_dir else None
    for split in sorted({r["split"] for r in entries}):
        dataset = ScoreDataset(args.manifest, cfg["data"], split, external=True, tokenizer=tokenizer)
        cache = FeatureDataset(dataset, args.cache_dir)
        for i in range(len(dataset)):
            if args.cache_dir:
                cache.get_features(i, extractor, args.device)
            else:
                dataset[i]
            if (i + 1) % 100 == 0:
                print(json.dumps({"split": split, "prepared": i + 1, "total": len(dataset)}), flush=True)
        report[split] = [{"id": r["id"], "tokens": t, "frames": f, **v}
                         for r, t, f, v in zip(dataset.entries, dataset.token_lengths, dataset.frame_lengths, dataset.reports)]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    write_json(str(path) + ".validation.json", report)
    write_json(str(path) + ".tokenizer.json", tokenizer.metadata())
    print(json.dumps({split: {"samples": len(rows), "max_frames": max(r["frames"] for r in rows),
                             "max_tokens": max(r["tokens"] for r in rows)} for split, rows in report.items()}, indent=2))


def train(args):
    from .training import train as train_complete_pairs
    train_complete_pairs(args)


def serialize_prediction(ids, output, external=True, tokenizer=None):
    """Never disguise invalid/truncated generation as a valid score or repair music."""
    output = Path(output)
    if output.suffix != ".krn":
        raise ValueError("Score output must end in .krn")
    tokens = [int(t) for t in ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = (tokenizer or KernTokenizer()).decode(tokens)
        report = validate(text, external=external)
    except (ValueError, UnicodeDecodeError) as exc:
        diagnostic = {"valid": False, "error": str(exc), "tokens": tokens}
        write_json(str(output) + ".invalid.json", diagnostic)
        return diagnostic
    temporary = output.with_suffix(".krn.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    Path(str(output) + ".invalid.json").unlink(missing_ok=True)
    return {"valid": True, **report}


def inference(args):
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    cfg = checkpoint["config"]
    end = args.end if args.end is not None else sf.info(args.audio).duration
    audio = load_audio(args.audio, args.start, end, cfg["data"])[None].to(args.device)
    model.eval()
    predicted = model.generate(features(frontend(cfg, args.device), audio),
                               target_seq_length=args.max_tokens or cfg["data"]["max_token_length"])[0].tolist()
    report = serialize_prediction(predicted, args.output, tokenizer=tokenizer_from_metadata(checkpoint["tokenizer"]))
    report.update(audio=str(Path(args.audio).resolve()), start=args.start, end=end, checkpoint=args.checkpoint)
    write_json(str(args.output) + ".report.json", report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit("Invalid prediction: see .invalid.json; no new .krn was written")


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def symbols(text):
    # Distinguish chord members, column boundaries and row boundaries in SER.
    return text.replace("\t", " <tab> ").replace("\n", " <nl> ").split()


def evaluate(args):
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    cfg = checkpoint["config"]
    tokenizer = tokenizer_from_metadata(checkpoint["tokenizer"])
    dataset = ScoreDataset(args.manifest, cfg["data"], args.split, external=True, tokenizer=tokenizer)
    extractor = frontend(cfg, args.device)
    model.eval()
    results = []
    for index in range(len(dataset)):
        audio, target, name = dataset[index]
        predicted = model.generate(features(extractor, audio[None].to(args.device)),
                                   target_seq_length=cfg["data"]["max_token_length"])[0].tolist()
        report = serialize_prediction(predicted, Path(args.output) / (name + ".krn"), tokenizer=tokenizer)
        # Count invalid and truncated predictions too; never evaluate only successes.
        raw_ids = predicted[:predicted.index(2)] if 2 in predicted else predicted
        raw_bytes = b"".join(tokenizer.token_bytes(t) for t in raw_ids if t >= 3)
        raw = raw_bytes.decode("utf-8", errors="replace")
        truth = tokenizer.decode(target.tolist())
        reference_symbols = symbols(truth)
        report.update(id=name, category=dataset.entries[index].get("category", "unspecified"),
                      exact_match=report["valid"] and raw == truth,
                      token_exact_match=predicted == target.tolist(),
                      byte_edits=edit_distance(raw_bytes, truth.encode()), reference_bytes=len(truth.encode()),
                      symbol_edits=edit_distance(symbols(raw), reference_symbols), reference_symbols=len(reference_symbols))
        report.pop("tokens", None)
        results.append(report)
    def summarize(rows):
        return {"samples": len(rows), "valid_rate": sum(r["valid"] for r in rows) / len(rows),
                "exact_match_rate": sum(r["exact_match"] for r in rows) / len(rows),
                "byte_error_rate": sum(r["byte_edits"] for r in rows) / sum(r["reference_bytes"] for r in rows),
                "symbol_error_rate": sum(r["symbol_edits"] for r in rows) / sum(r["reference_symbols"] for r in rows)}
    report = {"split": args.split, "checkpoint_step": checkpoint["step"], **summarize(results),
              "by_category": {k: summarize([r for r in results if r["category"] == k]) for k in sorted({r["category"] for r in results})},
              "examples": results}
    write_json(Path(args.output) / "metrics.json", report)
    print(json.dumps(report, indent=2))


def main(command, argv=None):
    args = parser(command).parse_args(argv)
    torch.set_num_threads(args.threads)
    {"prepare": prepare, "train": train, "inference": inference, "evaluate": evaluate}[command](args)
