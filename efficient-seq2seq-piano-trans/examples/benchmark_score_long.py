"""Exercise the full-size model on complete pairs; not an accuracy benchmark.

python -m examples.benchmark_score_long --manifest outputs/score-long/manifest.jsonl
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from a2s.data import ScoreDataset, read_manifest
from a2s.runtime import configuration, features, frontend
from a2s.tokenizer import KernFieldTokenizer
from a2s.training import autocast, precision_mode
from model.T5 import Transformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="outputs/score-long/benchmark.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    cfg = configuration("config/score_long.yaml")
    tok = KernFieldTokenizer.fit(Path(r["score"]).read_text() for r in read_manifest(args.manifest) if r["split"] == "train")
    cfg["tokenizer"], cfg["model"]["vocab_size"] = tok.metadata(), tok.vocab_size
    dataset = ScoreDataset(args.manifest, cfg["data"], tokenizer=tok, external=True)
    model = Transformer(cfg["model"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    extractor = frontend(cfg, args.device)
    mode = precision_mode(cfg["training"], args.device)
    results = []
    cuda = args.device.startswith("cuda")
    for index in range(len(dataset)):
        model.train()
        model.decoder.initialize_decoder_cache()
        optimizer.zero_grad(set_to_none=True)
        wave, y, name = dataset[index]
        x = features(extractor, wave[None].to(args.device))
        y = y[None].to(args.device)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with autocast(mode, args.device):
            logits = model(x, y)["decoder_outputs"]
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise RuntimeError("Non-finite gradients")
        optimizer.step()
        if cuda:
            torch.cuda.synchronize()
        row = {"id": name, "seconds_audio": len(wave) / cfg["data"]["sample_rate"],
               "frames": x.size(1), "tokens": y.size(1), "loss": loss.item(),
               "gradient_norm": gradient.item(), "step_seconds": round(time.monotonic() - started, 3),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if cuda else None}
        model.eval()
        with autocast(mode, args.device):
            row["generated_tokens_smoke_only"] = model.generate(x, 8).size(1)
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {"config": cfg, "device": torch.cuda.get_device_name() if cuda else args.device,
              "torch": torch.__version__, "precision": mode,
              "parameters": sum(p.numel() for p in model.parameters()), "examples": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
