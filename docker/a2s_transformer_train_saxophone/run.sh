#!/usr/bin/env bash
set -euo pipefail

cd /workspace/efficient-seq2seq-piano-trans
output_dir=/workspace/outputs
config="$output_dir/score_saxophone.yaml"
threads="${A2S_THREADS:-4}"

if (( $# )); then
    echo "Este lanzador entrena desde cero y no acepta argumentos ni checkpoints." >&2
    exit 1
fi

# Fail before preparing the corpus if the assigned device cannot meet the request.
python - <<'PY'
import json
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Se requiere exactamente una GPU CUDA visible dentro de Docker")
gpu = torch.cuda.get_device_properties(0)
if gpu.total_memory < 20 * 1024**3:
    raise SystemExit(f"La GPU {gpu.name} tiene menos de 20 GiB de VRAM")
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": gpu.name, "vram_gib": round(gpu.total_memory / 1024**3, 2),
                  "initialization": "random", "task": "audio_to_score"}), flush=True)
PY

# Refuse to reuse a run, including a partially completed one.
mkdir "$output_dir/run"

python -m a2s.import_data saxophone \
    --root /workspace/data/saxophone \
    --partitions-dir a2s_partitions --audio-dir wav --score-dir krn_clean \
    --output "$output_dir"

# Fit on train only; validate complete pairs and cache log-mels on disk.
python -u prepare.py --task audio_to_score \
    --device cpu --threads "$threads" \
    --config "$config" \
    --manifest "$output_dir/manifest.jsonl" \
    --fit-tokenizer "$output_dir/vocab.json" \
    --cache-dir "$output_dir/features" \
    --output "$output_dir/prepared.jsonl"

# Omitting --resume initializes every model parameter randomly.
python -u train.py --task audio_to_score \
    --device cuda --threads "$threads" \
    --config "$config" \
    --tokenizer "$output_dir/vocab.json" \
    --manifest "$output_dir/prepared.jsonl" \
    --cache-dir "$output_dir/features" \
    --output "$output_dir/run"

python - <<'PY'
import json
from pathlib import Path
root = Path('/workspace/outputs')
for name in ('best.pt', 'latest.pt'):
    path = root / 'run' / name
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit(f"Falta el checkpoint final: {path}")
last = json.loads((root / 'run/training.jsonl').read_text().splitlines()[-1])
(root / 'complete.json').write_text(json.dumps({"task": "audio_to_score",
    "initialization": "random", "step": last["step"],
    "early_stopped": last.get("early_stopped", False)}, indent=2) + '\n')
PY
