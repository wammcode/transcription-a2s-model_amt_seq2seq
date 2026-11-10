#!/usr/bin/env bash
set -euo pipefail
cd /workspace/efficient-seq2seq-piano-trans
if (( $# )); then
    echo "Este lanzador no acepta argumentos; configure midi.yaml antes de enviar el trabajo." >&2
    exit 1
fi
python - <<'CHECK'
import json
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Se requiere exactamente una GPU CUDA asignada por Slurm')
gpu = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info()
if gpu.total_memory < 20 * 1024**3 or free < 20 * 1024**3:
    raise SystemExit(f'Se requieren al menos 20 GiB totales y libres: {gpu.name}, {free / 1024**3:.2f} GiB libres')
metadata = {'task': 'audio_to_midi', 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': gpu.name, 'vram_gib': gpu.total_memory / 1024**3,
            'free_vram_gib': free / 1024**3, 'initialization': 'random'}
Path('/workspace/outputs/gpu.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata), flush=True)
CHECK
args=(--task audio_to_midi --root /workspace/data
      --partitions "${A2M_PARTITIONS:?}" --output /workspace/outputs
      --config /workspace/outputs/job/midi.yaml --device cuda --threads "${A2M_THREADS:-4}")
export PYTHONPATH="/workspace/efficient-seq2seq-piano-trans${PYTHONPATH:+:$PYTHONPATH}"
python -u /workspace/outputs/job/pipeline.py prepare "${args[@]}"
python -u /workspace/outputs/job/pipeline.py train "${args[@]}"
bash /workspace/outputs/job/predict_test.sh
