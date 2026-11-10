#!/usr/bin/env bash
set -euo pipefail
cd /workspace/efficient-seq2seq-piano-trans

python -u - <<'PY'
import hashlib
import json
from pathlib import Path
import torch

root = Path('/workspace/predictions')
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Se requiere una GPU CUDA visible')
gpu = torch.cuda.get_device_properties(0)
if gpu.total_memory < 20 * 1024**3:
    raise SystemExit('Se requieren al menos 20 GiB de VRAM')
source = Path('/workspace/training/prepared.jsonl')
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
rows = [row for row in rows if row['split'] == 'test']
expected = [tuple(line.split()) for line in Path('/workspace/data/quartets/a2s_partitions/quartets/test.txt').read_text().splitlines() if line.strip()]
actual = [(Path(row['audio']).name, Path(row['score']).name) for row in rows]
if not rows or len(actual) != len(set(actual)) or len(expected) != len(actual) or set(expected) != set(actual):
    raise SystemExit('El manifiesto no coincide exactamente con la partición test original')
(root / 'test_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
checkpoint = Path('/workspace/training/run/best.pt')
with checkpoint.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
metadata = {'training_job': 106715, 'checkpoint': 'run/best.pt', 'checkpoint_sha256': digest,
            'split': 'test', 'samples': len(rows), 'gpu': gpu.name,
            'vram_gib': round(gpu.total_memory / 1024**3, 2), 'decoding': 'greedy',
            'precision': 'fp32', 'batch_size': 32, 'reference_conditioning': False}
(root / 'prediction_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata), flush=True)
PY

# Verify batch masks, reference-independent generation and invalid-output handling.
python -m pytest -q -p no:cacheprovider \
    tests/test_long_audio.py::test_variable_padding_does_not_change_predictions_or_pooling \
    tests/test_long_audio.py::test_batched_greedy_generation_matches_individual_audio \
    tests/test_long_audio.py::test_evaluation_compares_scores_not_tokenization_choices \
    tests/test_long_audio.py::test_batched_evaluation_preserves_invalid_outputs_and_test_split

python -u evaluate.py --task audio_to_score --device cuda --threads 4 \
    --checkpoint /workspace/training/run/best.pt \
    --manifest /workspace/predictions/test_manifest.jsonl --split test \
    --batch-size 32 --max-batch-frames 32000 \
    --cache-dir /workspace/training/features \
    --output /workspace/predictions/results

python - <<'PY'
import json
from pathlib import Path
root = Path('/workspace/predictions')
expected = json.loads((root / 'prediction_config.json').read_text())['samples']
metrics = json.loads((root / 'results/metrics.json').read_text())
if metrics['samples'] != expected or sum(1 for _ in (root / 'results/raw').glob('*.krn')) != expected:
    raise SystemExit('No se generaron todas las predicciones de test')
summary = {key: value for key, value in metrics.items() if key != 'examples'}
(root / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
PY
