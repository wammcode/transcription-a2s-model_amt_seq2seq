#!/usr/bin/env bash
set -euo pipefail
cd /workspace/efficient-seq2seq-piano-trans
export PYTHONPATH="/workspace/efficient-seq2seq-piano-trans${PYTHONPATH:+:$PYTHONPATH}"
test -s /workspace/outputs/training_complete.json
extra=()
if [[ -n "${A2M_PREDICTION_OUTPUT:-}" ]]; then
    extra=(--prediction-output "$A2M_PREDICTION_OUTPUT")
fi
python -u /workspace/outputs/job/pipeline.py predict --task audio_to_midi \
    --root /workspace/data --partitions "${A2M_PARTITIONS:?}" \
    --output /workspace/outputs --device cuda --threads "${A2M_THREADS:-4}" "${extra[@]}"
