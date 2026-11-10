#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then
    echo "Uso: sbatch predict_test.slurm /ruta/outputs/audio_to_midi/DATASET/slurm-TRAIN_JOB" >&2
    exit 1
fi
training_dir="$(realpath "$1")"
test -s "$training_dir/training_complete.json"
test -s "$training_dir/run/best.pt"
job_id="${SLURM_JOB_ID:?Submit with sbatch}"
output_dir="$training_dir/predictions/test/slurm-$job_id"
image_name="efficient-seq2seq-a2m:$dataset-predict-$job_id"
container_name="wilson-a2m-$dataset-predict-$job_id"
gpu_device="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$gpu_device" || "$gpu_device" == *,* ]]; then
    echo "Se requiere una GPU reservada por Slurm" >&2
    exit 1
fi
mkdir -p "$(dirname "$output_dir")"
mkdir "$output_dir"
docker build -t "$image_name" "$training_dir/job"
started=0
cleanup() {
    if (( started )); then docker stop --time 30 "$container_name" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
started=1
docker run --rm --init --name "$container_name" --gpus "device=$gpu_device" \
    --shm-size=8g --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$training_dir/source,dst=/workspace/efficient-seq2seq-piano-trans,readonly" \
    --mount "type=bind,src=$data_dir,dst=/workspace/data,readonly" \
    --mount "type=bind,src=$training_dir,dst=/workspace/outputs,readonly" \
    --mount "type=bind,src=$output_dir,dst=/workspace/predictions" \
    --env "A2M_PARTITIONS=$partitions" --env A2M_THREADS=4 \
    --env A2M_PREDICTION_OUTPUT=/workspace/predictions \
    --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    "$image_name" bash /workspace/outputs/job/predict_test.sh
started=0
test -s "$output_dir/complete.json"
echo "MIDI de test guardados en $output_dir"
