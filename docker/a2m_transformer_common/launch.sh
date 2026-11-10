#!/usr/bin/env bash
# Sourced by each dataset's Slurm script after setting dataset and partitions.
set -euo pipefail
project_root="$(pwd -P)"
model_dir="$project_root/efficient-seq2seq-piano-trans"
docker_dir="$project_root/docker/a2m_transformer_train_$dataset"
common_dir="$project_root/docker/a2m_transformer_common"
run_id="${SLURM_JOB_ID:?Submit p1.slurm with sbatch}"
output_dir="$model_dir/outputs/audio_to_midi/$dataset/slurm-$run_id"
image_name="efficient-seq2seq-a2m:$dataset-$run_id"
container_name="wilson-a2m-$dataset-$run_id"

command -v docker >/dev/null || { echo "Docker no está disponible en $(hostname)" >&2; exit 1; }
gpu_device="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$gpu_device" || "$gpu_device" == *,* ]]; then
    echo "Se requiere exactamente una GPU reservada por Slurm: '$gpu_device'" >&2
    exit 1
fi
for split in train val test; do
    test -s "$data_dir/$partitions/$split.txt"
done
test -d "$data_dir/wav"
test -d "$data_dir/midi"
mkdir -p "$(dirname "$output_dir")"
mkdir "$output_dir"
mkdir "$output_dir/job" "$output_dir/source"
cp "$docker_dir/"{p1.slurm,Dockerfile,requirements.txt,run.sh,predict_test.sh,predict_test.slurm,midi.yaml} "$output_dir/job/"
cp "$common_dir/"{pipeline.py,launch.sh,test_entrypoint.sh} "$output_dir/job/"
# Snapshot only source, never existing outputs, checkpoints or datasets.
for entry in config model data utils loss task_dispatch.py; do
    cp -r "$model_dir/$entry" "$output_dir/source/"
done
git -C "$project_root" rev-parse HEAD > "$output_dir/source-commit.txt"
git -C "$project_root" diff --binary > "$output_dir/source-working-tree.patch"

container_started=0
cleanup() {
    if (( container_started )); then
        # Only this job's uniquely named container; never touch another process.
        docker stop --time 30 "$container_name" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
echo "Audio to MIDI $dataset: nodo=$(hostname), GPU=$gpu_device, salida=$output_dir"
docker build -t "$image_name" "$output_dir/job"
docker image inspect "$image_name" --format '{{.Id}}' > "$output_dir/docker-image-id.txt"
# Verify the image launcher with the actual training UID/GID before processing data.
docker run --rm --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output_dir/job,dst=/workspace/check,readonly" \
    "$image_name" bash /workspace/check/test_entrypoint.sh | tee "$output_dir/entrypoint-check.log"
grep -q '^PASS:' "$output_dir/entrypoint-check.log"
container_started=1
docker run --rm --init --name "$container_name" \
    --label "a2m.slurm-job=$run_id" --gpus "device=$gpu_device" \
    --shm-size=8g --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output_dir/source,dst=/workspace/efficient-seq2seq-piano-trans,readonly" \
    --mount "type=bind,src=$data_dir,dst=/workspace/data,readonly" \
    --mount "type=bind,src=$output_dir,dst=/workspace/outputs" \
    --env "A2M_PARTITIONS=$partitions" --env A2M_THREADS=4 \
    --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 "$image_name"
container_started=0
test -s "$output_dir/complete.json"
echo "Entrenamiento e inferencia terminados: $output_dir"
