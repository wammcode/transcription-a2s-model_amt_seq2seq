#!/usr/bin/env bash
#SBATCH --job-name=a2s-saxophone-seq2seq
#SBATCH --output=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq/docker/a2s_transformer_train_saxophone/%x-%j.out
#SBATCH --error=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq/docker/a2s_transformer_train_saxophone/%x-%j.err
#SBATCH --chdir=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --time=2-00:00:00

set -euo pipefail

# Slurm runs a copy from /var/spool; --chdir supplies the project root.
project_root="$(pwd -P)"
model_dir="$project_root/efficient-seq2seq-piano-trans"
data_dir="$project_root/data/data_saxophone/real_a2s_sax_dataset"
docker_dir="$project_root/docker/a2s_transformer_train_saxophone"
run_id="${SLURM_JOB_ID:?Submit this script with sbatch}"
output_dir="$model_dir/outputs/saxophone/slurm-$run_id"
image_name="efficient-seq2seq-a2s:saxophone-$run_id"
container_name="a2s-saxophone-$run_id"

for split in train val test; do
    test -s "$data_dir/a2s_partitions/$split.txt" || {
        echo "Falta la partición $split en $data_dir/a2s_partitions" >&2
        exit 1
    }
done
for directory in "$model_dir/a2s" "$data_dir/wav" "$data_dir/krn_clean"; do
    test -d "$directory" || { echo "Falta $directory" >&2; exit 1; }
done
command -v docker >/dev/null || { echo "Docker no está disponible en $(hostname)." >&2; exit 1; }

# Use physical GPU IDs assigned by Slurm, even when CUDA renumbers them.
gpu_device="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$gpu_device" || "$gpu_device" == *,* ]]; then
    echo "Se requiere exactamente una GPU asignada por Slurm: '$gpu_device'." >&2
    exit 1
fi

# Each submission starts a separate run; never resume or overwrite checkpoints.
mkdir -p "$(dirname "$output_dir")"
mkdir "$output_dir"
cp "$docker_dir/p1.slurm" "$docker_dir/Dockerfile" "$docker_dir/requirements.txt" "$docker_dir/run.sh" "$output_dir/"
cp "$model_dir/config/score_saxophone.yaml" "$output_dir/score_saxophone.yaml"

cleanup() {
    # Stop our container if Slurm cancels the job or its time limit expires.
    docker stop --time 30 "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "Audio to Score Saxophone desde cero en $(hostname), $(date --iso-8601=seconds)"
echo "GPU física: $gpu_device; salida: $output_dir"
docker build -t "$image_name" "$docker_dir"
docker image inspect "$image_name" --format '{{.Id}}' > "$output_dir/docker-image-id.txt"

docker run --rm --init --name "$container_name" \
    --gpus "device=$gpu_device" \
    --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$model_dir,dst=/workspace/efficient-seq2seq-piano-trans,readonly" \
    --mount "type=bind,src=$data_dir,dst=/workspace/data/saxophone,readonly" \
    --mount "type=bind,src=$output_dir,dst=/workspace/outputs" \
    --env A2S_THREADS=4 \
    --env OMP_NUM_THREADS=4 \
    --env MKL_NUM_THREADS=4 \
    "$image_name"

test -s "$output_dir/complete.json" || {
    echo "El contenedor no completó el entrenamiento; revise los registros." >&2
    exit 1
}
echo "Entrenamiento terminado: $(date --iso-8601=seconds)"
echo "Checkpoints: $output_dir/run/best.pt y $output_dir/run/latest.pt"
