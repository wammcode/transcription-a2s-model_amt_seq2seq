#!/usr/bin/env bash
#SBATCH --job-name=a2s-quartets-test
#SBATCH --output=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq/docker/a2s_transformer_train_quartets/%x-%j.out
#SBATCH --error=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq/docker/a2s_transformer_train_quartets/%x-%j.err
#SBATCH --chdir=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --time=2-00:00:00

set -euo pipefail
project_root="$(pwd -P)"
model_dir="$project_root/efficient-seq2seq-piano-trans"
data_dir="$project_root/data/data_quartets/Quartets"
docker_dir="$project_root/docker/a2s_transformer_train_quartets"
training_dir="$model_dir/outputs/quartets/slurm-106715"
job_id="${SLURM_JOB_ID:?Submit with sbatch}"
output_dir="$training_dir/predictions/test/slurm-$job_id"
image_name="efficient-seq2seq-a2s:quartets-106715"
container_name="a2s-quartets-test-$job_id"

for file in "$training_dir/run/best.pt" "$training_dir/prepared.jsonl" \
            "$training_dir/complete.json" "$data_dir/a2s_partitions/quartets/test.txt"; do
    test -s "$file" || { echo "Falta $file" >&2; exit 1; }
done
gpu_device="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$gpu_device" || "$gpu_device" == *,* ]]; then
    echo "Se requiere exactamente una GPU de Slurm: '$gpu_device'." >&2
    exit 1
fi
command -v docker >/dev/null
mkdir -p "$(dirname "$output_dir")"
mkdir "$output_dir"
cp "$docker_dir/predict_test.slurm" "$docker_dir/predict_test.sh" "$output_dir/"
cp "$model_dir/a2s/cli.py" "$output_dir/a2s_cli.py"

cleanup() { docker stop --time 30 "$container_name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "Predicciones test Quartets con best.pt de 106715: $(date --iso-8601=seconds)"
echo "Nodo: $(hostname); GPU: $gpu_device; salida: $output_dir"
# Reuse the training environment when available on this node.
if ! docker image inspect "$image_name" >/dev/null 2>&1; then
    docker build -t "$image_name" "$docker_dir"
fi
docker image inspect "$image_name" --format '{{.Id}}' > "$output_dir/docker-image-id.txt"
docker run --rm --init --name "$container_name" \
    --gpus "device=$gpu_device" --shm-size=8g --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$model_dir,dst=/workspace/efficient-seq2seq-piano-trans,readonly" \
    --mount "type=bind,src=$data_dir,dst=/workspace/data/quartets,readonly" \
    --mount "type=bind,src=$training_dir,dst=/workspace/training,readonly" \
    --mount "type=bind,src=$docker_dir,dst=/workspace/job,readonly" \
    --mount "type=bind,src=$output_dir,dst=/workspace/predictions" \
    --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    "$image_name" bash /workspace/job/predict_test.sh
test -s "$output_dir/complete.json"
echo "Predicciones terminadas: $(date --iso-8601=seconds); $output_dir"
