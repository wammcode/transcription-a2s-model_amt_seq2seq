#!/usr/bin/env bash
#SBATCH --job-name=a2m-quartets-seq2seq
#SBATCH --output=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq/docker/a2m_transformer_train_quartets/%x-%j.out
#SBATCH --error=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq/docker/a2m_transformer_train_quartets/%x-%j.err
#SBATCH --chdir=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --time=2-00:00:00

set -euo pipefail
dataset=quartets
data_dir="$(pwd -P)/data/data_quartets/Quartets"
partitions=amt_partitions/quartets
source "$(pwd -P)/docker/a2m_transformer_common/launch.sh"
