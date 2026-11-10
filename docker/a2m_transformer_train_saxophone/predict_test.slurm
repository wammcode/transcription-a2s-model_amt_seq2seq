#!/usr/bin/env bash
#SBATCH --job-name=a2m-saxophone-test
#SBATCH --output=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq/docker/a2m_transformer_train_saxophone/%x-%j.out
#SBATCH --error=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq/docker/a2m_transformer_train_saxophone/%x-%j.err
#SBATCH --chdir=/home/grfia/wilson/project_2026_1/transcription-a2s-model_amt_seq2seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --time=2-00:00:00

set -euo pipefail
dataset=saxophone
data_dir="$(pwd -P)/data/data_saxophone/real_a2s_sax_dataset"
partitions=amt_partitions/saxophone_amt
source "$(pwd -P)/docker/a2m_transformer_common/predict_launch.sh"
