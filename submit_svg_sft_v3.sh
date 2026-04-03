#!/bin/bash
#SBATCH --job-name=svg_sft_v3
#SBATCH --output=/scratch/hk4488/SVG-Generation/logs/sft_v3_%j.log
#SBATCH --error=/scratch/hk4488/SVG-Generation/logs/sft_v3_%j.err
#SBATCH --chdir=/home/hk4488/SVG-Generation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --account=torch_pr_627_general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4488@nyu.edu
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
ENV_PREFIX=/scratch/hk4488/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

CLEAN_CSV=/home/hk4488/SVG-Generation/data/train_clean.csv
OUTPUT_DIR=/scratch/hk4488/SVG-Generation/outputs/svg_sft_v3
LOG_DIR=/scratch/hk4488/SVG-Generation/logs

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p /scratch/hk4488/.cache/huggingface
mkdir -p /scratch/hk4488/.cache/torch

# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export HF_HOME=/scratch/hk4488/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/hk4488/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/hk4488/.cache/huggingface/datasets
export TORCH_HOME=/scratch/hk4488/.cache/torch
export WANDB_PROJECT="svg-generation-sft"
export WANDB_API_KEY="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"
export WANDB_RUN_NAME="svg_sft_v3_r64_3ep_lr5e5"

# ── Diagnostics ────────────────────────────────────────────────────────────────
echo "===== Job diagnostics ====="
echo "Job ID  : $SLURM_JOB_ID"
echo "Host    : $(hostname)"
echo "Python  : $PYTHON"
$PYTHON -V
nvidia-smi
echo "==========================="

# ── Train ──────────────────────────────────────────────────────────────────────
$PYTHON train_svg.py \
    --train-csv "$CLEAN_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" \
    --max-seq-length 8192 \
    --lora-r 64 \
    --lora-alpha 64 \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-5 \
    --warmup-steps 50 \
    --num-train-epochs 3 \
    --logging-steps 20 \
    --save-steps 200 \
    --eval-steps 200 \
    --report-to wandb
