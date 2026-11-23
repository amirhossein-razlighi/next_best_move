#!/bin/bash
# Evaluate a checkpoint on a Slurm cluster (one node). Adapt the #SBATCH lines to your cluster.
#   mkdir -p logs && CONFIG=configs/chess/image.yaml CHECKPOINT=checkpoints/chess_image sbatch examples/slurm/evaluate.sbatch --split test [--steps 1] [--votes 8]
# Extra arguments go to nbm.evaluate (--split, --steps, --votes, --limit, --seed, --out, key.path=value overrides).
# Image model with model.split_devices: true -> one process per GPU pair; otherwise one per GPU. Results do not depend on the
# number of processes (the noise is drawn per row, or per fixed global batch).
#SBATCH -J nbm_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.out
set -euo pipefail
: "${CONFIG:?set CONFIG=configs/<task>/<image|video>.yaml}"
export NBM_MODELS=${NBM_MODELS:-$PWD/models} NBM_DATA=${NBM_DATA:-$PWD/data} NBM_OUTPUTS=${NBM_OUTPUTS:-$PWD/outputs}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=$(nvidia-smi -L | wc -l)
if grep -qE "^family: *image" "$CONFIG" && grep -qE "split_devices: *true" "$CONFIG"; then NPROC=$((GPUS / 2)); else NPROC=$GPUS; fi
echo "evaluate $CONFIG | checkpoint ${CHECKPOINT:-none} | node $(hostname) | $NPROC processes | args: $*"
torchrun --standalone --nproc_per_node "$NPROC" -m nbm.evaluate --config "$CONFIG" ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} "$@"
