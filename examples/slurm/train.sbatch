#!/bin/bash
# Train one paper model on a Slurm cluster (one node). Adapt the #SBATCH lines to your cluster.
#   mkdir -p logs && CONFIG=configs/chess/image.yaml sbatch examples/slurm/train.sbatch [key.path=value ...]
# Image model with model.split_devices: true -> one process per GPU *pair* (8 GPUs = 4 processes); otherwise one per GPU.
# The paper's global batch is train.batch_size x train.grad_accum x train.processes; with fewer processes raise grad_accum.
# Resume after a time limit: resubmit with train.resume=latest (same data stream, optimizer and schedule).
#SBATCH -J nbm_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%j.out
set -euo pipefail
: "${CONFIG:?set CONFIG=configs/<task>/<image|video>.yaml}"
export NBM_MODELS=${NBM_MODELS:-$PWD/models} NBM_DATA=${NBM_DATA:-$PWD/data} NBM_OUTPUTS=${NBM_OUTPUTS:-$PWD/outputs}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=$(nvidia-smi -L | wc -l)
if grep -qE "^family: *image" "$CONFIG" && grep -qE "split_devices: *true" "$CONFIG"; then NPROC=$((GPUS / 2)); else NPROC=$GPUS; fi
echo "train $CONFIG | node $(hostname) | $GPUS GPUs, $NPROC processes | overrides: $*"
torchrun --standalone --nproc_per_node "$NPROC" -m nbm.train --config "$CONFIG" "$@"
