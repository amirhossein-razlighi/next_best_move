#!/bin/bash
# Download the two base models into $NBM_MODELS (default ./models):
#   Qwen/Qwen-Image-2.1               -> $NBM_MODELS/Qwen-Image-2.1             (image model, ~31 GB)
#   Wan-AI/Wan2.2-TI2V-5B-Diffusers   -> $NBM_MODELS/Wan2.2-TI2V-5B-Diffusers   (video model, ~32 GB)
# usage: scripts/download_base_models.sh [image|video|all]   (default: all)
# Uses the `hf` command of huggingface_hub (installed with this package; the older `huggingface-cli` no longer works with
# huggingface_hub >= 1.0). Set HF_TOKEN (or run `hf auth login`) for faster downloads.
set -euo pipefail
ROOT=${NBM_MODELS:-./models}
WHICH=${1:-all}
mkdir -p "$ROOT"
get() { echo "downloading $1 -> $ROOT/$2"; hf download "$1" --local-dir "$ROOT/$2"; }
case $WHICH in
  image) get Qwen/Qwen-Image-2.1 Qwen-Image-2.1 ;;
  video) get Wan-AI/Wan2.2-TI2V-5B-Diffusers Wan2.2-TI2V-5B-Diffusers ;;
  all) get Qwen/Qwen-Image-2.1 Qwen-Image-2.1
       get Wan-AI/Wan2.2-TI2V-5B-Diffusers Wan2.2-TI2V-5B-Diffusers ;;
  *) echo "usage: $0 [image|video|all]"; exit 1 ;;
esac
