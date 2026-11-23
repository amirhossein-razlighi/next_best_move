#!/bin/bash
# Download released LoRA checkpoints (one directory per model, each with pytorch_lora_weights.safetensors) into
# ${NBM_CHECKPOINTS:-./checkpoints}.
# usage: scripts/download_checkpoints.sh <name> [<name> ...]   names: chess_image chess_video shortest_path_image
#   shortest_path_video terrain_image terrain_video connect4_image connect4_video go_image billiards_image billiards_video
#   nqueens_image
# Evaluate one with: python -m nbm.evaluate --config configs/<task>/<image|video>.yaml --checkpoint checkpoints/<name>
# TODO(release): fill in BASE_URL once the weights are uploaded.
set -euo pipefail
BASE_URL="TODO"
DEST=${NBM_CHECKPOINTS:-./checkpoints}
[ "$BASE_URL" = "TODO" ] && { echo "download links are not published yet"; exit 1; }
mkdir -p "$DEST"
for name in "$@"; do
  mkdir -p "$DEST/$name"
  curl -L --fail -o "$DEST/$name/pytorch_lora_weights.safetensors" "$BASE_URL/$name/pytorch_lora_weights.safetensors"
done
