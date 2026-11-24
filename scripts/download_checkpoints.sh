#!/bin/bash
# Download released LoRA checkpoints from Hugging Face (AmirHossein-razlighi/next-best-move) into
# ${NBM_CHECKPOINTS:-./checkpoints}, one directory per model, each with pytorch_lora_weights.safetensors.
# usage: scripts/download_checkpoints.sh <name> [<name> ...] | all   names: chess_image chess_video shortest_path_image
#   shortest_path_video terrain_image terrain_video connect4_image connect4_video go_image billiards_image billiards_video
#   nqueens_image
# Evaluate one with: python -m nbm.evaluate --config configs/<task>/<image|video>.yaml --checkpoint checkpoints/<name>
set -euo pipefail
REPO=AmirHossein-razlighi/next-best-move
NAMES="chess_image chess_video shortest_path_image shortest_path_video terrain_image terrain_video connect4_image
  connect4_video go_image billiards_image billiards_video nqueens_image"
DEST=${NBM_CHECKPOINTS:-./checkpoints}
[ $# -eq 0 ] && { echo "usage: $0 <name> [<name> ...] | all   names: $(echo $NAMES)"; exit 1; }
[ "$1" = "all" ] && set -- $NAMES
for name in "$@"; do
  [[ " $(echo $NAMES) " == *" $name "* ]] || { echo "unknown checkpoint: $name (names: $(echo $NAMES))"; exit 1; }
  hf download "$REPO" --include "$name/*" --local-dir "$DEST"
done
