#!/bin/bash
# Download the paper's datasets from Hugging Face (datasets/AmirHossein-razlighi/next-best-move) into ${NBM_DATA:-./data},
# one directory per task, as named by data.dir in the configs. Every dataset can also be rebuilt with its task's data
# builder (docs/tasks/<task>.md).
# usage: scripts/download_data.sh <name> [<name> ...] | all   names: chess shortest_path terrain connect4 go billiards nqueens
set -euo pipefail
REPO=AmirHossein-razlighi/next-best-move
NAMES="chess shortest_path terrain connect4 go billiards nqueens"
DEST=${NBM_DATA:-./data}
[ $# -eq 0 ] && { echo "usage: $0 <name> [<name> ...] | all   names: $NAMES"; exit 1; }
[ "$1" = "all" ] && set -- $NAMES
for name in "$@"; do
  [[ " $NAMES " == *" $name "* ]] || { echo "unknown dataset: $name (names: $NAMES)"; exit 1; }
  hf download "$REPO" --repo-type dataset --include "$name/*" --local-dir "$DEST"
done
