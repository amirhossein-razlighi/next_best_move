#!/bin/bash
# Download the paper's datasets into ${NBM_DATA:-./data} (one directory per task, as named by data.dir in the configs).
# Every dataset can also be rebuilt with its task's data builder (docs/tasks/<task>.md).
# TODO(release): fill in BASE_URL once the datasets are uploaded.
set -euo pipefail
BASE_URL="TODO"
DEST=${NBM_DATA:-./data}
[ "$BASE_URL" = "TODO" ] && { echo "download links are not published yet; build the data with the task builders (docs/tasks/)"; exit 1; }
mkdir -p "$DEST"
for name in "$@"; do
  curl -L --fail "$BASE_URL/$name.tar" | tar -x -C "$DEST"
done
