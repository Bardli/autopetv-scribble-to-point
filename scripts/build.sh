#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${1:-autopetv-scribble-to-point:2fold-8tta}"

python3 "$repo_dir/scripts/verify_model.py" \
  --model-dir "$repo_dir/third_party/autoPET-interactive/_model"
docker build --tag "$image_tag" "$repo_dir"
