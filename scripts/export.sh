#!/usr/bin/env bash
set -euo pipefail

image_tag="${1:-autopetv-scribble-to-point:2fold-8tta}"
output_path="${2:-autopetv-scribble-to-point_2fold-8tta.tar.gz}"

docker save "$image_tag" | gzip -c > "$output_path"
printf 'Wrote %s\n' "$output_path"
