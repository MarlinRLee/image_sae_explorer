#!/usr/bin/env bash
# Chain the two precompute steps for one SAE into the sidecars the explorer
# serves:
#
#   1. precompute_explorer_data.py  -> <output>.pt        (top images, UMAP, stats)
#   2. precompute_heatmaps.py       -> <output>_heatmaps.pt (per-feature heatmaps)
#
# Both steps need a GPU. The shared flags (--sae-path / --image-dir / --layer /
# --backbone / --token-type) are passed through to whichever step uses them, so
# you give them once. Any extra flags after `--` go to step 1
# (precompute_explorer_data.py), e.g. --d-model, --top-k, --interleave-classes,
# --coverage-threshold, --recursive.
#
# Usage:
#   bash scripts/precompute_all.sh \
#       --sae-path   /path/to/sae_..._k160.pth \
#       --image-dir  /scratch/val \
#       --output     explorer_data_my_sae.pt \
#       --backbone   dinov3 \
#       [--layer 24] [--token-type spatial] [--extra-image-dir DIR] \
#       -- --d-model 32000 --top-k 160 --interleave-classes
#
# This is a convenience wrapper; for a one-off run you can still call the two
# scripts directly (see docs/ADD_YOUR_OWN.md).
set -euo pipefail

SAE_PATH="" IMAGE_DIR="" OUTPUT="" BACKBONE="dinov3"
LAYER="" TOKEN_TYPE="spatial" EXTRA_IMAGE_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sae-path)        SAE_PATH="$2"; shift 2;;
    --image-dir)       IMAGE_DIR="$2"; shift 2;;
    --output|--output-path) OUTPUT="$2"; shift 2;;
    --backbone)        BACKBONE="$2"; shift 2;;
    --layer)           LAYER="$2"; shift 2;;
    --token-type)      TOKEN_TYPE="$2"; shift 2;;
    --extra-image-dir) EXTRA_IMAGE_DIR="$2"; shift 2;;
    --)                shift; EXTRA_ARGS=("$@"); break;;
    -h|--help)         sed -n '2,30p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$SAE_PATH" || -z "$IMAGE_DIR" || -z "$OUTPUT" ]]; then
  echo "ERROR: --sae-path, --image-dir, and --output are required." >&2
  echo "Run 'bash scripts/precompute_all.sh --help' for usage." >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

LAYER_ARGS=()
[[ -n "$LAYER" ]] && LAYER_ARGS=(--layer "$LAYER")
EXTRA_DIR_ARGS=()
[[ -n "$EXTRA_IMAGE_DIR" ]] && EXTRA_DIR_ARGS=(--extra-image-dir "$EXTRA_IMAGE_DIR")

echo "==> [1/2] precompute_explorer_data.py -> $OUTPUT"
python "$HERE/precompute_explorer_data.py" \
  --sae-path "$SAE_PATH" \
  --image-dir "$IMAGE_DIR" \
  --output-path "$OUTPUT" \
  --backbone "$BACKBONE" \
  --token-type "$TOKEN_TYPE" \
  "${LAYER_ARGS[@]}" "${EXTRA_DIR_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "==> [2/2] precompute_heatmaps.py -> ${OUTPUT%.pt}_heatmaps.pt"
python "$HERE/precompute_heatmaps.py" \
  --data "$OUTPUT" \
  --sae-path "$SAE_PATH" \
  --image-dir "$IMAGE_DIR" \
  "${LAYER_ARGS[@]}" "${EXTRA_DIR_ARGS[@]}"

echo "==> Done. Upload $OUTPUT + ${OUTPUT%.pt}_heatmaps.pt + the SAE .pth to your"
echo "    HF dataset repo (see scripts/upload_hf.sh), then add a registry block."
