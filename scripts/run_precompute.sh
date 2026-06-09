#!/bin/bash -l
#SBATCH --job-name=precompute
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/precompute_%j.out

# Generic precompute explorer data job — replaces all run_precompute_*.sh scripts.
#
# Required env vars (pass via sbatch --export or set in environment):
#   SAE_PATH    : path to SAE state dict (.pth)
#   OUTPUT_PATH : path for the output explorer_data.pt
#   D_MODEL     : number of SAE features (e.g. 32000)
#   TOP_K       : SAE sparsity (e.g. 160)
#
# Optional env vars:
#   BACKBONE          : dinov3 (default) | clip
#   TOKEN_TYPE        : spatial (default) | cls
#   LAYER             : intermediate layer index (omit/empty for final layer)
#   IMAGE_DIR         : (default: /scratch.global/lee02328/val)
#   EXTRA_IMAGE_DIR   : (default: /scratch.global/lee02328/coco/val2017)
#   HEATMAP_IMAGE_DIR : image dir for heatmaps step (default: IMAGE_DIR)
#
# Example:
#   sbatch --job-name=precompute_cls \
#          --output=logs/precompute_cls_%j.out \
#          --export=SAE_PATH=.../sae.pth,OUTPUT_PATH=.../explorer_data_cls.pt,D_MODEL=20000,TOP_K=80,TOKEN_TYPE=cls \
#          run_precompute.sh

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SAE_PATH="${SAE_PATH:?SAE_PATH must be set}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH must be set}"
D_MODEL="${D_MODEL:?D_MODEL must be set}"
TOP_K="${TOP_K:?TOP_K must be set}"
BACKBONE="${BACKBONE:-dinov3}"
TOKEN_TYPE="${TOKEN_TYPE:-spatial}"
IMAGE_DIR="${IMAGE_DIR:-/scratch.global/lee02328/val}"
EXTRA_IMAGE_DIR="${EXTRA_IMAGE_DIR:-/scratch.global/lee02328/coco/val2017}"
LAYER_FLAG="${LAYER:+--layer $LAYER}"
THUMBNAIL_FLAG="${THUMBNAIL_DIR:+--thumbnail-dir $THUMBNAIL_DIR}"
PATCH_NORM_FLAG="${PATCH_NORM_STATS:+--patch-norm-stats $PATCH_NORM_STATS}"

echo "============================================"
echo "Precomputing SAE Explorer Data"
echo "  Backbone:   $BACKBONE | token: $TOKEN_TYPE | layer: ${LAYER:-final}"
echo "  d_model:    $D_MODEL  | top_k: $TOP_K"
echo "  SAE:        $SAE_PATH"
echo "  Output:     $OUTPUT_PATH"
echo "============================================"

python scripts/precompute_explorer_data.py \
    --backbone        "$BACKBONE" \
    --token-type      "$TOKEN_TYPE" \
    --image-dir       "$IMAGE_DIR" \
    --extra-image-dir "$EXTRA_IMAGE_DIR" \
    --sae-path        "$SAE_PATH" \
    --output-path     "$OUTPUT_PATH" \
    --d-model         $D_MODEL \
    --top-k           $TOP_K \
    --top-n           16 \
    --batch-size      64 \
    --num-workers     8 \
    --recursive \
    --umap-reservoir  50000 \
    --interleave-classes \
    $LAYER_FLAG \
    $THUMBNAIL_FLAG \
    $PATCH_NORM_FLAG

echo "============================================"
echo "Precompute complete: $OUTPUT_PATH"
echo "============================================"

HEATMAP_IMAGE_DIR="${HEATMAP_IMAGE_DIR:-$IMAGE_DIR}"

echo "============================================"
echo "Precomputing heatmaps"
echo "  Data:       $OUTPUT_PATH"
echo "  Image dir:  $HEATMAP_IMAGE_DIR"
echo "============================================"

python scripts/precompute_heatmaps.py \
    --data            "$OUTPUT_PATH" \
    --sae-path        "$SAE_PATH" \
    --image-dir       "$HEATMAP_IMAGE_DIR" \
    --extra-image-dir "$EXTRA_IMAGE_DIR" \
    --batch-size      64 \
    --num-workers     8 \
    $LAYER_FLAG \
    $PATCH_NORM_FLAG

echo "============================================"
echo "Heatmaps complete: ${OUTPUT_PATH%.pt}_heatmaps.pt"
echo "============================================"

echo "============================================"
echo "Adding CLIP text alignment scores"
echo "  Data:  $OUTPUT_PATH"
echo "============================================"

python scripts/add_clip_embeddings.py \
    --data            "$OUTPUT_PATH" \
    --image-dir       "$IMAGE_DIR" \
    --extra-image-dir "$EXTRA_IMAGE_DIR" \
    --extra-image-dir /scratch.global/lee02328/coco/train2017 \
    --n-top-images    4 \
    --batch-size      64

echo "============================================"
echo "CLIP embeddings complete: $OUTPUT_PATH"
echo "============================================"
