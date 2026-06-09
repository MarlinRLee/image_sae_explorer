#!/bin/bash -l
#SBATCH --job-name=layer_comparison
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/layer_comparison_%j.out

# Generate cross-layer SAE comparison figures for the demo paper.
# Compares the final-layer SAE (explorer_data_d32000_k160_val.pt)
# against the layer-18 SAE (explorer_data_18.pt).
#
# Produces PDF figures in figures/layer_comparison/:
#   - Summary statistics bar chart (L0, dead fraction, mean activation)
#   - Decoder norm distributions
#   - Side-by-side MEI examples per layer
#   - UMAP scatter plots per layer
#
# No new inference needed — runs purely from existing explorer_data*.pt files.

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

BASE_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"

cd "$BASE_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FINAL_DATA="$BASE_DIR/explorer_data_d32000_k160_val.pt"
LAYER18_DATA="$BASE_DIR/explorer_data_18.pt"
IMAGE_DIR="/scratch.global/lee02328/val"
EXTRA_IMAGE_DIR="/scratch.global/lee02328/coco/val2017"
OUTPUT_DIR="$BASE_DIR/figures/layer_comparison"

echo "============================================"
echo "Layer Comparison"
echo "  Final layer: $FINAL_DATA"
echo "  Layer 18:    $LAYER18_DATA"
echo "  Output:      $OUTPUT_DIR"
echo "============================================"

python analysis/layer_comparison.py \
    --final-data      "$FINAL_DATA" \
    --layer18-data    "$LAYER18_DATA" \
    --image-dir       "$IMAGE_DIR" \
    --extra-image-dir "$EXTRA_IMAGE_DIR" \
    --output-dir      "$OUTPUT_DIR" \
    --labels          "Final layer" "Layer 18" \
    --n-features      6 \
    --img-size        112

echo "============================================"
echo "Layer comparison figures complete!"
echo "============================================"
