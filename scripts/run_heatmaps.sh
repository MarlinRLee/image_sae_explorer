#!/bin/bash -l
#SBATCH --job-name=heatmaps
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/heatmaps_%j.out

# Generic pre-compute heatmaps job — replaces all run_heatmaps_*.sh scripts.
#
# Required env vars (pass via sbatch --export or set in environment):
#   DATA_PATH  : path to explorer_data.pt
#   SAE_PATH   : path to SAE state dict (.pth)
#
# Optional env vars:
#   LAYER          : intermediate backbone layer (omit or set empty for final layer)
#   IMAGE_DIR      : primary image dir  (default: /scratch.global/lee02328/val)
#   EXTRA_IMAGE_DIR: secondary image dir (default: /scratch.global/lee02328/coco/val2017)
#
# Example — submit directly:
#   sbatch --job-name=heatmaps_dino_final \
#          --output=logs/heatmaps_dino_final_%j.out \
#          --export=DATA_PATH=.../explorer_data_d32000_k160_val.pt,SAE_PATH=.../sae.pth \
#          run_heatmaps.sh
#
# Or submit via submit_all_heatmaps.sh which handles all models in one call.

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATA_PATH="${DATA_PATH:?DATA_PATH must be set (e.g. via sbatch --export)}"
SAE_PATH="${SAE_PATH:?SAE_PATH must be set (e.g. via sbatch --export)}"
IMAGE_DIR="${IMAGE_DIR:-/scratch.global/lee02328/val}"
EXTRA_IMAGE_DIR="${EXTRA_IMAGE_DIR:-/scratch.global/lee02328/coco/val2017}"
LAYER_FLAG="${LAYER:+--layer $LAYER}"
FORCE_SPATIAL_FLAG="${FORCE_SPATIAL:+--force-spatial}"

echo "============================================"
echo "Pre-computing heatmaps"
echo "  Data:       $DATA_PATH"
echo "  SAE:        $SAE_PATH"
echo "  Layer:      ${LAYER:-final}"
echo "  Image dirs: $IMAGE_DIR + $EXTRA_IMAGE_DIR"
echo "============================================"

python scripts/precompute_heatmaps.py \
    --data            "$DATA_PATH" \
    --sae-path        "$SAE_PATH" \
    --image-dir       "$IMAGE_DIR" \
    --extra-image-dir "$EXTRA_IMAGE_DIR" \
    --batch-size      64 \
    --num-workers     8 \
    $LAYER_FLAG \
    $FORCE_SPATIAL_FLAG

echo "============================================"
echo "Done: $(ls -lh ${DATA_PATH%.pt}_heatmaps.pt 2>/dev/null || echo 'output not found')"
echo "============================================"
