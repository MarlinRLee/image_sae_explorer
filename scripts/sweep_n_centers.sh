#!/bin/bash -l
#SBATCH --job-name=sweep_nc
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/sweep_nc_%j.out

# Sweep over different numbers of K-means centers for SI-SAE (d_model=20000).
# Atoms beyond n_centers are initialized as pure random noise.
#
# Usage: sbatch sweep_n_centers.sh
# Or to run a single value: sbatch sweep_n_centers.sh 15000

ulimit -n 65536

# Use local disk for Python multiprocessing temp files (avoids NFS .nfs* cleanup errors)
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/src

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Data directories ---
IN_TRAIN_DIR="/scratch.global/lee02328/data/DINOV3/train/final"
COCO_TRAIN_DIR="/scratch.global/lee02328/data/DINOV3/coco_train/final"
VAL_DIR="/scratch.global/lee02328/data/DINOV3/coco_val/final"

# --- Create combined training directory with symlinks ---
# Use flock to prevent races when multiple jobs run in parallel
COMBINED_DIR="/scratch.global/lee02328/data/DINOV3/combined_train"
LOCK_FILE="/scratch.global/lee02328/data/DINOV3/.combined_train.lock"
mkdir -p "$COMBINED_DIR"

(
    flock -x 200

    # Check if symlinks already look correct (skip rebuild if so)
    EXPECTED_IN=$(ls "$IN_TRAIN_DIR"/shard_*.pt 2>/dev/null | wc -l)
    EXPECTED_COCO=$(ls "$COCO_TRAIN_DIR"/shard_*.pt 2>/dev/null | wc -l)
    EXPECTED_TOTAL=$((EXPECTED_IN + EXPECTED_COCO))
    CURRENT_TOTAL=$(ls "$COMBINED_DIR"/shard_*.pt 2>/dev/null | wc -l)

    if [ "$CURRENT_TOTAL" -ne "$EXPECTED_TOTAL" ]; then
        echo "Rebuilding combined dir ($CURRENT_TOTAL != $EXPECTED_TOTAL expected)"
        find "$COMBINED_DIR" -maxdepth 1 -type l -delete 2>/dev/null

        for f in "$IN_TRAIN_DIR"/shard_*.pt; do
            ln -sf "$f" "$COMBINED_DIR/$(basename "$f")"
        done

        IDX=$EXPECTED_IN
        for f in "$COCO_TRAIN_DIR"/shard_*.pt; do
            ln -sf "$f" "$COMBINED_DIR/shard_$(printf '%04d' $IDX).pt"
            IDX=$((IDX + 1))
        done
    else
        echo "Combined dir already has $CURRENT_TOTAL shards, skipping rebuild"
    fi
) 200>"$LOCK_FILE"

TOTAL_SHARDS=$(ls "$COMBINED_DIR"/shard_*.pt 2>/dev/null | wc -l)
echo "Total combined shards: $TOTAL_SHARDS"

CHECKPOINT_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/checkpoints/dinov3_l24_spatial"
OUTPUT_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/models/dinov3_l24_spatial"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$OUTPUT_DIR"

# --- N_CENTERS values to sweep ---
if [ -n "$1" ]; then
    # Single value provided as argument
    N_CENTERS_LIST=("$1")
else
    # Default sweep: from fewer centers to full K-means
    N_CENTERS_LIST=(5000 10000 15000 20000)
fi

for NC in "${N_CENTERS_LIST[@]}"; do
    echo "============================================"
    echo "Running SI-SAE with n_centers=$NC (d_model=20000)"
    echo "============================================"

    python main.py "$COMBINED_DIR" SI-SAE \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --checkpoint-every 5 \
        --val-dir "$VAL_DIR" \
        --mixed-precision \
        --n-centers "$NC"

    echo "Finished n_centers=$NC"
    echo ""
done

echo "============================================"
echo "Sweep complete!"
echo "============================================"
