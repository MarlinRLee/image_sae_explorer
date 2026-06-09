#!/bin/bash -l
#SBATCH --job-name=sae_train
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/sae_train_%j.out

# Generic SAE training job — replaces all per-model run_train_*.sh scripts.
#
# Required env vars (pass via sbatch --export or set in environment):
#   IN_TRAIN_DIR   : ImageNet train shard directory
#   COCO_TRAIN_DIR : COCO train shard directory
#   VAL_DIR        : Validation shard directory
#   COMBINED_DIR   : Path for the combined symlink directory
#   CHECKPOINT_DIR : Where to save checkpoints
#   OUTPUT_DIR     : Where to save the final trained model
#
# Optional env vars:
#   MODEL_TYPE      : SI-SAE (default) or SAE
#   D_MODEL         : Dictionary size   (passed as --d-model if set)
#   K_FRACTION      : k/d_model ratio   (passed as --k-fraction if set)
#   PER_INIT        : Per-init value    (passed as --per-init if set)
#   REANIMATE_COEFF : Default 0.33      (passed as --reanimate-coeff)
#   RESAMPLE_EVERY  : Default 5         (passed as --resample-every)
#   EVAL_ONLY       : Set to 1 to run in eval-only mode
#
# Example:
#   sbatch --job-name=train_dino_l12 \
#          --output=logs/sae_train_dino_l12_%j.out \
#          --export=IN_TRAIN_DIR=.../train_layer12/final,COCO_TRAIN_DIR=.../coco_train_layer12/final,VAL_DIR=.../coco_val_layer12/final,COMBINED_DIR=.../combined_train_layer12,CHECKPOINT_DIR=.../checkpoints/dinov3_l12_spatial,OUTPUT_DIR=.../models/dinov3_l12_spatial,D_MODEL=32000,K_FRACTION=0.0050,PER_INIT=0.1 \
#          run_train.sh

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/src
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Required vars ---
IN_TRAIN_DIR="${IN_TRAIN_DIR:?IN_TRAIN_DIR must be set}"
COCO_TRAIN_DIR="${COCO_TRAIN_DIR:?COCO_TRAIN_DIR must be set}"
VAL_DIR="${VAL_DIR:?VAL_DIR must be set}"
COMBINED_DIR="${COMBINED_DIR:?COMBINED_DIR must be set}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"

# --- Optional vars with defaults ---
MODEL_TYPE="${MODEL_TYPE:-SI-SAE}"
REANIMATE_COEFF="${REANIMATE_COEFF:-0.33}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-5}"
EVAL_ONLY="${EVAL_ONLY:-}"

# --- Build combined directory (symlinks, flock-protected) ---
LOCK_FILE="$(dirname "$COMBINED_DIR")/.$(basename "$COMBINED_DIR").lock"
mkdir -p "$COMBINED_DIR" "$CHECKPOINT_DIR" "$OUTPUT_DIR"

N_IN_SHARDS=$(ls "$IN_TRAIN_DIR"/shard_*.pt 2>/dev/null | wc -l)
N_COCO_SHARDS=$(ls "$COCO_TRAIN_DIR"/shard_*.pt 2>/dev/null | wc -l)

(
    flock -x 200

    EXPECTED_TOTAL=$((N_IN_SHARDS + N_COCO_SHARDS))
    CURRENT_TOTAL=$(ls "$COMBINED_DIR"/shard_*.pt 2>/dev/null | wc -l)

    if [ "$CURRENT_TOTAL" -ne "$EXPECTED_TOTAL" ]; then
        echo "Rebuilding combined dir ($CURRENT_TOTAL != $EXPECTED_TOTAL expected)"
        find "$COMBINED_DIR" -maxdepth 1 -type l -delete 2>/dev/null

        for f in "$IN_TRAIN_DIR"/shard_*.pt; do
            ln -sf "$f" "$COMBINED_DIR/$(basename "$f")"
        done

        IDX=$N_IN_SHARDS
        for f in "$COCO_TRAIN_DIR"/shard_*.pt; do
            ln -sf "$f" "$COMBINED_DIR/shard_$(printf '%04d' $IDX).pt"
            IDX=$((IDX + 1))
        done
    else
        echo "Combined dir already has $CURRENT_TOTAL shards, skipping rebuild"
    fi
) 200>"$LOCK_FILE"

TOTAL_SHARDS=$(ls "$COMBINED_DIR"/shard_*.pt 2>/dev/null | wc -l)

echo "============================================"
echo "SAE training: $MODEL_TYPE${EVAL_ONLY:+ (eval-only)}"
echo "  ImageNet train shards: $N_IN_SHARDS"
echo "  COCO train shards:     $N_COCO_SHARDS"
echo "  Total combined:        $TOTAL_SHARDS"
echo "  Val dir:               $VAL_DIR"
echo "  Checkpoint dir:        $CHECKPOINT_DIR"
echo "  Output dir:            $OUTPUT_DIR"
echo "  Extra args: $@"
echo "============================================"

# Build optional flag list
EVAL_ONLY_FLAG="${EVAL_ONLY:+--eval-only}"
D_MODEL_FLAG="${D_MODEL:+--d-model $D_MODEL}"
K_FRACTION_FLAG="${K_FRACTION:+--k-fraction $K_FRACTION}"
PER_INIT_FLAG="${PER_INIT:+--per-init $PER_INIT}"

python main.py "$COMBINED_DIR" "$MODEL_TYPE" \
    --checkpoint-dir   "$CHECKPOINT_DIR" \
    --output-dir       "$OUTPUT_DIR" \
    --checkpoint-every 5 \
    --val-dir          "$VAL_DIR" \
    --mixed-precision \
    --reanimate-coeff  "$REANIMATE_COEFF" \
    --resample-every   "$RESAMPLE_EVERY" \
    $EVAL_ONLY_FLAG \
    $D_MODEL_FLAG \
    $K_FRACTION_FLAG \
    $PER_INIT_FLAG \
    "$@"

echo "============================================"
echo "Done: $MODEL_TYPE"
echo "============================================"
