#!/bin/bash -l
#SBATCH --job-name=extract_layer
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/extract_layer_%j.out

# Extract vision backbone hidden states to float16 shards.
# Replaces: run_extract.sh, run_extract_cls.sh, run_extract_clip.sh,
#           run_extract_dino_layer.sh, run_extract_clip_layer.sh
#
# Usage: sbatch run_extract_layer.sh <split> [layer] [token_type] [backbone]
#   <split>      : in_train | in_val | coco_train | coco_val
#   [layer]      : hidden_states index (e.g. 12, 16, 18); empty = final layer
#   [token_type] : spatial (default) | cls | all | both
#   [backbone]   : dinov3 (default) | clip

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

pip install -q --ignore-installed --no-deps "huggingface-hub==1.4.1"

cd ~/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SPLIT="${1:?Usage: sbatch run_extract_layer.sh <split> [layer] [token_type] [backbone]}"
LAYER="${2:-}"
TOKEN_TYPE="${3:-spatial}"
BACKBONE="${4:-dinov3}"

# Validate
if [[ "$BACKBONE" != "dinov3" && "$BACKBONE" != "clip" ]]; then
    echo "Unknown backbone: $BACKBONE. Valid: dinov3 | clip"; exit 1
fi

# Layer flag (empty if not set)
LAYER_FLAG="${LAYER:+--layer $LAYER}"
LAYER_SUFFIX="${LAYER:+_layer${LAYER}}"

# Shard / batch size depends on token density
if [ "$TOKEN_TYPE" = "cls" ]; then
    BATCH_SIZE=128; IMAGES_PER_SHARD=100000
else
    BATCH_SIZE=64;  IMAGES_PER_SHARD=2816
fi

# Build output directory
case "$SPLIT" in
    in_train)   DATA_DIR="/scratch.global/lee02328/train";              FLAT_FLAG="" ;;
    in_val)     DATA_DIR="/scratch.global/lee02328/val";                FLAT_FLAG="" ;;
    coco_train) DATA_DIR="/scratch.global/lee02328/coco/train2017";     FLAT_FLAG="--flat" ;;
    coco_val)   DATA_DIR="/scratch.global/lee02328/coco/val2017";       FLAT_FLAG="--flat" ;;
    *) echo "Unknown split: $SPLIT. Valid: in_train | in_val | coco_train | coco_val"; exit 1 ;;
esac

SPLIT_NAME="${SPLIT/in_train/train}"    # in_train -> train
SPLIT_NAME="${SPLIT_NAME/in_val/val}"   # in_val   -> val

DATA_ROOT="/scratch.global/lee02328/data"
if [ "$BACKBONE" = "dinov3" ]; then
    # DINOv3: DINOV3/{split_name}{_layer?}{_cls?}/final
    TOKEN_SUFFIX="$( [ "$TOKEN_TYPE" = "cls" ] && echo "_cls" || echo "" )"
    OUTPUT_DIR="${DATA_ROOT}/DINOV3/${SPLIT_NAME}${LAYER_SUFFIX}${TOKEN_SUFFIX}/final"
else
    # CLIP: CLIP/{split_name}{_layer?}   (both-mode adds spatial/ and cls/ subdirs)
    OUTPUT_DIR="${DATA_ROOT}/CLIP/${SPLIT_NAME}${LAYER_SUFFIX}"
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Extracting backbone hidden states"
echo "  Backbone:   $BACKBONE"
echo "  Split:      $SPLIT"
echo "  Layer:      ${LAYER:-final}"
echo "  Token type: $TOKEN_TYPE"
echo "  Input:      $DATA_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Shard size: $IMAGES_PER_SHARD images"
echo "============================================"

python scripts/extract_activations.py \
    --backbone         "$BACKBONE" \
    --data_dir         "$DATA_DIR" \
    --output_dir       "$OUTPUT_DIR" \
    --token-type       "$TOKEN_TYPE" \
    --batch_size       $BATCH_SIZE \
    --images_per_shard $IMAGES_PER_SHARD \
    --num_workers      8 \
    $FLAT_FLAG \
    $LAYER_FLAG

echo "============================================"
echo "Done: $BACKBONE ${LAYER:-final} | $SPLIT ($TOKEN_TYPE)"
echo "============================================"
