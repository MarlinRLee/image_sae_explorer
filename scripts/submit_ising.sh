#!/bin/bash -l
#SBATCH --job-name=ising
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/ising_%j.out

# Precompute Ising-coupling feature groups for the manifold explorer panel.
# Focused on the primary model: DINOv2 layer-11 spatial SAE (d=10000, k=100).
#
# Produces explorer_data_dinov2_layer11_d10000_k100_val_ising.pt next to the
# existing explorer sidecar, auto-discovered by the explorer's loader.

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

BASE=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE
cd "$BASE"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

EXPLORER_DATA="${EXPLORER_DATA:-$BASE/explorer_data/explorer_data_dinov2_layer11_d10000_k100_val.pt}"
SAE_PATH="${SAE_PATH:-$BASE/models/dinov2_l11_spatial/sae_1_SI-SAE_d10000_k100_per_init0.02_state_dict.pth}"
# Couplings are estimated from the FULL training set: the cached raw layer-11
# token shards (~18.9M tokens) streamed through the SAE encoder, no backbone.
# The backbone still runs over IMAGE_DIR only to collect the per-image manifold
# sample (which needs per-image identity the flat shards don't carry).
TOKEN_SHARDS="${TOKEN_SHARDS:-/scratch.global/lee02328/data/train/layer_11/shard_*.pt}"
IMAGE_DIR="${IMAGE_DIR:-/scratch.global/lee02328/val}"
EXTRA_IMAGE_DIR="${EXTRA_IMAGE_DIR:-}"

echo "============================================"
echo "Precomputing Ising couplings (full train set)"
echo "  Explorer data: $EXPLORER_DATA"
echo "  SAE:           $SAE_PATH"
echo "  Token shards:  $TOKEN_SHARDS"
echo "============================================"

python scripts/precompute_ising.py \
    --explorer-data   "$EXPLORER_DATA" \
    --sae-path        "$SAE_PATH" \
    --backbone        dinov2 \
    --layer           11 \
    --token-type      spatial \
    --token-shards    "$TOKEN_SHARDS" \
    --image-dir       "$IMAGE_DIR" \
    ${EXTRA_IMAGE_DIR:+--extra-image-dir "$EXTRA_IMAGE_DIR"} \
    --recursive \
    --min-frequency   200 \
    --max-features    4000 \
    --ridge           1e-2 \
    --sample-images   4000 \
    --batch-size      64 \
    --num-workers     8

echo "============================================"
echo "Ising precompute complete."
echo "============================================"
