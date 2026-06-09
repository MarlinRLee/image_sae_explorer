#!/bin/bash -l
#SBATCH --job-name=clip_embeddings
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/clip_embeddings_%j.out

# Add CLIP text-alignment scores to an existing explorer_data.pt file.
# Saves: clip_text_scores, clip_feature_embeds, clip_text_vocab
# A .bak copy of the original file is created automatically.
#
# Usage:
#   sbatch run_clip_embeddings.sh                    # enrich default .pt file
#   sbatch run_clip_embeddings.sh explorer_data.pt   # enrich specific file

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/scripts

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASE_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
DATA_PATH="${1:-$BASE_DIR/explorer_data_d32000_k160_val.pt}"

echo "============================================"
echo "Adding CLIP text alignment to explorer data"
echo "  Input:  $DATA_PATH"
echo "  (backup will be saved as $DATA_PATH.bak)"
echo "============================================"

python add_clip_embeddings.py \
    --data            "$DATA_PATH" \
    --image-dir       /scratch.global/lee02328/val \
    --extra-image-dir /scratch.global/lee02328/coco/val2017 \
    --extra-image-dir /scratch.global/lee02328/coco/train2017 \
    --n-top-images    4 \
    --batch-size      64

echo "============================================"
echo "CLIP embedding complete!"
echo "============================================"
