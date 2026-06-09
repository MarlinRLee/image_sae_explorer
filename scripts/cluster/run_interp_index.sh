#!/bin/bash -l
#SBATCH --job-name=interp_index
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/interp_index_%j.out

# Add the LPIPS Interpretability Index (Klindt et al., 2023) to an existing
# explorer_data.pt file. Saves: interp_index, interp_index_m, interp_index_metric.
# A .bak copy of the original file is created automatically.
#
# Usage:
#   sbatch run_interp_index.sh                    # enrich default .pt file
#   sbatch run_interp_index.sh explorer_data.pt   # enrich specific file

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae
python -c "import lpips" 2>/dev/null || pip install -q lpips

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/scripts

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASE_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
DATA_PATH="${1:-$BASE_DIR/explorer_data_d32000_k160_val.pt}"

echo "============================================"
echo "Adding LPIPS Interpretability Index to explorer data"
echo "  Input:  $DATA_PATH"
echo "  (backup will be saved as $DATA_PATH.bak)"
echo "============================================"

python add_interpretability_index.py \
    --data            "$DATA_PATH" \
    --image-dir       /scratch.global/lee02328/val \
    --extra-image-dir /scratch.global/lee02328/coco/val2017 \
    --extra-image-dir /scratch.global/lee02328/coco/train2017 \
    --interp-m        5 \
    --lpips-net       alex \
    --resize          64 \
    --batch-size      256

echo "============================================"
echo "Interpretability Index complete!"
echo "============================================"
