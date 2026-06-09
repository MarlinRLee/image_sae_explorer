#!/bin/bash -l
#SBATCH --job-name=ii_by_agg
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/ii_by_agg_%j.out

# Compute the LPIPS Interpretability Index under each MEI aggregation
# (max / mean / crop) and plot the three distributions overlaid. Uses the
# stored *_img_idx rankings — no backbone/SAE rerun needed. See
# scripts/compute_ii_by_aggregation.py.

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae
python -c "import lpips" 2>/dev/null || pip install -q lpips

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/scripts
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASE_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
DATA_PATH="${1:-$BASE_DIR/explorer_data/explorer_data_d32000_k160_val.pt}"

echo "============================================"
echo "Interpretability Index by MEI aggregation"
echo "  data: $DATA_PATH"
echo "============================================"

python compute_ii_by_aggregation.py \
    --data            "$DATA_PATH" \
    --interp-m        5 \
    --lpips-net       alex \
    --resize          64 \
    --batch-size      256 \
    --image-dir       /scratch.global/lee02328/val \
    --extra-image-dir /scratch.global/lee02328/coco/val2017 \
    --extra-image-dir /scratch.global/lee02328/coco/train2017 \
    --out-dir         "$BASE_DIR/figures"

echo "============================================"
echo "II-by-aggregation complete!"
echo "============================================"
