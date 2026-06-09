#!/bin/bash -l
#SBATCH --job-name=psych_simmetrics
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/psych_simmetrics_%j.out

# Coherent-unit psychophysics on the main DINOv3 dataset, at high resolution,
# under the paper's three similarity levels (low=color / mid=LPIPS / high=CLIP).
# See scripts/psychophysics_sim_metrics.py.

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae
python -c "import lpips" 2>/dev/null || pip install -q lpips

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/scripts
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASE_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
DATA_PATH="$BASE_DIR/explorer_data/explorer_data_d32000_k160_val.pt"
SAE_PATH="$BASE_DIR/models/dinov3_l24_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth"

if [ ! -f "$SAE_PATH" ]; then
    echo "Local SAE weights missing; pulling from HuggingFace..."
    SAE_PATH=$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id="Ramnie/sae-explorer-data", repo_type="dataset",
                      filename="sae_dinov3_l24_spatial_d32000_k160.pth"))
PY
)
fi

echo "============================================"
echo "Coherent-unit psychophysics x similarity metrics (high res)"
echo "  data: $DATA_PATH"
echo "============================================"

python psychophysics_sim_metrics.py \
    --data            "$DATA_PATH" \
    --sae-path        "$SAE_PATH" \
    --backbone        dinov3 \
    --token-type      spatial \
    --d-model         32000 \
    --top-k           160 \
    --image-dir       /scratch.global/lee02328/val \
    --extra-image-dir /scratch.global/lee02328/coco/val2017 \
    --pool-size       6000 \
    --ii-min          -0.42 \
    --n-features-sample 1500 \
    --n-trials        10 \
    --resize          224 \
    --lpips-net       alex \
    --clip-model      openai/clip-vit-large-patch14 \
    --out-dir         "$BASE_DIR/figures"

echo "============================================"
echo "Sim-metric psychophysics complete!"
echo "============================================"
