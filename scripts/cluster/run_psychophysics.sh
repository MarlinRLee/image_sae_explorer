#!/bin/bash -l
#SBATCH --job-name=psychophysics
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/psychophysics_%j.out

# In-silico psychophysics validation of the Interpretability Index + 2AFC
# diversity, broken out by MEI aggregation (max / mean / crop-mean).
# Full faithful sweep: re-runs DINOv3 + SAE over an image pool. See
# scripts/psychophysics_eval.py.
#
# Prereq: the primary explorer_data must already carry 'interp_index'
# (run run_interp_index.sh on it first). Submit chained, e.g.:
#   IMG=10531421   # the image-download job id
#   J1=$(sbatch --parsable --dependency=afterok:$IMG run_interp_index.sh \
#          $BASE/explorer_data/explorer_data_d32000_k160_val.pt)
#   sbatch --dependency=afterok:$J1 run_psychophysics.sh

ulimit -n 65536
export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae
python -c "import lpips" 2>/dev/null || pip install -q lpips

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/scripts
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASE_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
DATA_PATH="${1:-$BASE_DIR/explorer_data/explorer_data_d32000_k160_val.pt}"
SAE_PATH="$BASE_DIR/models/dinov3_l24_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth"

# Fall back to the HuggingFace data repo if the local weights are absent.
if [ ! -f "$SAE_PATH" ]; then
    echo "Local SAE weights missing; pulling from HuggingFace..."
    SAE_PATH=$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id="Ramnie/sae-explorer-data", repo_type="dataset",
                      filename="sae_dinov3_l24_spatial_d32000_k160.pth"))
PY
)
fi

# Diversity feature subset. Default: the top-800 features by II. Override by
# exporting II_MIN to instead take ALL features with II >= that value -- a lower
# cutoff = broader subset = more A100 time (e.g. II_MIN=-0.45 -> ~7.5k features).
# DIV_PARTNERS sets random partners per feature.
DIV_PARTNERS="${DIV_PARTNERS:-200}"
if [ -n "$II_MIN" ]; then
    II_SELECT="--ii-min $II_MIN"
else
    II_SELECT="--ii-top-n 800"
fi

echo "============================================"
echo "In-silico psychophysics + diversity"
echo "  data: $DATA_PATH"
echo "  sae:  $SAE_PATH"
echo "  diversity subset: $II_SELECT, $DIV_PARTNERS partners"
echo "============================================"

python psychophysics_eval.py \
    --data            "$DATA_PATH" \
    --sae-path        "$SAE_PATH" \
    --backbone        dinov3 \
    --token-type      spatial \
    --d-model         32000 \
    --top-k           160 \
    --image-dir       /scratch.global/lee02328/val \
    --extra-image-dir /scratch.global/lee02328/coco/val2017 \
    --pool-size       6000 \
    --n-features-sample 3000 \
    --n-trials        10 \
    $II_SELECT \
    --diversity-partners $DIV_PARTNERS \
    --lpips-net       alex \
    --resize          64 \
    --out-dir         "$BASE_DIR/figures"

echo "============================================"
echo "Psychophysics evaluation complete!"
echo "============================================"
