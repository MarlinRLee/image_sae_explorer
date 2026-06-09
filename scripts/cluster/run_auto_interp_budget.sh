#!/bin/bash -l
#SBATCH --job-name=auto_interp_budget
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p amdsmall
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/auto_interp_budget_%j.out

# Budget-limited Gemini auto-interp for the PRIMARY registry model, synced
# with Hugging Face. No GPU required (API + CPU image encoding only).
#
# Flow: pull current labels from HF -> label only features missing BOTH a
# manual and an auto-interp label (most-frequent first) -> stop at ~$BUDGET
# of measured Gemini usage -> push the updated auto-interp JSONs back to HF.
#
# Configure via env (sbatch --export=...) or edit defaults below:
#   BUDGET    : USD to spend before stopping        (default: 10)
#   DATA_PATH : local .pt for the primary model     (default: d32000_k160_val)
#   MODEL     : Gemini model name                   (default: gemini-2.5-flash)
#   N_IMAGES  : MEIs sent per feature               (default: 6)
#   DRY_RUN   : set to 1 to count candidates only   (default: unset)
#
# Examples:
#   sbatch run_auto_interp_budget.sh
#   sbatch --export=BUDGET=5,DRY_RUN=1 run_auto_interp_budget.sh

ulimit -n 65536
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

# --- Conda ---
source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/scripts

# --- Secrets ---
# GOOGLE_API_KEY in ~/.api_keys ; HF write token in ~/.hf_token (chmod 600).
if [[ -f "$HOME/.api_keys" ]]; then source "$HOME/.api_keys"; fi
if [[ -z "$HF_TOKEN" && -f "$HOME/.hf_token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
    export HF_TOKEN
fi

if [[ -z "$GOOGLE_API_KEY" ]]; then
    echo "ERROR: GOOGLE_API_KEY is not set (add it to ~/.api_keys)."; exit 1
fi
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN is not set (save a write token to ~/.hf_token)."; exit 1
fi

# --- Parameters ---
BASE_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
BUDGET="${BUDGET:-10}"
DATA_PATH="${DATA_PATH:-$BASE_DIR/explorer_data/explorer_data_d32000_k160_val.pt}"
MODEL="${MODEL:-gemini-2.5-flash}"
N_IMAGES="${N_IMAGES:-6}"
IMAGE_DIR="${IMAGE_DIR:-/scratch.global/lee02328/val}"
EXTRA_IMAGE_DIR="${EXTRA_IMAGE_DIR:-/scratch.global/lee02328/coco/val2017}"
DRY_RUN_FLAG="${DRY_RUN:+--dry-run}"

echo "============================================"
echo "Budget auto-interp via Gemini (primary model)"
echo "  Budget:   \$$BUDGET"
echo "  Data:     $DATA_PATH"
echo "  Model:    $MODEL"
echo "============================================"

python -u auto_interp_hf.py \
    --registry "$BASE_DIR/configs/models.yaml" \
    --data "$DATA_PATH" \
    --image-dir "$IMAGE_DIR" \
    --extra-image-dir "$EXTRA_IMAGE_DIR" \
    --budget "$BUDGET" \
    --model "$MODEL" \
    --n-images "$N_IMAGES" \
    $DRY_RUN_FLAG

echo "============================================"
echo "Done."
echo "============================================"
