#!/bin/bash -l
#SBATCH --job-name=auto_interp
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p amdsmall
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/auto_interp_%j.out

# Batch auto-interp labeling via Gemini API.  No GPU required.
#
# Configure via env vars (pass with sbatch --export, or edit defaults below):
#
#   DATA_PATH   : path to explorer_data.pt          (required)
#   START_FEAT  : first feature index               (default: 0)
#   END_FEAT    : last feature index, exclusive     (default: all)
#   N_IMAGES    : MEIs sent to Gemini per feature   (default: 6)
#   MODEL       : Gemini model name                 (default: gemini-2.0-flash)
#   SLEEP       : seconds between API calls         (default: 0.5)
#
# Examples:
#
#   # Label all features of one model:
#   sbatch run_auto_interp.sh
#
#   # Label features 0-4000 of a specific model, restartable:
#   sbatch --export=DATA_PATH=../explorer_data_clip_layer16_d32000_k160_val.pt,END_FEAT=4000 \
#          run_auto_interp.sh
#
#   # Resume where you left off (skips already-labeled features):
#   sbatch --export=DATA_PATH=../explorer_data_d32000_k160_val.pt,START_FEAT=4000 \
#          run_auto_interp.sh

ulimit -n 65536
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

# --- Conda ---
source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

cd /users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/scripts

# --- Gemini API key ---
# Store your key in ~/.api_keys (chmod 600 ~/.api_keys) as a single line:
#     export GOOGLE_API_KEY="AIza..."
# That file is never committed to git.
if [[ -f "$HOME/.api_keys" ]]; then
    source "$HOME/.api_keys"
fi

if [[ -z "$GOOGLE_API_KEY" ]]; then
    echo "ERROR: GOOGLE_API_KEY is not set."
    echo "  Add 'export GOOGLE_API_KEY=\"AIza...\"' to ~/.api_keys"
    exit 1
fi

# --- Job parameters ---
BASE_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
DATA_PATH="${DATA_PATH:-$BASE_DIR/explorer_data_d32000_k160_val.pt}"
START_FEAT="${START_FEAT:-0}"
N_IMAGES="${N_IMAGES:-6}"
MODEL="${MODEL:-gemini-2.5-flash}"
SLEEP="${SLEEP:-0.1}"
IMAGE_DIR="${IMAGE_DIR:-/scratch.global/lee02328/val}"
EXTRA_IMAGE_DIR="${EXTRA_IMAGE_DIR:-/scratch.global/lee02328/coco/val2017}"

# Build optional --end-feat flag only if END_FEAT is set
END_FEAT_FLAG="${END_FEAT:+--end-feat $END_FEAT}"

OUTPUT_JSON="${DATA_PATH%.pt}_auto_interp.json"

echo "============================================"
echo "Auto-interp via Gemini API"
echo "  Data:        $DATA_PATH"
echo "  Output:      $OUTPUT_JSON"
echo "  Model:       $MODEL"
echo "  Features:    $START_FEAT – ${END_FEAT:-end}"
echo "  Images/feat: $N_IMAGES"
echo "  Sleep (s):   $SLEEP"
echo "============================================"

python -u auto_interp_vlm.py \
    --data         "$DATA_PATH" \
    --image-dir    "$IMAGE_DIR" \
    --extra-image-dir "$EXTRA_IMAGE_DIR" \
    --model        "$MODEL" \
    --n-images     "$N_IMAGES" \
    --start-feat   "$START_FEAT" \
    --sleep        "$SLEEP" \
    --skip-labeled \
    $END_FEAT_FLAG

echo "============================================"
echo "Done! Output: $OUTPUT_JSON"
echo "  $(wc -l < "$OUTPUT_JSON" 2>/dev/null || echo '?') lines in JSON"
echo "============================================"
