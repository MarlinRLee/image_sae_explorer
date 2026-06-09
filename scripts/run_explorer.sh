#!/bin/bash -l
#SBATCH --job-name=sae_explorer
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p amdsmall
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/explorer_%j.out

# Launch the interactive SAE Feature Explorer (Bokeh server).
#
# Usage (sbatch):
#   sbatch run_explorer.sh                          # defaults
#   sbatch run_explorer.sh /path/to/explorer_data.pt  # custom data
#   sbatch run_explorer.sh /path/to/data.pt 5007    # custom data + port
#
# Usage (interactive srun):
#   srun --partition=interactive --cpus-per-task=4 --mem=32g --time=2:00:00 --pty bash
#   bash run_explorer.sh [data_path] [port]
#
# After the job starts, check the output file for the node name, then:
#   ssh -L <port>:<node>:<port> lee02328@login.msi.umn.edu
#   Open: http://localhost:<port>/explorer_app

export TMPDIR=/tmp

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate imagenet_sae

# --- Configurable paths ---
DATA_PATH="${1:-/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/explorer_data/explorer_data_d32000_k160_val.pt}"
PORT="${2:-5006}"
IMAGE_DIR="/scratch.global/lee02328/val"
EXTRA_IMAGE_DIR="/scratch.global/lee02328/coco/val2017"

# --- All models shown in the "Active model" dropdown ---
# Each entry in COMPARE_DATA needs matching COMPARE_LABELS and COMPARE_SAE_PATHS entries.
BASE_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
COMPARE_DATA=()
COMPARE_LABELS=()
COMPARE_SAE_PATHS=()

_add_model() {
    local data_file="$1" label="$2" sae_file="$3"
    local data_path="$BASE_DIR/$data_file"
    if [ -f "$data_path" ]; then
        COMPARE_DATA+=("$data_path")
        COMPARE_LABELS+=("$label")
        local sae_path="$BASE_DIR/$sae_file"
        [ -f "$sae_path" ] && COMPARE_SAE_PATHS+=("$sae_path") || COMPARE_SAE_PATHS+=("")
    fi
}

_add_model "explorer_data/explorer_data_18.pt"                              "DINOv3 L18 Spatial (d=20K)"  "models/dinov3_l18_spatial/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth"
_add_model "explorer_data/explorer_data_cls_d20000_k80_val.pt"              "DINOv3 L24 CLS (d=20K)"      "models/dinov3_l24_cls/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth"
_add_model "explorer_data/explorer_data_clip_spatial_d32000_k160_val.pt"    "CLIP L24 Spatial (d=32K)"    "models/clip_l24_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth"
_add_model "explorer_data/explorer_data_clip_cls_d20000_k80_val.pt"         "CLIP L24 CLS (d=20K)"        "models/clip_l24_cls/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth"
_add_model "explorer_data/explorer_data_dino_layer12_d32000_k160_val.pt"    "DINOv3 L12 Spatial (d=32K)"  "models/dinov3_l12_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth"
_add_model "explorer_data/explorer_data_clip_layer16_d32000_k160_val.pt"    "CLIP L16 Spatial (d=32K)"    "models/clip_l16_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth"
_add_model "explorer_data/explorer_data_dinov2_layer11_d10000_k100_val.pt"  "DINOv2 L11 Spatial (d=10K)"  "models/dinov2_l11_spatial/sae_1_SI-SAE_d10000_k100_per_init0.02_state_dict.pth"
_add_model "explorer_data/explorer_data_dinov2_layer08_d10000_k50_val.pt"   "DINOv2 L8 Spatial (d=10K)"   "models/dinov2_l08_spatial/sae_1_SI-SAE_d10000_k50_per_init0.1_state_dict.pth"
_add_model "explorer_data/explorer_data_dinov2_l11_patchnorm_d10000_k100_val.pt" "DINOv2 L11 PatchNorm (d=10K)" "models/dinov2_l11_patchnorm/sae_1_SAE_d10000_k100_state_dict.pth"

# Primary SAE weights
PRIMARY_SAE="$BASE_DIR/models/dinov3_l24_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth"

# Build flags
COMPARE_ARGS=()
if [ ${#COMPARE_DATA[@]} -gt 0 ]; then
    COMPARE_ARGS+=(--compare-data      "${COMPARE_DATA[@]}")
    COMPARE_ARGS+=(--compare-labels    "${COMPARE_LABELS[@]}")
    COMPARE_ARGS+=(--compare-sae-paths "${COMPARE_SAE_PATHS[@]}")
fi

SAE_ARGS=()
[ -f "$PRIMARY_SAE" ] && SAE_ARGS+=(--sae-path "$PRIMARY_SAE")

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    echo "Run precompute first: sbatch run_precompute.sh"
    exit 1
fi

NODE=$(hostname)
echo "============================================"
echo "SAE Feature Explorer"
echo "  Node:       $NODE"
echo "  Port:       $PORT"
echo "  Data:       $DATA_PATH"
echo "  Images:     $IMAGE_DIR"
echo "  Extra imgs: $EXTRA_IMAGE_DIR"
echo "  Comparisons: ${#COMPARE_DATA[@]} dataset(s)"
for i in "${!COMPARE_DATA[@]}"; do
    echo "    [$i] ${COMPARE_LABELS[$i]}: ${COMPARE_DATA[$i]}"
done
echo ""
echo "To connect, run on your local machine:"
echo "  ssh -L ${PORT}:${NODE}:${PORT} lee02328@login.msi.umn.edu"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:${PORT}/explorer_app"
echo "============================================"

bokeh serve "$BASE_DIR/scripts/explorer_app.py" \
    --port "$PORT" \
    --allow-websocket-origin="*" \
    --session-token-expiration 86400 \
    --args \
      --data "$DATA_PATH" \
      --image-dir "$IMAGE_DIR" \
      --extra-image-dir "$EXTRA_IMAGE_DIR" \
      --primary-label "DINOv3 L24 Spatial (d=32K)" \
      "${COMPARE_ARGS[@]}" \
      "${SAE_ARGS[@]}"
