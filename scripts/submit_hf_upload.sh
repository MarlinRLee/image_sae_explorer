#!/bin/bash -l
#SBATCH --job-name=hf_upload
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p amdsmall
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/hf_upload_%j.out

# Upload explorer data and thumbnails to Hugging Face Hub.
#
# ##########################################################################
# # !! DANGER: THIS SCRIPT OVERWRITES HUMAN + GEMINI LABELS ON HF !!        #
# ##########################################################################
# # Step [1/3] globs and uploads local *_auto_interp.json and              #
# # *_feature_names.json sidecars. Labels live ONLY on HF (edited live in   #
# # the Space, with _authors/_history). Local copies are stale/empty, so    #
# # running this REVERTS or WIPES the labels on Ramnie/sae-explorer-data.   #
# # It also re-uploads SAE weights + the thumbnails tarball.                #
# #                                                                         #
# # For a routine refresh of regenerated data only (.pt + heatmaps) that    #
# # KEEPS all labels, use:  scripts/submit_hf_upload_data_only.sh           #
# #                                                                         #
# # If you REALLY mean a full upload incl. label JSONs, re-run with:        #
# #     sbatch --export=ALLOW_LABEL_OVERWRITE=1 submit_hf_upload.sh         #
# ##########################################################################
#
# Prerequisites:
#   1. Create two private HF Dataset repos on huggingface.co:
#        YOUR_USERNAME/sae-explorer-data    (for .pt files)
#        YOUR_USERNAME/sae-explorer-images  (for thumbnails)
#   2. Generate a write-access token at huggingface.co/settings/tokens
#   3. Save the token to ~/.hf_token  (chmod 600 ~/.hf_token)
#   4. Set HF_DATA_REPO and HF_IMAGES_REPO below, then:
#        sbatch submit_hf_upload.sh

# ── Configure these ───────────────────────────────────────────────────────────
HF_DATA_REPO="Ramnie/sae-explorer-data"
HF_IMAGES_REPO="Ramnie/sae-explorer-images"
HF_TOKEN_FILE="${HOME}/.hf_token"

PT_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
IMAGES_DIR="/users/9/lee02328/hf_images"
TAR_PATH="/users/9/lee02328/hf_images.tar.gz"
# ─────────────────────────────────────────────────────────────────────────────

set -e

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate base

if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: Token file not found: $HF_TOKEN_FILE"
    echo "Save your HF write token there: echo 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token"
    exit 1
fi
HF_TOKEN=$(cat "$HF_TOKEN_FILE")

if [ "${ALLOW_LABEL_OVERWRITE:-0}" != "1" ]; then
    echo "ABORTED: this script uploads label JSONs (and SAE weights + thumbnails)"
    echo "and would overwrite human + Gemini labels on $HF_DATA_REPO. For a"
    echo "label-safe data-only upload use scripts/submit_hf_upload_data_only.sh."
    echo "To force a full upload, set ALLOW_LABEL_OVERWRITE=1."
    exit 1
fi

echo "============================================"
echo "Hugging Face Upload"
echo "  Images repo: $HF_IMAGES_REPO"
echo "  Images dir:  $IMAGES_DIR"
echo "============================================"

# ── Upload .pt / JSON data files ─────────────────────────────────────────────
echo ""
echo "[1/3] Uploading explorer data files..."
python - <<PYEOF
import os, glob
from huggingface_hub import HfApi

api = HfApi(token="${HF_TOKEN}")

patterns = [
    "${PT_DIR}/explorer_data/explorer_data*.pt",
    "${PT_DIR}/explorer_data/explorer_data*_auto_interp.json",
    "${PT_DIR}/explorer_data/explorer_data*_feature_names.json",
    "${PT_DIR}/explorer_data/explorer_data*_heatmaps.pt",
]

files = []
for pat in patterns:
    files.extend(glob.glob(pat))
files = sorted(set(files))

for fpath in files:
    size_mb = os.path.getsize(fpath) / 1e6
    fname = os.path.basename(fpath)
    print(f"  Uploading {fname} ({size_mb:.0f} MB)...", flush=True)
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=fname,
        repo_id="${HF_DATA_REPO}",
        repo_type="dataset",
        commit_message=f"Upload {fname}",
    )
print("  Data upload complete.")
PYEOF

# ── Upload SAE weights with unique HF filenames ───────────────────────────────
echo ""
echo "[2/4] Uploading SAE weights..."
python - <<PYEOF
import os
from huggingface_hub import HfApi

api = HfApi(token="${HF_TOKEN}")
base = "${PT_DIR}"

# Maps (local relative path) -> (HF filename)
SAE_MAP = {
    "models/dinov3_l24_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth":    "sae_dinov3_l24_spatial_d32000_k160.pth",
    "models/dinov3_l18_spatial/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth":     "sae_dinov3_l18_spatial_d20000_k80.pth",
    "models/dinov3_l24_cls/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth":         "sae_dinov3_l24_cls_d20000_k80.pth",
    "models/clip_l24_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth":      "sae_clip_l24_spatial_d32000_k160.pth",
    "models/clip_l24_cls/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth":           "sae_clip_l24_cls_d20000_k80.pth",
    "models/dinov3_l12_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth":    "sae_dinov3_l12_spatial_d32000_k160.pth",
    "models/clip_l16_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth":      "sae_clip_l16_spatial_d32000_k160.pth",
    "models/dinov2_l11_spatial/sae_1_SI-SAE_d10000_k100_per_init0.02_state_dict.pth":   "sae_dinov2_l11_spatial_d10000_k100.pth",
    "models/dinov2_l08_spatial/sae_1_SI-SAE_d10000_k50_per_init0.1_state_dict.pth":     "sae_dinov2_l08_spatial_d10000_k50.pth",
}

for local_rel, hf_name in SAE_MAP.items():
    local_path = os.path.join(base, local_rel)
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {local_rel}")
        continue
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  Uploading {hf_name} ({size_mb:.0f} MB)...", flush=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hf_name,
        repo_id="${HF_DATA_REPO}",
        repo_type="dataset",
        commit_message=f"Upload SAE weights {hf_name}",
    )
print("  SAE weights upload complete.")
PYEOF

# ── Tar thumbnails into a single archive ─────────────────────────────────────
echo ""
echo "[3/4] Creating tar archive of thumbnails..."

tar -czf "$TAR_PATH" -C "$(dirname $IMAGES_DIR)" "$(basename $IMAGES_DIR)"
du -sh "$TAR_PATH"

# ── Upload single tar file ────────────────────────────────────────────────────
echo ""
echo "[4/4] Uploading tar archive..."
python - <<PYEOF
import os
from huggingface_hub import HfApi

api = HfApi(token="${HF_TOKEN}")
tar_path = "${TAR_PATH}"
size_mb = os.path.getsize(tar_path) / 1e6
print(f"  Uploading hf_images.tar.gz ({size_mb:.0f} MB)...", flush=True)
api.upload_file(
    path_or_fileobj=tar_path,
    path_in_repo="hf_images.tar.gz",
    repo_id="${HF_IMAGES_REPO}",
    repo_type="dataset",
    commit_message="Add explorer thumbnails (tar archive)",
)
print("  Done.")
PYEOF

echo ""
echo "============================================"
echo "Upload complete."
echo "  Set these Secrets in your HF Space:"
echo "    HF_DATASET_REPO = $HF_DATA_REPO"
echo "    HF_IMAGES_REPO  = $HF_IMAGES_REPO"
echo "    HF_TOKEN        = (a write-access token)"
echo "============================================"
