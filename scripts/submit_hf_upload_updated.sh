#!/bin/bash -l
#SBATCH --job-name=hf_upload_updated
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p amdsmall
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/hf_upload_updated_%j.out

# Upload all explorer_data .pt files (including those updated by add_clip_embeddings.py).
# This covers the non-clip-prefixed files that were enriched in-place, plus all heatmaps.
#
# ##########################################################################
# # !! DANGER: THIS SCRIPT OVERWRITES HUMAN + GEMINI LABELS ON HF !!        #
# ##########################################################################
# # It globs and uploads local *_auto_interp.json and *_feature_names.json #
# # sidecars. Labels live ONLY on HF (edited live in the Space, with        #
# # _authors/_history). Local copies are stale/empty, so running this       #
# # REVERTS or WIPES the labels on Ramnie/sae-explorer-data.                #
# #                                                                         #
# # To upload ONLY data (.pt + heatmaps) and KEEP all labels, use:          #
# #     scripts/submit_hf_upload_data_only.sh                               #
# #                                                                         #
# # If you REALLY mean to overwrite the label JSONs, re-run with:           #
# #     sbatch --export=ALLOW_LABEL_OVERWRITE=1 submit_hf_upload_updated.sh #
# ##########################################################################

HF_DATA_REPO="Ramnie/sae-explorer-data"
HF_TOKEN_FILE="${HOME}/.hf_token"
PT_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"

set -e

if [ "${ALLOW_LABEL_OVERWRITE:-0}" != "1" ]; then
    echo "ABORTED: this script uploads label JSONs and would overwrite human +"
    echo "Gemini labels on $HF_DATA_REPO. For a label-safe data-only upload use"
    echo "scripts/submit_hf_upload_data_only.sh. To force, set ALLOW_LABEL_OVERWRITE=1."
    exit 1
fi

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate base

if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: Token file not found: $HF_TOKEN_FILE"
    exit 1
fi
HF_TOKEN=$(cat "$HF_TOKEN_FILE")

echo "Uploading all explorer data files to $HF_DATA_REPO ..."

python - <<PYEOF
import os, glob
from huggingface_hub import HfApi

api = HfApi(token="${HF_TOKEN}")

patterns = [
    "${PT_DIR}/explorer_data/explorer_data*.pt",
    "${PT_DIR}/explorer_data/explorer_data*_auto_interp.json",
    "${PT_DIR}/explorer_data/explorer_data*_feature_names.json",
]

files = []
for pat in patterns:
    files.extend(glob.glob(pat))
# Exclude .bak files
files = sorted(set(f for f in files if not f.endswith('.bak')))

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
print("Done.")
PYEOF
