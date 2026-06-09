#!/bin/bash -l
#SBATCH --job-name=hf_upload_data_only
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p amdsmall
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/hf_upload_data_only_%j.out

# Upload ONLY the explorer_data .pt files (data + _heatmaps) to HF.
#
# Deliberately uploads NO JSON sidecars. Labels (feature_names / auto_interp,
# plus their _authors / _history files) live only on HF and are edited live in
# the Space — re-uploading stale local copies would clobber them. Feature
# indices are stable when the SAE weights are unchanged, so the existing HF
# labels keep matching freshly regenerated .pt files.

HF_DATA_REPO="Ramnie/sae-explorer-data"
HF_TOKEN_FILE="${HOME}/.hf_token"
PT_DIR="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"

set -e

source /projects/standard/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate base

if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: Token file not found: $HF_TOKEN_FILE"
    exit 1
fi
HF_TOKEN=$(cat "$HF_TOKEN_FILE")

echo "Uploading ONLY .pt data files to $HF_DATA_REPO (no JSON labels touched) ..."

python - <<PYEOF
import os, sys
from huggingface_hub import HfApi

api = HfApi(token="${HF_TOKEN}")
ddir = "${PT_DIR}/explorer_data"

# Explicit list: the 7 HF-registry data files (configs/models.yaml). For each we
# upload the data .pt and its _heatmaps.pt. No globbing (avoids sweeping stale
# union/ising .pt), no JSON (preserves human + Gemini labels on HF).
DATA_FILES = [
    "explorer_data_d32000_k160_val.pt",            # dinov3_l24_spatial
    "explorer_data_18.pt",                         # dinov3_l18_spatial
    "explorer_data_dino_layer12_d32000_k160_val.pt",  # dinov3_l12_spatial
    "explorer_data_cls_d20000_k80_val.pt",         # dinov3_l24_cls
    "explorer_data_clip_spatial_d32000_k160_val.pt",  # clip_l24_spatial
    "explorer_data_clip_layer16_d32000_k160_val.pt",  # clip_l16_spatial
    "explorer_data_clip_cls_d20000_k80_val.pt",    # clip_l24_cls
]

files, missing = [], []
for d in DATA_FILES:
    base = d[:-3]  # strip .pt
    for name in (d, base + "_heatmaps.pt"):
        p = os.path.join(ddir, name)
        (files if os.path.exists(p) else missing).append(p)

if missing:
    print("ERROR: expected files not found (did all 7 precompute jobs finish?):")
    for m in missing:
        print("  MISSING", m)
    sys.exit(1)

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
print("Done. No label JSONs were uploaded.")
PYEOF
