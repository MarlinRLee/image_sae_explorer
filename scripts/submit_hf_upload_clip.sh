#!/bin/bash -l
#SBATCH --job-name=hf_upload_clip
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p amdsmall
#SBATCH --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/hf_upload_clip_%j.out

# Upload only the new CLIP explorer_data files to Hugging Face Hub.

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

echo "Uploading CLIP explorer data files to $HF_DATA_REPO ..."

python - <<PYEOF
import os, glob
from huggingface_hub import HfApi

api = HfApi(token="${HF_TOKEN}")

files = sorted(glob.glob("${PT_DIR}/explorer_data_clip*.pt"))

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
