#!/bin/bash
# Pull feature name JSON files from HF dataset repo back to local directory.
#
# Run this to sync labels that were added via the HF Space explorer back to
# your local explorer_data directory.
#
# Usage:
#   scripts/pull_hf_names.sh [--dest <dir>]
#
# Default dest: same directory as this script's parent (project root)

set -e

HF_DATA_REPO="Ramnie/sae-explorer-data"
HF_TOKEN_FILE="${HOME}/.hf_token"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)"

for arg in "$@"; do
    case "$arg" in
        --dest) shift; DEST_DIR="$1" ;;
    esac
done

if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: Token file not found: $HF_TOKEN_FILE"
    echo "Save your HF write token: echo 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token"
    exit 1
fi
HF_TOKEN=$(cat "$HF_TOKEN_FILE")

echo "Pulling feature name JSONs from $HF_DATA_REPO -> $DEST_DIR ..."

python - <<PYEOF
import os
from huggingface_hub import HfApi

api = HfApi(token="${HF_TOKEN}")
repo_files = api.list_repo_files(repo_id="${HF_DATA_REPO}", repo_type="dataset")
name_files = [f for f in repo_files if f.endswith("_feature_names.json")]

if not name_files:
    print("  No feature name files found in the dataset repo.")
else:
    from huggingface_hub import hf_hub_download
    for fname in name_files:
        local_path = os.path.join("${DEST_DIR}", fname)
        print(f"  Downloading {fname}...", flush=True)
        hf_hub_download(
            repo_id="${HF_DATA_REPO}",
            filename=fname,
            repo_type="dataset",
            local_dir="${DEST_DIR}",
            local_dir_use_symlinks=False,
            token="${HF_TOKEN}",
        )
        print(f"    -> {local_path}")
    print(f"Done. Pulled {len(name_files)} file(s).")
PYEOF
