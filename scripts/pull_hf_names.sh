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

set -euo pipefail

export HF_DATA_REPO="${HF_DATA_REPO:-Ramnie/sae-explorer-data}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.hf_token}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            DEST_DIR="${2:?--dest requires a directory argument}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--dest <dir>]" >&2
            exit 1
            ;;
    esac
done
export DEST_DIR

if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: Token file not found: $HF_TOKEN_FILE"
    echo "Save your HF write token: echo 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token"
    exit 1
fi
HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
export HF_TOKEN

echo "Pulling feature name JSONs from $HF_DATA_REPO -> $DEST_DIR ..."

python - <<'PYEOF'
import os
from huggingface_hub import HfApi, hf_hub_download

repo  = os.environ["HF_DATA_REPO"]
dest  = os.environ["DEST_DIR"]
token = os.environ["HF_TOKEN"]

api = HfApi(token=token)
repo_files = api.list_repo_files(repo_id=repo, repo_type="dataset")
name_files = [f for f in repo_files if f.endswith("_feature_names.json")]

if not name_files:
    print("  No feature name files found in the dataset repo.")
else:
    for fname in name_files:
        print(f"  Downloading {fname}...", flush=True)
        hf_hub_download(
            repo_id=repo,
            filename=fname,
            repo_type="dataset",
            local_dir=dest,
            token=token,
        )
        print(f"    -> {os.path.join(dest, fname)}")
    print(f"Done. Pulled {len(name_files)} file(s).")
PYEOF
