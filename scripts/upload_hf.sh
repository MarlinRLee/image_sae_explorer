#!/bin/bash
# Upload explorer data to the Hugging Face dataset repos.
#
# Usage:
#   bash scripts/upload_hf.sh                                   # data-only (default, label-safe)
#   ALLOW_LABEL_OVERWRITE=1 bash scripts/upload_hf.sh full      # full upload
#
# Modes
#   data-only (default)
#       Uploads each registry model's .pt and _heatmaps.pt — nothing else.
#       Deliberately uploads NO JSON sidecars: labels (feature_names /
#       auto_interp, plus their _authors/_history files) live only on HF and
#       are edited live in the Space, so re-uploading stale local copies would
#       clobber them. Feature indices are stable while the SAE weights are
#       unchanged, so existing HF labels keep matching regenerated .pt files.
#   full
#       Additionally uploads the label JSON sidecars, the SAE weight files in
#       SAE_MAP below, and the thumbnails tarball.
#       !! DANGER: overwrites human + Gemini labels on HF_DATA_REPO !!
#       Requires ALLOW_LABEL_OVERWRITE=1 as a deliberate second step.
#
# The data-file list is read from configs/models.yaml — the registry is the
# single source of truth; do not duplicate filenames here.
#
# Prerequisites:
#   1. Create two private HF Dataset repos on huggingface.co:
#        YOUR_USERNAME/sae-explorer-data    (for .pt files + SAE weights)
#        YOUR_USERNAME/sae-explorer-images  (for the thumbnails tarball)
#   2. Generate a write-access token at huggingface.co/settings/tokens
#   3. Save the token to ~/.hf_token  (chmod 600 ~/.hf_token)
#   4. Override the variables below via the environment if your layout differs.

set -euo pipefail

MODE="${1:-data-only}"
if [[ "$MODE" != "data-only" && "$MODE" != "full" ]]; then
    echo "ERROR: unknown mode '$MODE' (expected 'data-only' or 'full')" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Configure these (env vars override) ──────────────────────────────────────
export HF_DATA_REPO="${HF_DATA_REPO:-Ramnie/sae-explorer-data}"
export HF_IMAGES_REPO="${HF_IMAGES_REPO:-Ramnie/sae-explorer-images}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.hf_token}"
export PT_DIR="${PT_DIR:-${REPO_ROOT}}"           # dir containing explorer_data/
export IMAGES_DIR="${IMAGES_DIR:-${HOME}/hf_images}"
export TAR_PATH="${TAR_PATH:-${HOME}/hf_images.tar.gz}"
export REGISTRY="${REGISTRY:-${REPO_ROOT}/configs/models.yaml}"
# ─────────────────────────────────────────────────────────────────────────────

if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: Token file not found: $HF_TOKEN_FILE"
    echo "Save your HF write token there: echo 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token"
    exit 1
fi
HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
export HF_TOKEN

if [[ "$MODE" == "full" && "${ALLOW_LABEL_OVERWRITE:-0}" != "1" ]]; then
    echo "ABORTED: 'full' mode uploads label JSONs (plus SAE weights + thumbnails)"
    echo "and would overwrite human + Gemini labels on $HF_DATA_REPO. The default"
    echo "data-only mode is label-safe. To force a full upload, set"
    echo "ALLOW_LABEL_OVERWRITE=1."
    exit 1
fi

export UPLOAD_MODE="$MODE"

echo "============================================"
echo "Hugging Face upload — mode: $MODE"
echo "  Data repo:   $HF_DATA_REPO"
echo "  Registry:    $REGISTRY"
echo "  Data dir:    $PT_DIR/explorer_data"
echo "============================================"

# ── Upload .pt data files (+ label JSONs in full mode) ───────────────────────
echo ""
echo "[1/3] Uploading explorer data files..."
python - <<'PYEOF'
import glob, os, sys
import yaml
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
ddir = os.path.join(os.environ["PT_DIR"], "explorer_data")
full = os.environ["UPLOAD_MODE"] == "full"

with open(os.environ["REGISTRY"]) as fh:
    registry = yaml.safe_load(fh)

files, missing = [], []
for model in registry["models"]:
    base = model["data_file"][:-3]  # strip .pt
    for name in (model["data_file"], base + "_heatmaps.pt"):
        p = os.path.join(ddir, name)
        (files if os.path.exists(p) else missing).append(p)

if missing:
    print("ERROR: registry data files not found (did all precompute jobs finish?):")
    for m in missing:
        print("  MISSING", m)
    sys.exit(1)

if full:
    for pat in ("explorer_data*_auto_interp.json", "explorer_data*_feature_names.json"):
        files.extend(glob.glob(os.path.join(ddir, pat)))

for fpath in sorted(set(files)):
    size_mb = os.path.getsize(fpath) / 1e6
    fname = os.path.basename(fpath)
    print(f"  Uploading {fname} ({size_mb:.0f} MB)...", flush=True)
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=fname,
        repo_id=os.environ["HF_DATA_REPO"],
        repo_type="dataset",
        commit_message=f"Upload {fname}",
    )
print("  Data upload complete." + ("" if full else " No label JSONs were uploaded."))
PYEOF

if [[ "$MODE" == "data-only" ]]; then
    echo ""
    echo "Done (data-only). SAE weights, label JSONs and thumbnails untouched."
    exit 0
fi

# ── Upload SAE weights with unique HF filenames ───────────────────────────────
echo ""
echo "[2/3] Uploading SAE weights..."
python - <<'PYEOF'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
base = os.environ["PT_DIR"]

# Maps (local relative path) -> (HF filename; must match `sae_file` in the registry)
SAE_MAP = {
    "models/dinov3_l24_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth":    "sae_dinov3_l24_spatial_d32000_k160.pth",
    "models/dinov3_l18_spatial/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth":     "sae_dinov3_l18_spatial_d20000_k80.pth",
    "models/dinov3_l24_cls/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth":         "sae_dinov3_l24_cls_d20000_k80.pth",
    "models/clip_l24_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth":      "sae_clip_l24_spatial_d32000_k160.pth",
    "models/clip_l24_cls/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth":           "sae_clip_l24_cls_d20000_k80.pth",
    "models/dinov3_l12_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth":    "sae_dinov3_l12_spatial_d32000_k160.pth",
    "models/clip_l16_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth":      "sae_clip_l16_spatial_d32000_k160.pth",
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
        repo_id=os.environ["HF_DATA_REPO"],
        repo_type="dataset",
        commit_message=f"Upload SAE weights {hf_name}",
    )
print("  SAE weights upload complete.")
PYEOF

# ── Tar + upload thumbnails ───────────────────────────────────────────────────
echo ""
echo "[3/3] Creating and uploading the thumbnails tarball..."
tar -czf "$TAR_PATH" -C "$(dirname "$IMAGES_DIR")" "$(basename "$IMAGES_DIR")"
du -sh "$TAR_PATH"
python - <<'PYEOF'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
tar_path = os.environ["TAR_PATH"]
size_mb = os.path.getsize(tar_path) / 1e6
print(f"  Uploading hf_images.tar.gz ({size_mb:.0f} MB)...", flush=True)
api.upload_file(
    path_or_fileobj=tar_path,
    path_in_repo="hf_images.tar.gz",
    repo_id=os.environ["HF_IMAGES_REPO"],
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
