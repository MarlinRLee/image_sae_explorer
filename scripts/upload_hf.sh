#!/usr/bin/env bash
# Upload explorer data to your Hugging Face dataset repos.
#
# Usage:
#   bash scripts/upload_hf.sh                                  # data-only (default, label-safe)
#   ALLOW_LABEL_OVERWRITE=1 bash scripts/upload_hf.sh full     # everything
#
# Modes
#   data-only (default)
#       Uploads each registry model's .pt and _heatmaps.pt — nothing else.
#       Deliberately uploads NO label JSONs: labels (feature_names /
#       auto_interp, plus their _authors/_history files) are edited live in
#       the explorer and the canonical copies live on HF, so re-uploading
#       stale local ones would clobber them. Feature indices are stable while
#       the SAE weights are unchanged, so existing labels keep matching
#       regenerated .pt files.
#   full
#       Additionally uploads the label JSONs, each registry model's SAE
#       checkpoint (looked up as $SAE_DIR/<sae_file>; missing files are
#       skipped with a note), and the thumbnails tarball.
#       !! Overwrites human + Gemini labels on the data repo — requires
#       ALLOW_LABEL_OVERWRITE=1 as a deliberate second step.
#
# Everything is driven by the registry (configs/models.yaml): the file list
# comes from each model's data_file/sae_file, and the target repos default to
# the registry's `defaults:` block. Point those defaults at YOUR repos and
# both download (demo/run_local.sh) and upload use them.
#
# One-time setup:
#   1. Create two HF dataset repos on huggingface.co:
#        <you>/sae-explorer-data    (.pt sidecars + SAE weights)
#        <you>/sae-explorer-images  (thumbnails tarball)
#      and put them in the `defaults:` block of configs/models.yaml.
#   2. Save a write token to ~/.hf_token  (chmod 600 ~/.hf_token).
#
# Env overrides: REGISTRY, HF_DATA_REPO, HF_IMAGES_REPO, HF_TOKEN_FILE,
#   PT_DIR (parent of explorer_data/), SAE_DIR (default ./models),
#   IMAGES_DIR (default ~/hf_images), TAR_PATH (default ~/hf_images.tar.gz).

set -euo pipefail

MODE="${1:-data-only}"
if [[ "$MODE" != "data-only" && "$MODE" != "full" ]]; then
    echo "ERROR: unknown mode '$MODE' (expected 'data-only' or 'full')" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REGISTRY="${REGISTRY:-${REPO_ROOT}/configs/models.yaml}"

# Repo names default to the registry's `defaults:` block.
read -r REG_DATA_REPO REG_IMAGES_REPO REG_TARBALL < <(python - "$REGISTRY" <<'PYEOF'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])).get("defaults") or {}
print(d.get("hf_data_repo", ""), d.get("hf_images_repo", ""),
      d.get("images_tarball", "hf_images.tar.gz"))
PYEOF
)
export HF_DATA_REPO="${HF_DATA_REPO:-$REG_DATA_REPO}"
export HF_IMAGES_REPO="${HF_IMAGES_REPO:-$REG_IMAGES_REPO}"
export IMAGES_TARBALL="${REG_TARBALL}"
if [[ -z "$HF_DATA_REPO" ]]; then
    echo "ERROR: no data repo — set defaults.hf_data_repo in $REGISTRY (or HF_DATA_REPO)" >&2
    exit 1
fi

export PT_DIR="${PT_DIR:-${REPO_ROOT}}"            # parent of explorer_data/
export SAE_DIR="${SAE_DIR:-${REPO_ROOT}/models}"   # holds <sae_file> checkpoints
IMAGES_DIR="${IMAGES_DIR:-${HOME}/hf_images}"
export TAR_PATH="${TAR_PATH:-${HOME}/hf_images.tar.gz}"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.hf_token}"
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ ! -f "$HF_TOKEN_FILE" ]]; then
        echo "ERROR: token file not found: $HF_TOKEN_FILE" >&2
        echo "Save your HF write token there: echo 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token" >&2
        exit 1
    fi
    HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
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
[[ "$MODE" == "full" ]] && echo "  SAE dir:     $SAE_DIR"
echo "============================================"

if [[ "$MODE" == "full" ]]; then
    echo "Creating thumbnails tarball from $IMAGES_DIR ..."
    tar -czf "$TAR_PATH" -C "$(dirname "$IMAGES_DIR")" "$(basename "$IMAGES_DIR")"
    du -sh "$TAR_PATH"
fi

python - <<'PYEOF'
import glob, os, sys
import yaml
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
ddir = os.path.join(os.environ["PT_DIR"], "explorer_data")
full = os.environ["UPLOAD_MODE"] == "full"

with open(os.environ["REGISTRY"]) as fh:
    registry = yaml.safe_load(fh)


def upload(path, repo, name=None):
    name = name or os.path.basename(path)
    print(f"  Uploading {name} ({os.path.getsize(path) / 1e6:.0f} MB)...", flush=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo,
                    repo_type="dataset", commit_message=f"Upload {name}")


# ── .pt data files (registry-listed; all must exist) ─────────────────────
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

print(f"[1/{3 if full else 1}] Uploading explorer data files...")
for fpath in sorted(set(files)):
    upload(fpath, os.environ["HF_DATA_REPO"])

if not full:
    print("Done (data-only). SAE weights, label JSONs and thumbnails untouched.")
    sys.exit(0)

# ── SAE checkpoints (full mode; $SAE_DIR/<sae_file> per registry entry) ──
print("[2/3] Uploading SAE weights...")
sae_dir = os.environ["SAE_DIR"]
for model in registry["models"]:
    sae_file = model.get("sae_file")
    if not sae_file:
        continue
    path = os.path.join(sae_dir, sae_file)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        continue
    upload(path, os.environ["HF_DATA_REPO"])

# ── Thumbnails tarball (full mode; tar created by the bash wrapper) ──────
print("[3/3] Uploading the thumbnails tarball...")
upload(os.environ["TAR_PATH"], os.environ["HF_IMAGES_REPO"],
       name=os.environ["IMAGES_TARBALL"])
PYEOF

echo ""
echo "Upload complete."
if [[ "$MODE" == "full" ]]; then
    echo "If deploying a HF Space, set its Secrets:"
    echo "  HF_DATASET_REPO = $HF_DATA_REPO"
    echo "  HF_IMAGES_REPO  = $HF_IMAGES_REPO"
    echo "  HF_TOKEN        = (a write-access token)"
fi
