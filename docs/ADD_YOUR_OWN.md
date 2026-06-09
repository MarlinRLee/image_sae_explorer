# Add your own SAE, data, and images

The demo serves whatever `configs/models.yaml` lists — the registry is the
single source of truth. Adding a model means producing two `.pt` sidecar
files for it, uploading them (plus the SAE checkpoint) to your HF dataset
repo, and appending one registry block. This doc walks the full pipeline;
the sidecar schema itself is in [DATA_FORMAT.md](DATA_FORMAT.md).

Training and precompute need a GPU and the extra deps:

```bash
pip install -r requirements-pipeline.txt
```

## 0. Point the registry at your repos (one-time)

Create two HF dataset repos (e.g. `<you>/sae-explorer-data` for `.pt` files +
SAE weights, `<you>/sae-explorer-images` for the thumbnails tarball), save a
write token to `~/.hf_token` (`chmod 600`), and set the registry defaults:

```yaml
defaults:
  hf_data_repo:   <you>/sae-explorer-data
  hf_images_repo: <you>/sae-explorer-images
  images_tarball: hf_images.tar.gz
```

Both the downloader (`demo/run_local.sh`) and the uploader
(`scripts/upload_hf.sh`) read these — change them once, everything follows.

## 1. Extract backbone activations

```bash
python scripts/extract_activations.py --backbone dinov3 \
    --data_dir /path/to/train_images \
    --output_dir shards/train
python scripts/extract_activations.py --backbone dinov3 \
    --data_dir /path/to/val_images \
    --output_dir shards/val
```

Supports `--backbone dinov3|clip`, `--layer N` for intermediate layers, and
`--token-type spatial|cls|both`. Output is float16 shards of shape
`(n_tokens, d_hidden)`.

## 2. Train a TopK SAE

```bash
python src/main.py shards/train --d-model 32000 --k-fraction 0.005 \
    --val-dir shards/val --mixed-precision
```

Checkpoints land in `checkpoints/`, final weights in
`models/sae_d<d>_k<k>_state_dict.pth`. Keep the `_k<top_k>` tag in the
filename — the explorer's SAE loader parses `top_k` out of it.

You can also skip steps 1–2 entirely and bring an externally trained TopK
SAE checkpoint, as long as its filename contains `_k<top_k>`.

## 3. Precompute the explorer sidecars (GPU)

```bash
bash scripts/precompute_all.sh \
    --sae-path  models/sae_d32000_k160_state_dict.pth \
    --image-dir /path/to/val_images \
    --output    explorer_data/explorer_data_my_sae.pt \
    --backbone  dinov3 --layer 24 \
    -- --d-model 32000 --top-k 160 --interleave-classes
```

This chains `precompute_explorer_data.py` (top images, UMAP, stats →
`explorer_data_my_sae.pt`) and `precompute_heatmaps.py` (per-feature patch
heatmaps → `explorer_data_my_sae_heatmaps.pt`). `--help` shows the
flag pass-through; you can also run the two scripts directly.

**This is where the image dataset is chosen.** The images you pass via
`--image-dir` (plus optionally `--extra-image-dir` for a second tree) are
the corpus the explorer displays — the image source is fixed at precompute
time, not serve time. To serve those images, collect them (or thumbnails of
them) into one directory; `upload_hf.sh full` tars and uploads it. The
tarball is shared by every model in a registry, so all models should be
precomputed against the same image corpus.

## 4. Optional enrichments

Each of these edits an existing `explorer_data_*.pt` in place (with a `.bak`
backup) and unlocks one explorer capability. All are skippable — absent
fields just disable the corresponding UI element.

| Script | Adds | Unlocks |
|---|---|---|
| `scripts/add_clip_embeddings.py --data <pt>` | `clip_text_scores`, `clip_feature_embeds` | CLIP text search panel |
| `scripts/add_interpretability_index.py --data <pt> --image-dir <imgs>` | `interp_index` | Sorting features by interpretability (Klindt et al. 2023, −LPIPS) |
| `scripts/auto_interp.py` | label JSON sidecars | Pre-populated Gemini feature labels (needs `GOOGLE_API_KEY`; budget-capped; syncs with the HF label JSONs) |

## 5. Upload

```bash
bash scripts/upload_hf.sh                                # data-only (label-safe)
ALLOW_LABEL_OVERWRITE=1 bash scripts/upload_hf.sh full   # + SAE weights, labels, thumbnails
```

The default data-only mode uploads each registry model's `.pt` +
`_heatmaps.pt` and nothing else — it can never clobber the feature labels
that live on HF. `full` mode additionally uploads label JSONs, the
thumbnails tarball, and each registry model's SAE checkpoint, looked up as
`models/<sae_file>` (override with `SAE_DIR`) — so name/copy your checkpoint
to match the `sae_file` you'll put in the registry.

## 6. Register, validate, run

Append one block to `configs/models.yaml`:

```yaml
  - id:         my_sae
    label:      My SAE
    data_file:  explorer_data_my_sae.pt
    sae_file:   sae_my_sae_d32000_k160.pth   # filename must contain _k<top_k>
    backbone:   dinov3
    layer:      24
    token_type: spatial
```

`backbone` / `layer` / `token_type` must match what you precomputed — the
on-demand heatmaps and the export-classifier button trust the registry.
Exactly one entry in the file must have `primary: true`.

Then:

```bash
bash demo/run_local.sh                                   # downloads the new files
python demo/validate_registry.py --data-dir local_data   # catches mismatches early
```

## Troubleshooting the pipeline

- **`KeyError: 'top_img_idx'` when the explorer loads your `.pt`** — the file
  is missing required fields; compare against
  [DATA_FORMAT.md](DATA_FORMAT.md) or the synthetic reference generator
  `demo/build_demo_data.py` (`_make_explorer_pt`).
- **Image grid shows gray placeholders** — the basenames stored in the `.pt`
  don't exist under the served `--image-dir`. Re-check step 3's image dirs,
  or pass `--extra-image-dir` if your images span two trees.
- **"CLIP text search unavailable for this dataset"** — run
  `scripts/add_clip_embeddings.py` on that `.pt` (step 4) and re-upload.
- **Export button says "Dead feature" / "No SAE checkpoint"** — pick a
  feature with `frequency > 0`, and make sure the registry block's
  `sae_file` name contains `_k<top_k>`.
- **A model errors mid-session** — `python demo/validate_registry.py
  --data-dir local_data` catches missing fields and backbone/token
  mismatches before serving.
