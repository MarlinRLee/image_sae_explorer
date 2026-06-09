# SAE Feature Explorer

Interactive Bokeh-served explorer for Sparse Autoencoder (SAE) features
trained on vision-transformer activations (DINOv3 / CLIP). Click features in
a UMAP, see their top-activating images with heatmap overlays, search by CLIP
text, label features by hand or with Gemini, and export any feature as a
standalone Python image classifier.

A live deployment runs at https://huggingface.co/spaces/Ramnie/sae-explorer.

## Quickstart

Requires **Python 3.10+**. Serving runs on CPU; no GPU needed.

```bash
git clone https://github.com/MarlinRLee/image_sae_explorer.git
cd image_sae_explorer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash demo/run_local.sh
```

Open http://localhost:5006/explorer_app.

The launcher downloads every model listed in `configs/models.yaml` (sidecar
`.pt` + heatmaps + SAE weights, several GB total) plus a thumbnails tarball,
then starts the Bokeh server. Re-runs skip already-downloaded files. Two
useful variants:

- **`bash demo/run_local.sh --synthetic`** — ~30-second offline smoke test on
  generated data; no downloads, no HF account.
- **One model instead of seven** — trim `configs/models.yaml` to just the
  `primary: true` block before the first run.

> `pip install torch` pulls the CUDA wheel (~2 GB). For CPU only:
> `pip install --index-url https://download.pytorch.org/whl/cpu torch`

### What you'll see

- **UMAP scatter** — every dot is a feature; hover for top-3 thumbnails,
  click to drive the detail panel.
- **Feature detail** — name input, Gemini label button, and heatmap grids
  (top / mean / 75th-percentile aggregations, plus a compare view).
- **Sortable feature list** — searchable by manual or auto-interp name.
- **Patch Explorer** — load any image, click/drag patches to find which
  features fire on those tokens.
- **CLIP Text Search** — rank features against any text query.
- **Cross-SAE Compare** — side-by-side feature renders across two SAEs.

## Export a feature as a classifier

Select a feature and click **Export classifier (.py)**. You get a
self-contained `sae_classifier_feat<N>.py` that, with no arguments, downloads
the backbone + SAE + demo images, scores the feature's own top-activating
image, and prints the activation:

```bash
pip install torch torchvision transformers huggingface_hub Overcomplete Pillow
python sae_classifier_feat123.py                 # scores a guaranteed-positive toy image
python sae_classifier_feat123.py my_image.jpg --threshold 0.5
```

It also exposes `make_classifier()` for use as a library.

## Add your own SAE + data

`configs/models.yaml` is the single source of truth — add one block and the
model dropdown picks it up. The pipeline, end to end:

1. `scripts/extract_activations.py` — backbone activation shards
2. `python src/main.py <shards>` — train a TopK SAE
3. `bash scripts/precompute_all.sh` — build the explorer sidecars (GPU); your
   image dataset is chosen here via `--image-dir`
4. optional enrichments — CLIP text search, interpretability index, bulk
   Gemini labels
5. `bash scripts/upload_hf.sh` — push to your HF dataset repo
6. append a registry block, `python demo/validate_registry.py`, re-run

Step-by-step instructions are in
[`docs/ADD_YOUR_OWN.md`](docs/ADD_YOUR_OWN.md); the `.pt` schema is in
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).

## Optional features

All inert without their env var — export it in the shell you launch from:

| Feature | Enable by |
|---|---|
| Gemini auto-interp button | `export GOOGLE_API_KEY=...` |
| Feature names persisted to HF (otherwise session-local) | `export HF_TOKEN=...` and `export HF_DATASET_REPO=<you>/sae-explorer-data` |
| Free-text CLIP search | on by default when `transformers` is installed |

## Project layout

```
configs/models.yaml          # model registry — single source of truth
docs/                        # ADD_YOUR_OWN.md pipeline guide, DATA_FORMAT.md schema
demo/                        # everything needed to RUN the explorer
  run_local.sh               # launcher (downloads data, starts bokeh serve)
  explorer_app.py            # Bokeh entry point
  explorer/                  # app modules + panels
  build_demo_data.py         # synthetic dataset for --synthetic
  validate_registry.py       # sanity-check models.yaml against the .pt files
scripts/                     # pipeline that PREPARES demo data (GPU)
src/                         # SAE training + shared inference helpers
requirements.txt             # demo runtime deps
requirements-pipeline.txt    # extra deps for training / precompute
```

## Troubleshooting

- **`--data-dir is not a directory`** — run `bash demo/run_local.sh` first to
  populate `./local_data/`; registry paths are resolved relative to it.
- **Image grid shows gray placeholders** — `--image-dir` doesn't contain the
  basenames stored in the `.pt` file.
- **"CLIP text search unavailable for this dataset"** — the active dataset
  has no precomputed CLIP scores; switch datasets or see
  [`docs/ADD_YOUR_OWN.md`](docs/ADD_YOUR_OWN.md).
- **A model errors mid-session** — run
  `python demo/validate_registry.py --data-dir local_data` to catch registry
  mismatches before serving.
