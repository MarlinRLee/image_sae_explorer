# SAE Feature Explorer

Interactive Bokeh-served explorer for Sparse Autoencoder (SAE) features
trained on vision-transformer activations (DINOv2 / DINOv3 / CLIP). Click
features in a UMAP, see their top-activating images with heatmap overlays,
search by CLIP text, label features by hand or with Gemini, and **export any
feature as a standalone Python image classifier**.

A live deployment runs at https://huggingface.co/spaces/Ramnie/sae-explorer.

---

## Quickstart

Requires **Python 3.10+**. The demo runs on CPU; no GPU is required to serve.

```bash
git clone https://github.com/MarlinRLee/sae-explorer.git
cd sae-explorer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_explorer_local.sh
```

Open http://localhost:5006/explorer_app.

`run_explorer_local.sh` reads `configs/models.yaml`, downloads every listed
model (`.pt` sidecar + `_heatmaps.pt` + SAE `.pth`) from the Hugging Face
dataset repo into `./local_data/`, downloads + extracts the thumbnails tarball
into `./local_images/`, and launches `bokeh serve` on port 5006. Re-runs skip
already-downloaded files.

> The default `pip install torch` pulls the CUDA wheel (~2 GB). For CPU only:
> `pip install --index-url https://download.pytorch.org/whl/cpu torch`

## Export a feature as a classifier

Select a feature and click **Export classifier (.py)** (next to the name
field). You get a self-contained `sae_classifier_feat<N>.py` that, with **no
arguments**, downloads the backbone + SAE + the demo image tarball, scores the
feature's own top-activating image, and prints the activation:

```bash
pip install torch torchvision transformers huggingface_hub Overcomplete Pillow
python sae_classifier_feat123.py                 # scores a guaranteed-positive toy image
python sae_classifier_feat123.py my_image.jpg    # score your own image
python sae_classifier_feat123.py my_image.jpg --threshold 0.5 --agg mean
```

It also exposes `make_classifier()` for use as a library — a one-feature
"export classifier" you can drop into other code. The generator lives in
`scripts/explorer/classifier_export.py`.

## Add your own SAE + data

The model list lives entirely in `configs/models.yaml` — add one block and the
dropdown picks it up. The data files behind each block follow the schema in
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).

1. **Train** an SAE on backbone activations:
   ```bash
   python src/main.py <shards-dir> SI-SAE --d-model 32000 ...
   ```
2. **Precompute** the explorer sidecars (GPU; runs both precompute steps):
   ```bash
   bash scripts/precompute_all.sh \
       --sae-path  <sae>_k160.pth \
       --image-dir /scratch/val \
       --output    explorer_data_my_new_sae.pt \
       --backbone  dinov3 --layer 24 \
       -- --d-model 32000 --top-k 160 --interleave-classes
   ```
   (Or call `scripts/precompute_explorer_data.py` then
   `scripts/precompute_heatmaps.py` directly — `precompute_all.sh --help` shows
   the equivalent flags.)
3. **Upload** the sidecar + heatmaps + SAE checkpoint to your HF dataset repo:
   ```bash
   bash scripts/submit_hf_upload.sh
   ```
4. **Append** one block to `configs/models.yaml`:
   ```yaml
   - id:         my_new_sae
     label:      My New SAE
     data_file:  explorer_data_my_new_sae.pt
     sae_file:   sae_my_new_sae_k160.pth     # filename must contain _k<top_k>_
     backbone:   dinov3
     layer:      24
     token_type: spatial
   ```
   `backbone` / `layer` / `token_type` must match what you precomputed — the
   on-demand heatmap and export-classifier inference trust the registry.
5. **Validate** the registry before serving:
   ```bash
   python scripts/validate_registry.py --data-dir local_data
   ```
6. **Re-run** locally (downloads the new files, picks them up automatically):
   ```bash
   bash scripts/run_explorer_local.sh
   ```

### Using a new image dataset

The image source is fixed at precompute time, not serve time — point step 2 at
your images with `--image-dir /path/to/images` (add `--extra-image-dir` if they
span two trees). Upload the resulting `.pt` files plus a thumbnail tarball; the
tarball repo is shared across all models in a registry
(`defaults.hf_images_repo`), so change it there to swap the image source for
everything.

## Deploy to a Hugging Face Space

`hf_space/` is a clone of the production Space repo (gitignored in this repo,
so it is its own git checkout). To deploy:

1. Point `hf_space/` at your own Space (one-time): create a Space, then
   `git -C hf_space remote set-url origin https://huggingface.co/spaces/<you>/<space>`.
2. Update `configs/models.yaml` with the models the Space should serve.
3. Sync the canonical source into `hf_space/` and push:
   ```bash
   bash scripts/sync_hf_space.sh && (cd hf_space && git push)
   ```

`sync_hf_space.sh` rsyncs the **working tree** (not `HEAD`) of `scripts/`,
`configs/`, and `src/clip_utils.py` into `hf_space/`, so uncommitted edits
deploy. The Space bakes every registry model into its Docker image at build
time (`hf_space/Dockerfile`) for instant cold-start.

## Optional features

| Feature | Enable by | What it does |
|---|---|---|
| Free-text CLIP search | (always on if `transformers` installed) | Encodes any query with CLIP and ranks features by cosine similarity to image embeds. |
| Gemini auto-interp | export `GOOGLE_API_KEY=...` | "Label with Gemini" button calls `gemini-2.5-flash` on the top-activating images and saves the returned label. |
| Persisted feature names | export `HF_TOKEN=...` and `HF_DATASET_REPO=Ramnie/sae-explorer-data` | Names typed into the demo are debounce-pushed to the HF dataset repo so they persist across sessions. **Without these, edits are session-local.** |

All three are inert without the relevant environment variables. Export them in
the same shell you launch from — `bash scripts/run_explorer_local.sh` inherits
the env into the bokeh subprocess:

```bash
export GOOGLE_API_KEY=AIza...        # enables the Label with Gemini button
export HF_TOKEN=hf_...               # enables persisted feature names
bash scripts/run_explorer_local.sh
```

Add the `export` lines to `~/.bashrc` (or a project-local `.env` you `source`)
to make them stick across terminals.

## Project layout

```
configs/
  models.yaml                # registry — single source of truth for the demo
docs/
  DATA_FORMAT.md             # explorer_data*.pt / _heatmaps.pt schema
scripts/
  explorer_app.py            # Bokeh entry point (composes panels)
  explorer/                  # state, rendering, persistence, html_views, images,
                             # activations, registry, loaders, classifier_export
  explorer/panels/           # feature_list, clip_search, cross_sae,
                             # patch_explorer, summary
  bootstrap_demo.py          # registry-driven downloader + launcher
  run_explorer_local.sh      # local launcher (calls bootstrap_demo.py)
  precompute_*.py            # sidecar generation pipeline
  precompute_all.sh          # chains both precompute steps for one SAE
  validate_registry.py       # sanity-check models.yaml against the .pt files
  submit_hf_upload.sh        # upload sidecars + SAE to the HF dataset repo
  sync_hf_space.sh           # push canonical source to the HF Space repo
hf_space/                    # production HF Space repo (gitignored)
src/                         # training / precompute source (separate concern)
requirements.txt             # demo runtime deps
```

## Troubleshooting

- **`bokeh serve` exits with `--data-dir is not a directory`** — the registry's
  `data_file` paths are resolved relative to `--data-dir`. Run
  `bash scripts/run_explorer_local.sh` first to populate `./local_data/`.
- **Image grid shows gray placeholders** — `--image-dir` doesn't contain the
  basenames stored in the `.pt` file. Set `--extra-image-dir` to a second
  directory if your images live in two trees.
- **"CLIP text search unavailable for this dataset"** — the active dataset has
  no precomputed CLIP scores. Run `scripts/add_clip_embeddings.py` against that
  `.pt`, or switch to a CLIP-enabled dataset in the dropdown.
- **Export button says "Dead feature" / "No SAE checkpoint"** — pick a feature
  that fired (`frequency > 0`), and make sure the active model's registry block
  has a `sae_file` whose name contains `_k<top_k>_`.
- **A new model errors mid-session** — run
  `python scripts/validate_registry.py --data-dir local_data` to catch missing
  fields or backbone/token mismatches before serving.
