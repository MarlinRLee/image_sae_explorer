# Local quickstart

Two paths, depending on whether you want real model data or a synthetic
fixture. Both end with the demo open at http://localhost:5006/explorer_app.

## Option A — synthetic data (no internet, no HF account, ~30 seconds)

Best for the absolute first run, code review, or smoke tests after edits:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_explorer_local.sh --synthetic
```

The launcher generates `./demo_data/explorer_data_demo.pt`,
`_heatmaps.pt`, a sister `_demo2.pt` with CLIP, and five random JPEGs
under `./demo_data/images/`. It then starts the Bokeh server.

Synthetic features have realistic shapes — about half the dictionary is
"dead", half "live" — so every UI branch (UMAP `live_mask`, dead-feature
banner, no-image slot, dataset-switch CLIP availability transition) is
exercised. Random JPEGs make for unhelpful heatmaps but the layout, the
clicks, and the dataset switch are all real.

Iterate by re-running. The launcher reuses an existing `./demo_data/`
unless you delete it.

## Option B — real sample data from HuggingFace (~5 minutes, ~250 MB)

For the actual user experience:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_explorer_local.sh
```

This downloads:

- `explorer_data_dinov2_layer11_d10000_k100_val.pt` (~50 MB) — the
  production primary SAE.
- `explorer_data_dinov2_layer11_d10000_k100_val_heatmaps.pt` (~50 MB) —
  the matching heatmap sidecar.
- `hf_images.tar.gz` from `Ramnie/sae-explorer-images` (~150 MB) —
  thumbnail JPEGs the explorer uses to render image grids.

Downloads are idempotent — re-running skips files already present in
`./local_data/` and `./local_images/`.

## Tour of the UI

Once the demo opens, you'll see (left → right, top → bottom):

- **Active model dropdown** — switches between the primary dataset and
  any `--compare-data` extras (e.g. the synthetic CLIP variant).
- **UMAP scatter** — every dot is a feature. Hover for top-3 thumbnails;
  click to drive the detail panel.
- **Feature detail** — name input, Gemini auto-interp button (needs
  `GOOGLE_API_KEY`), and the heatmap grids: Top, Mean, 75th-percentile,
  plus a "Compare aggregations" view.
- **Sortable feature list** — searchable by manual or auto-interp name.
- **Patch Explorer** (collapsible) — load any image by index, click /
  drag patches to find which features fire on those tokens.
- **CLIP Text Search** (collapsible) — works against precomputed CLIP
  scores; falls back to live encoding for free-text queries when
  `transformers` is installed and a CLIP model is reachable.
- **Cross-SAE Compare** (collapsible) — side-by-side feature renders
  across two SAEs; SAE A auto-fills with the currently selected feature.

## Where things live in the source

The Bokeh entry script is `scripts/explorer_app.py`. It imports from a
sibling `scripts/explorer/` package; if you're editing, this is the
"where do I look" map:

| File | Holds |
|---|---|
| `scripts/explorer/state.py` | `_State`, `_UI`, argparse + validation helpers |
| `scripts/explorer/runtime.py` | `args` / `state` / `ui` / `datasets` slots + `load_image` (late-bound runtime references) |
| `scripts/explorer/images.py` | Pure image helpers — path resolution, opens, alpha colormaps, `pil_to_data_url` |
| `scripts/explorer/rendering.py` | `render_heatmap_overlay`, hover thumbnails, the shared `_render_executor`, the prewarm thread |
| `scripts/explorer/html_views.py` | `_status_html`, image-grid HTML, comparison HTML |
| `scripts/explorer/activations.py` | `compute_patch_activations` + heatmap-reconstruction fallback (no GPU) |
| `scripts/explorer/persistence.py` | Atomic JSON save + debounced HF dataset push |
| `scripts/explorer_app.py` | Bokeh widget construction, callbacks, layout, `curdoc().add_root()` |

The bootstrap is the only file that mutates module-level Bokeh widgets;
every helper is callable in isolation (e.g. `from explorer.html_views
import _status_html` is fine outside Bokeh).

## Optional features

| Feature | Enable by | Notes |
|---|---|---|
| Free-text CLIP search | active dataset has `clip_feature_embeds` | Lazy-loads `transformers` on first out-of-vocab query |
| Gemini auto-interp | `export GOOGLE_API_KEY=...` | Disabled-state button if unset |
| Persisted feature names | `export HF_TOKEN=...` and `export HF_DATASET_REPO=Ramnie/sae-explorer-data` | Names debounce-push 2 s after the last edit |

Without any of those env vars, the demo still runs — the buttons either
disable themselves or no-op silently.

## Troubleshooting

- **`KeyError: 'top_img_idx'`** when loading a `.pt` — the file is
  missing keys the loader requires. Check the schema in
  `scripts/explorer/state.py` (`_State._REQUIRED`) or regenerate with
  `scripts/build_demo_data.py`.
- **Image grid is gray** — the basenames in `image_paths` don't exist
  under `--image-dir`. Set `--extra-image-dir` to a second tree if your
  thumbnails are split.
- **"CLIP text search unavailable for this dataset"** — switch to a
  CLIP-enabled dataset in the dropdown, or run
  `scripts/add_clip_embeddings.py` against the active `.pt`.
- **HF Space-style auto-interp not appearing** — the JSON sidecar
  (`<data>_auto_interp.json`) sits next to the `.pt`; the launcher
  doesn't download it. Either `hf_hub_download` it manually or
  `--names-file` to a local copy.
