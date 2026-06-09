# Local quickstart

Two paths, depending on whether you want real model data or a synthetic
fixture. Both end with the demo open at http://localhost:5006/explorer_app.

## Option A — synthetic data (no internet, no HF account, ~30 seconds)

Best for the absolute first run, code review, or smoke tests after edits:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash demo/run_local.sh --synthetic
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

## Option B — real production data from HuggingFace (several GB)

For the actual user experience:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash demo/run_local.sh
```

This downloads, for **every** model listed in `configs/models.yaml`
(seven at the time of writing), the `.pt` sidecar, its `_heatmaps.pt`
companion, and the SAE `.pth` checkpoint from `Ramnie/sae-explorer-data`,
plus the thumbnail tarball `hf_images.tar.gz` from
`Ramnie/sae-explorer-images`. Expect several GB on first run; trim
`configs/models.yaml` to just the `primary: true` block if you only want
one model.

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

The Bokeh entry script is `demo/explorer_app.py`. It imports from a
sibling `demo/explorer/` package; if you're editing, this is the
"where do I look" map:

| File | Holds |
|---|---|
| `demo/explorer/state.py` | `_State`, `_UI`, argparse + validation helpers |
| `demo/explorer/context.py` | `Context` — the bundle of args/state/datasets passed to every panel |
| `demo/explorer/registry.py` | `configs/models.yaml` parsing + per-model metadata |
| `demo/explorer/loaders.py` | `.pt` / sidecar / names-JSON loading |
| `demo/explorer/images.py` | Pure image helpers — path resolution, opens, alpha colormaps, `pil_to_data_url` |
| `demo/explorer/rendering.py` | `render_heatmap_overlay`, hover thumbnails, the shared render executor, the prewarm thread |
| `demo/explorer/html_views.py` | `_status_html`, image-grid HTML, comparison HTML |
| `demo/explorer/activations.py` | `compute_patch_activations` + heatmap-reconstruction fallback (no GPU) |
| `demo/explorer/persistence.py` | Atomic JSON save + debounced HF dataset push |
| `demo/explorer/gemini.py` | "Label with Gemini" panel + API call |
| `demo/explorer/classifier_export.py` | "Export classifier (.py)" generator |
| `demo/explorer/panels/` | `feature_list`, `clip_search`, `cross_sae`, `patch_explorer`, `summary` — uniform `build(ctx)` factories |
| `demo/explorer_app.py` | Bokeh entry point — UMAP view, dataset switch, layout, `curdoc().add_root()` |

Every helper module is callable in isolation (e.g. `from explorer.html_views
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
  `demo/explorer/state.py` (`_State._REQUIRED`) or regenerate with
  `demo/build_demo_data.py`.
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
