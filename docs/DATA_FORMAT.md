# Explorer data format

The explorer serves one model from a pair of files plus a few optional JSON
sidecars:

```
explorer_data_<name>.pt            # required — features, top images, UMAP, stats
explorer_data_<name>_heatmaps.pt   # optional — per-feature patch heatmaps
explorer_data_<name>_*.json        # optional — human/auto feature labels (written at serve time)
```

The `.pt` files are plain `torch.save` dicts (load with
`torch.load(path, weights_only=False)`). They are produced by
`scripts/precompute_explorer_data.py` and `scripts/precompute_heatmaps.py`
(see `scripts/precompute_all.sh` to run both at once). A minimal synthetic
example is built by `demo/build_demo_data.py` — read `_make_explorer_pt`
there for a runnable reference, and validate any `.pt` against the registry
with `python demo/validate_registry.py --data-dir <dir>`.

Shapes below use `D` = dictionary size (number of SAE features), `N` = number
of images, `S` = number of stored image slots per feature, `P` = `patch_grid`.

## `explorer_data_<name>.pt` — required fields

| Field | Type / shape | Meaning |
|---|---|---|
| `image_paths` | `list[str]`, len `N` | Image filenames (basenames resolved against `--image-dir`). |
| `d_model` | `int` | Dictionary size `D` (number of SAE features). |
| `n_images` | `int` | `N`. |
| `patch_grid` | `int` | `P` — backbone produces `P×P` spatial patch tokens. |
| `image_size` | `int` | Square input size used during precompute (e.g. 224/256). |
| `top_img_idx` | `int64 (D, S)` | Per feature, indices into `image_paths` of its top-activating images (`-1` = empty slot). |
| `top_img_act` | `float32 (D, S)` | Peak (max-over-patches) activation for each top slot. |
| `mean_img_idx` | `int64 (D, S)` | Top images ranked by **mean** activation over the image's patches. |
| `mean_img_act` | `float32 (D, S)` | Mean-aggregated activation for each `mean` slot. |
| `crop_img_idx` | `int64 (D, S)` | Top images ranked by **crop-and-average** (mean of the feature's top-K patches). |
| `crop_img_act` | `float32 (D, S)` | Crop-aggregated activation for each `crop` slot. |
| `feature_frequency` | `int64 (D,)` | How often each feature fired across the corpus (`0` = dead feature). |
| `feature_mean_act` | `float32 (D,)` | Mean activation per feature over firing patches. |
| `umap_coords` | `float32 (D, 2)` | UMAP of feature **activation** profiles; `NaN` rows = features excluded from the map (dead/non-live). |
| `dict_umap_coords` | `float32 (D, 2)` | UMAP of the SAE **decoder dictionary** vectors; `NaN` = excluded. |

The three aggregations (`top` = max, `mean` = average, `crop` = top-K-patch
average) are independent rankings of the same corpus; the explorer lets you
switch between them so a feature can be inspected by peak response, by
whole-image response, and by localized-region response.

## `explorer_data_<name>.pt` — optional fields

Absent fields are treated as `None` (see `demo/explorer/state.py`
`_OPTIONAL`). Older precompute outputs simply omit them.

| Field | Type / shape | Meaning |
|---|---|---|
| `backbone` | `str` | `dinov3` / `dinov2` / `clip` / `demo`. Used by on-demand inference and the export-classifier button; **must match the registry block.** |
| `layer` | `int` or absent | Intermediate backbone layer; absent/`None` = final layer. |
| `token_type` | `str` | `spatial` (patch tokens) or `cls` (global token). Must match the registry. |
| `p75_img_idx` / `p75_img_act` | `(D, S)` | 75th-percentile-ranked image slots (extra aggregation; not required). |
| `feature_p75_val` | `float32 (D,)` | Per-feature 75th-percentile activation. |
| `clip_text_scores` | `float32 (D, V)` | Cosine of each feature's image embeds vs a `V`-word vocab — powers CLIP text ranking. |
| `clip_text_vocab` | `list[str]`, len `V` | The vocab words. |
| `clip_feature_embeds` | `float32 (D, C)` | Per-feature CLIP image embedding (`C` = CLIP dim), for free-text search. |
| `interp_index` | `float32 (D,)` | Interpretability Index (Klindt et al. 2023), stored as `-LPIPS`. |

## `explorer_data_<name>_heatmaps.pt` — optional sidecar

Per-feature, per-slot patch heatmaps overlaid on the top images. Missing → the
explorer recomputes heatmaps on demand (backbone + SAE inference). Built by
`scripts/precompute_heatmaps.py`.

| Field | Type / shape | Meaning |
|---|---|---|
| `top_heatmaps` | `float16 (D, S, P²)` | Patch activations for each `top` slot, flattened `P×P`. |
| `mean_heatmaps` | `float16 (D, S, P²)` | …for each `mean` slot. |
| `crop_heatmaps` | `float16 (D, S, P²)` | …for each `crop` slot. |
| `p75_heatmaps` | `float16 (D, S, P²)` | …for each `p75` slot (if present). |
| `patch_grid` | `int` | `P` for the stored heatmaps (may differ from the data file's grid). |

## JSON label sidecars (written at serve time)

Created/updated by the explorer as you (or Gemini) name features; not part of
precompute. Loaded by `demo/explorer/loaders.py`. All are
`{feature_id: value}` maps keyed by stringified feature index.

| File suffix | Content |
|---|---|
| `_feature_names.json` | Hand-entered feature labels. |
| `_auto_interp.json` | Gemini-generated labels. |
| `_feature_names_authors.json` / `_auto_interp_authors.json` | Who set each label. |
| `_history.json` / `_auto_interp_history.json` | Per-feature `(label, author, timestamp)` history. |

See [ADD_YOUR_OWN.md](ADD_YOUR_OWN.md) for how these files are generated and
uploaded.
