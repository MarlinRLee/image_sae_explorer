"""
Interactive SAE Feature Explorer - Bokeh Server App.

Visualizes SAE features with:
  - UMAP scatter plot of features (activation-based and dictionary-based)
  - Click a feature to see its top-activating images with heatmap overlays
  - 75th percentile images for distribution understanding
  - Patch explorer: click patches of any image to find active features
  - Feature naming: assign names to features, saved to JSON, searchable

All display is driven by pre-computed sidecars (_heatmaps.pt, _patch_acts.pt).
No GPU or model weights are required at serve time.

The model list (primary + compares) comes from a YAML registry — see
``configs/models.yaml`` and ``scripts/explorer/registry.py``.

Launch:
    bokeh serve scripts/explorer_app.py --port 5006 \
        --allow-websocket-origin="*" --session-token-expiration 86400 \
        --args \
          --registry  configs/models.yaml \
          --data-dir  ./local_data \
          --image-dir ./local_images

Open: http://localhost:5006/explorer_app
"""

import os
import random
import sys
import uuid

import numpy as np
from PIL import Image

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.events import MouseMove
from bokeh.models import (
    ColumnDataSource, HoverTool, Div, Select, TextInput, Button,
    Slider, Toggle, CustomJS, InlineStyleSheet,
)
from bokeh.plotting import figure
from bokeh.palettes import Turbo256
from bokeh.transform import linear_cmap

# Make the sibling `explorer/` package and the project's src/ directory both
# importable from this entry script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
from explorer import images as _images_mod
from explorer import persistence as _persistence_mod
from explorer import gemini as _gemini
from explorer import classifier_export as _classifier_export
from explorer.context import Context
from explorer.state import _parse_args, _validate_args
from explorer.persistence import _save_json_atomic, _archive_label
from explorer.registry import load_registry
from explorer.loaders import build_dataset_list, ensure_loaded as _loaders_ensure_loaded
from explorer.html_views import (
    make_image_grid_html, make_compare_aggregations_html,
    image_card_width,
)
from explorer.rendering import (
    _render_executor, UMAP_THUMB_PX,
    render_heatmap_overlay, _mei_data_urls, _prewarm_hover_cache_async,
)
from explorer.panels import clip_search as _clip_search_mod
from explorer.panels import cross_sae as _cross_sae_mod
from explorer.panels import feature_list as _feature_list_mod
from explorer.panels import patch_explorer as _patch_explorer_mod
from explorer.panels import summary as _summary_mod


# ---------- Parse args ----------
# `_build_parser`, `_parse_args`, `_validate_args` live in explorer/state.py.
# Bokeh serve runs this file as a script, so parsing at import is unavoidable.
args = _parse_args()
_validate_args(args)


# ---------- Per-session context ----------
# `Context` (explorer/context.py) owns this session's `state` + `ui` and the
# late-bound `select_feature` / `ensure_loaded` callables. It is threaded
# explicitly into every panel `build(ctx, ...)` and rendering helper, which
# is what keeps concurrent browser sessions isolated. The module-level
# `state` / `ui` names below are thin aliases onto `ctx` so the rest of this
# bootstrap reads naturally.
_registry = load_registry(args.registry)
_all_datasets = build_dataset_list(_registry, args.data_dir, names_file=args.names_file)

ctx = Context(args, _all_datasets)
ctx.state.apply(0)
state = ctx.state
ui = ctx.ui


def _ensure_loaded(idx):
    """Load dataset at idx if it is still a lazy placeholder."""
    _loaders_ensure_loaded(_all_datasets, idx)


ctx.ensure_loaded = _ensure_loaded
# `ctx.select_feature` is wired further down, once `_select_feature` exists.


# `_save_json_atomic`, `_schedule_hf_push`, `_archive_label`, `_now_iso`
# all live in explorer/persistence.py. The thin wrappers below tie the
# save primitives to the active `state` and `ui`. Each save writes the
# labels JSON, the matching authors sidecar (who created each label),
# and the history sidecar (every label that has been overwritten or
# deleted) so the history is preserved without showing stale entries
# in the UI.
def _save_names():
    _save_json_atomic(state.names_file,
                      {str(k): v for k, v in sorted(state.feature_names.items())})
    _save_json_atomic(state.feature_name_authors_file,
                      {str(k): v for k, v in sorted(state.feature_name_authors.items())})
    _save_json_atomic(state.feature_names_history_file,
                      {str(k): v for k, v in sorted(state.feature_names_history.items())})
    print(f"Saved {len(state.feature_names)} feature names to {state.names_file}")
    _persistence_mod._schedule_hf_push(
        [state.names_file,
         state.feature_name_authors_file,
         state.feature_names_history_file], ui)


def _save_auto_interp():
    _save_json_atomic(state.auto_interp_file,
                      {str(k): v for k, v in sorted(state.auto_interp_names.items())})
    _save_json_atomic(state.auto_interp_authors_file,
                      {str(k): v for k, v in sorted(state.auto_interp_authors.items())})
    _save_json_atomic(state.auto_interp_history_file,
                      {str(k): v for k, v in sorted(state.auto_interp_history.items())})
    print(f"Saved {len(state.auto_interp_names)} auto-interp labels to {state.auto_interp_file}")
    _persistence_mod._schedule_hf_push(
        [state.auto_interp_file,
         state.auto_interp_authors_file,
         state.auto_interp_history_file], ui)


def _display_name(feat: int) -> str:
    """Return the label to show in tables: manual label takes priority over auto-interp."""
    m = state.feature_names.get(feat)
    if m:
        return m
    a = state.auto_interp_names.get(feat)
    return f"[auto] {a}" if a else ""


# `compute_patch_activations` lives in explorer/activations.py and is
# imported above; it lazy-loads backbone + SAE for on-demand inference
# when the dataset has no patch_acts sidecar.


# ---------- Image helpers ----------
# The pure helpers (_resolve_image_path, _open_image, _load_image_from_ds,
# pil_to_data_url, _pil_to_bokeh_rgba, ALPHA_JET/VIRIDIS, create_alpha_cmap,
# _missing_image_warned) live in explorer/images.py and are imported at the
# top of this file. Only `load_image` stays here because it closes over
# `state.image_paths` (which is still module-level during the refactor).
# `images.IMAGE_DIRS` and `images.THUMB` are configured below from `args`.
_images_mod.IMAGE_DIRS = tuple(d for d in (args.image_dir, args.extra_image_dir) if d)
_images_mod.THUMB = args.thumb_size
THUMB = args.thumb_size  # local alias for legacy in-file references
# `load_image` is imported from explorer.runtime above (it uses runtime.state).


# Heatmap-overlay + hover-thumbnail rendering helpers all live in
# explorer/rendering.py (imported above).


# `make_image_grid_html`, `make_compare_aggregations_html`
# live in explorer/html_views.py and are imported above.


# `make_cross_sae_comparison_html` lives in explorer/html_views.py.


# ---------- UMAP data source ----------
# live_mask / live_indices / freq / mean_act / log_freq / umap_backup are all
# already set by state.apply(0) above — just build the source from them.
def _umap_source_data(xs, ys, feats, mask):
    """Assemble the UMAP ColumnDataSource dict. Single source of truth for
    every UMAP rebuild (initial, dataset switch, activation/dictionary
    toggle) so the column set can't drift between the call sites. ``feats``
    is coerced to a plain list because the tap/zoom handlers call
    ``.index()`` on it."""
    data = dict(
        x=xs, y=ys, feature_idx=list(feats),
        frequency=state.freq[mask].tolist(),
        log_freq=state.log_freq[mask].tolist(),
        mean_act=state.mean_act[mask].tolist(),
    )
    # Interpretability Index column for the color-by-II mode. Always present
    # (NaN when the dataset has no II) so the scatter's color field exists
    # regardless of which mode is active.
    if state.has_ii:
        data['ii'] = state.ii[mask].tolist()
    else:
        data['ii'] = [float('nan')] * int(np.count_nonzero(mask))
    return data


umap_source = ColumnDataSource(data=_umap_source_data(
    state.umap_coords[state.live_mask, 0],
    state.umap_coords[state.live_mask, 1],
    state.live_indices, state.live_mask,
))


# ---------- UMAP figure ----------
color_mapper = linear_cmap(
    field_name='log_freq', palette=Turbo256,
    low=0, high=float(np.nanmax(state.log_freq[state.live_mask])) if state.live_mask.any() else 1,
)


def _ii_range(mask):
    """(low, high) for the Interpretability-Index color scale over the
    finite II values of the currently visible features."""
    if not state.has_ii:
        return (0.0, 1.0)
    vals = state.ii[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    return (lo, hi if hi > lo else lo + 1e-6)


_ii_lo, _ii_hi = _ii_range(state.live_mask)
ii_color_mapper = linear_cmap(
    field_name='ii', palette=Turbo256, low=_ii_lo, high=_ii_hi,
)


def _refit_umap_color(mask):
    """Rescale both color maps to the currently visible features (the
    dataset-switch and type-toggle paths both call this so the palette
    never stays pinned to a stale dataset's range). Frequency uses
    [0, max log-freq]; the II scale uses [min, max] of its finite values."""
    color_mapper['transform'].high = (
        float(np.nanmax(state.log_freq[mask])) if mask.any() else 1.0)
    if state.has_ii:
        lo, hi = _ii_range(mask)
        ii_color_mapper['transform'].low = lo
        ii_color_mapper['transform'].high = hi

umap_fig = figure(
    title="UMAP of SAE Features (by activation pattern)",
    width=700, height=650,
    tools="pan,wheel_zoom,box_zoom,reset,tap",
    active_scroll="wheel_zoom",
)
umap_scatter = umap_fig.scatter(
    'x', 'y', source=umap_source, size=4, alpha=0.6,
    color=color_mapper,
    selection_color="red", selection_alpha=1.0, nonselection_alpha=0.3,
    selection_line_color="black", selection_line_width=2.5,
    selection_line_alpha=1.0,
    hit_dilation=2.5,
)

# Scale point size with zoom; the selected glyph stays distinctly bigger
# than the rest so it's easy to find a single red dot among thousands.
# The "fully zoomed out" reference span is derived from the scatter's own
# x-extent and cached, then invalidated whenever the data is replaced — so
# the sizing stays correct across dataset switches and the activation/
# dictionary toggle. (A previous version cached the span of the *first*
# dataset on `window` and never reset it, mis-sizing every point after a
# switch until the user manually zoomed.)
_zoom_cb = CustomJS(
    args=dict(renderer=umap_scatter, x_range=umap_fig.x_range, source=umap_source),
    code="""
    const span = x_range.end - x_range.start;
    if (!(span > 0)) return;
    let base = window._umap_base_span;
    if (base === undefined) {
        const xs = source.data['x'];
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < xs.length; i++) {
            const v = xs[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        base = (hi > lo) ? (hi - lo) : span;
        window._umap_base_span = base;
    }
    const zoom = base / span;
    const new_size = Math.min(16, Math.max(4, 4 * Math.pow(zoom, 0.4)));
    renderer.glyph.size = new_size;
    renderer.nonselection_glyph.size = new_size;
    renderer.selection_glyph.size = Math.max(new_size * 2.0, 12);
""")
umap_fig.x_range.js_on_change('start', _zoom_cb)
umap_fig.x_range.js_on_change('end', _zoom_cb)
# Drop the cached reference span when the scatter data is replaced; the next
# range event recomputes it against the new dataset's / view's extent.
umap_source.js_on_change('data', CustomJS(code="window._umap_base_span = undefined;"))

# ---------- Hover tooltip with MEI thumbnails ----------
# An invisible quad covers the entire UMAP. Its single source row carries
# the pre-rendered HTML for the 3 nearest features, recomputed by the
# Python MouseMove handler. HoverTool is bound *only* to this quad so
# exactly ONE tooltip ever fires — no stats-without-images bleed-through
# from other UMAP points, no chase-the-cursor hit-test lag.
UMAP_HOVER_MAX = 3

hover_target_source = ColumnDataSource(data=dict(html=['']))
hover_target_quad = umap_fig.quad(
    left=-1e9, right=1e9, top=1e9, bottom=-1e9,
    source=hover_target_source,
    fill_alpha=0.0, line_alpha=0.0,
    level="underlay",
)
# Exclude the quad from auto-range — only the scatter should drive the view.
umap_fig.x_range.renderers = [umap_scatter]
umap_fig.y_range.renderers = [umap_scatter]

umap_fig.add_tools(HoverTool(
    tooltips="@html{safe}",
    renderers=[hover_target_quad],
    point_policy="follow_mouse",
    attachment="right",
))

# The per-session hover-thumbnail cache lives on `ui.umap_mei_cache`;
# `ui.umap_last_hover_feats` holds the last rendered tuple.


def _feature_block_html(f):
    urls = ui.umap_mei_cache.get(f)
    if urls is None:
        urls = _mei_data_urls(ctx, f)
        ui.umap_mei_cache[f] = urls
    freq_v = int(state.freq[f])
    ma_v = float(state.mean_act[f])
    name = state.feature_names.get(f, "") or state.auto_interp_names.get(f, "")
    name_html = (f'<div style=\"font-size:11px;color:#1a6faf\">{name}</div>'
                 if name else '')
    imgs_html = ''.join(
        f'<img src=\"{u}\" width=\"{UMAP_THUMB_PX}\" height=\"{UMAP_THUMB_PX}\"'
        f' style=\"border:1px solid #ccc;border-radius:4px;display:block;flex-shrink:0\"/>'
        for u in urls
    ) or '<div style=\"font-size:10px;color:#999\">no MEIs</div>'
    return (
        f'<div style=\"display:flex;align-items:center;gap:8px;'
        f'padding:4px 6px;border:1px solid #e6e6e6;border-radius:5px;'
        f'background:#fafafa;margin-bottom:4px\">'
        f'<div style=\"flex:0 0 110px;font-size:11px;line-height:1.4\">'
        f'<div><b>feat {f}</b></div>{name_html}'
        f'<div style=\"color:#666\">freq={freq_v}</div>'
        f'<div style=\"color:#666\">mean act={ma_v:.3f}</div>'
        f'</div>'
        f'<div style=\"display:flex;gap:4px;flex-shrink:0\">{imgs_html}</div>'
        f'</div>'
    )


# Debounce window (ms) before the hover tooltip recomputes. While the cursor
# keeps moving the timer is re-armed and nothing rebuilds, so the preview no
# longer flickers through a dozen feature sets per sweep — it only settles once
# the cursor pauses.
UMAP_HOVER_DEBOUNCE_MS = 120


def _resolve_umap_hover():
    """Rebuild the hover-target HTML for the 3 features nearest the last cursor
    position. Runs after the cursor has paused for UMAP_HOVER_DEBOUNCE_MS."""
    ui.umap_hover_pending = None
    xy = ui.umap_hover_xy
    if xy is None:
        return
    cx, cy = xy

    xs = np.asarray(umap_source.data['x'], dtype=np.float32)
    ys = np.asarray(umap_source.data['y'], dtype=np.float32)
    if xs.size == 0:
        return

    feat_idx_arr = np.asarray(umap_source.data['feature_idx'], dtype=np.int64)
    dx = xs - cx
    dy = ys - cy
    d2 = dx * dx + dy * dy

    rng_x = (umap_fig.x_range.end or 0) - (umap_fig.x_range.start or 0)
    rng_y = (umap_fig.y_range.end or 0) - (umap_fig.y_range.start or 0)
    plot_span = max(abs(rng_x), abs(rng_y), 1e-6)
    radius2 = (0.05 * plot_span) ** 2

    in_range = np.where(d2 < radius2)[0]
    nearest = in_range[np.argsort(d2[in_range])][:UMAP_HOVER_MAX]
    feats = tuple(int(feat_idx_arr[i]) for i in nearest)

    if not feats:
        if hover_target_source.data['html'][0]:
            hover_target_source.data = dict(html=[''])
            ui.umap_last_hover_feats = ()
        return

    if feats == ui.umap_last_hover_feats:
        return

    html = '<div style=\"font-family:sans-serif\">' + ''.join(
        _feature_block_html(f) for f in feats
    ) + '</div>'
    ui.umap_last_hover_feats = feats
    hover_target_source.data = dict(html=[html])


def _on_umap_mousemove(event):
    """Record the cursor position and (re)arm the debounce timer. The actual
    nearest-feature compute happens in _resolve_umap_hover once the cursor
    stops moving."""
    if event.x is None or event.y is None:
        return
    ui.umap_hover_xy = (float(event.x), float(event.y))

    # Re-arm: drop the previous pending resolve so only a settled cursor fires.
    if ui.umap_hover_pending is not None:
        try:
            curdoc().remove_timeout_callback(ui.umap_hover_pending)
        except Exception:
            pass
    ui.umap_hover_pending = curdoc().add_timeout_callback(
        _resolve_umap_hover, UMAP_HOVER_DEBOUNCE_MS,
    )


umap_fig.on_event(MouseMove, _on_umap_mousemove)


# ---------- Dataset / model selector ----------
dataset_select = Select(
    title="Active model:",
    value="0",
    options=[(str(i), ds['label']) for i, ds in enumerate(_all_datasets)],
    width=250,
)


def _on_dataset_switch(attr, old, new):
    idx = int(new)
    # Capture the previously selected feature so we can re-select it in the
    # new dataset if the index is still in range. Same SAE feature index is
    # not semantically the same feature across SAEs, but for users flipping
    # back and forth between two models on a fixed index this is exactly
    # the comparison flow they want.
    try:
        prev_feat = int(feature_input.value) if feature_input.value else None
    except ValueError:
        prev_feat = None

    _ensure_loaded(idx)
    state.apply(idx)

    # Rebuild UMAP scatter
    umap_source.data = _umap_source_data(
        state.umap_coords[state.live_mask, 0],
        state.umap_coords[state.live_mask, 1],
        state.live_indices, state.live_mask,
    )
    _refit_umap_color(state.live_mask)
    umap_source.selected.indices = []
    ui.umap_mei_cache.clear()
    ui.umap_last_hover_feats = ()
    ui.umap_hover_xy = None
    if ui.umap_hover_pending is not None:
        try:
            curdoc().remove_timeout_callback(ui.umap_hover_pending)
        except Exception:
            pass
        ui.umap_hover_pending = None
    hover_target_source.data = dict(html=[''])
    # Reset to the activation view for the new dataset.
    umap_type_select.value = "Activation Pattern"
    # Reset the color mode (and its availability) for the new dataset.
    umap_color_select.visible = state.has_ii
    umap_color_select.value = "Frequency"
    _apply_umap_color_mode("Frequency")
    umap_fig.title.text = f"UMAP — {_all_datasets[idx]['label']}"

    # Rebuild feature list (the panel owns its own reset: clears the search
    # filter and re-applies the freq-sorted order for the new dataset).
    _feature_list['on_dataset_changed'](idx)

    # Rebuild active-feature pool for random button
    ui.active_feats = [int(i) for i in range(state.d_model)
                       if state.feature_frequency[i].item() > 0]

    # Kick off hover-thumbnail prewarm for the new dataset (cancels any
    # in-flight prewarm for the previous dataset).
    _prewarm_hover_cache_async(ctx)

    # Update summary panel
    _summary['on_dataset_changed'](idx)

    # Show/hide patch explorer depending on token type and data availability.
    ds = _all_datasets[idx]
    has_heatmaps   = ds.get('top_heatmaps') is not None
    has_patch_acts = ds.get('patch_acts') is not None
    can_explore = (
        ds.get('token_type', 'spatial') == 'spatial'
        and (has_heatmaps or has_patch_acts)
    )
    # Rebuild the patch grid to match the new dataset's patch_grid (it may
    # differ from the primary dataset, e.g. 14×14 vs 16×16). Without this
    # the figure stays sized for the original grid and the click→patch
    # math goes wrong.
    _rebuild_patch_grid(state.patch_grid)
    patch_fig.visible = can_explore
    patch_info_div.visible = can_explore
    if not can_explore:
        if ds.get('token_type') == 'cls':
            reason = "CLS token — no patch grid"
        else:
            reason = "no pre-computed heatmaps or patch_acts for this model"
        patch_info_div.text = (
            f'<p style="color:#888;font-style:italic">Patch explorer unavailable: {reason}.</p>')
        patch_info_div.visible = True

    # Refresh the view-ranking options to drop Mean / Crop when the new
    # dataset has no corresponding heatmap tensor. If the current selection
    # is no longer offered, fall back to Top.
    new_view_opts = _available_view_options()
    view_select.options = new_view_opts
    if view_select.value not in new_view_opts:
        view_select.value = VIEW_TOP

    # Show/hide CLIP widgets and clear stale results for the new dataset.
    _clip_search['on_dataset_changed'](idx)

    # Re-select the previously focused feature when the index is still
    # in range; otherwise clear the detail panels.
    if prev_feat is not None and 0 <= prev_feat < state.d_model:
        _select_feature(prev_feat)
    else:
        feature_input.value = ""
        stats_div.text = "<h3>Select a feature to explore</h3>"
        for div in [top_heatmap_div, mean_heatmap_div, crop_heatmap_div]:
            div.text = ""


dataset_select.on_change('value', _on_dataset_switch)


# ---------- Detail panels ----------
# The detail panels stretch to fill the (responsive) middle column so the
# image grids inside them reflow with the window width. A min_width keeps a
# sane floor; the HTML caps thumbnail size, so a wide screen just packs more
# thumbnails per row rather than enlarging them.
stats_div = Div(text="<h3>Click a feature on the UMAP to explore it</h3>",
                sizing_mode="stretch_width")
top_heatmap_div = Div(text="", sizing_mode="stretch_width")
mean_heatmap_div = Div(text="", sizing_mode="stretch_width")
crop_heatmap_div = Div(text="", sizing_mode="stretch_width")
compare_agg_div = Div(text="", sizing_mode="stretch_width")   # side-by-side aggregation comparison

# Name editing widget (defined here so update_feature_display can reference it)
name_input = TextInput(
    title="Feature name:",
    placeholder="Enter a name for this feature...",
    width=420,
)

# Hidden author tag attached to every label saved during this session.
# Defaults to a per-session anonymous handle; the user can overwrite it via
# `identity_input` to attribute their work. The author lives in sibling
# *_authors.json files alongside the labels JSONs and is not surfaced in the
# main UI — pulling them is a separate operator action.
_DEFAULT_IDENTITY = f"anon-{uuid.uuid4().hex[:12]}"

# Bold the title of the identity input. Bokeh's TextInput.title is plain
# text; CSS via an InlineStyleSheet is the cleanest way to style just the
# label inside the widget's shadow DOM.
_BOLD_TITLE_CSS = InlineStyleSheet(css="""
:host label { font-weight: bold; }
""")

identity_input = TextInput(
    title="Your Name (attached to new manual labels):",
    placeholder=_DEFAULT_IDENTITY,
    value=_DEFAULT_IDENTITY,
    width=260,
    stylesheets=[_BOLD_TITLE_CSS],
)


def _current_manual_author() -> str:
    """Resolve the author to attribute to the next manual label save.

    Falls back to the per-session default if the user has cleared the input.
    """
    val = (identity_input.value or "").strip()
    return val or _DEFAULT_IDENTITY


# Auto-interp labels always get this fixed attribution — the model is the
# author of the text, regardless of which operator triggered the call.
_AUTO_INTERP_AUTHOR = "gemini-2.5-flash"


# Gemini auto-interp button — widgets + worker thread + click handler
# all live in explorer/gemini.py (imported at the top). The bootstrap
# supplies the post-label hook that mutates state (auto_interp_names /
# authors / history) and refreshes the table + detail panel.
_gemini_api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")


def _on_gemini_label_applied(feat, label):
    """Doc-thread callback fired by gemini.py after a successful API call.

    Mutates state, persists JSON, refreshes downstream UI. The Gemini
    module re-enables its button + sets its status div separately.
    """
    old_label = state.auto_interp_names.get(feat)
    if old_label is not None and old_label != label:
        _archive_label(state.auto_interp_history, feat, old_label,
                       state.auto_interp_authors.get(feat))
    state.auto_interp_names[feat] = label
    state.auto_interp_authors[feat] = _AUTO_INTERP_AUTHOR
    _save_auto_interp()
    _update_table_names()
    try:
        update_feature_display(feat)
    except Exception:
        pass


def _current_selected_feat():
    """Late-bound: returns the current ``feature_input.value`` as an int,
    or None if not parseable / not yet constructed. ``feature_input`` is
    a module-level global created later in this file; Python late-binds
    free variables in closures, so this works at click time."""
    try:
        return int(feature_input.value)
    except (ValueError, AttributeError, NameError):
        return None


gemini_btn, gemini_status_div, _on_gemini_click = _gemini.build(
    ctx,
    get_current_feat=_current_selected_feat,
    on_label_applied=_on_gemini_label_applied,
    api_key=_gemini_api_key,
)


# ---------- Export classifier button ----------
# Generates a standalone Python "classifier" for the selected feature (the
# classifier_export module bakes in the backbone/SAE/feature so the script
# downloads the model + SAE + a toy image and prints the feature's score).
# Bokeh server callbacks can't push a file, so the Python on_click stuffs the
# generated text into a hidden ColumnDataSource whose `data` change fires a
# CustomJS Blob download in the browser.
export_download_source = ColumnDataSource(data=dict(text=[], filename=[], nonce=[]))
export_download_source.js_on_change('data', CustomJS(
    args=dict(source=export_download_source),
    code="""
        const text = source.data['text'][0];
        const fn   = source.data['filename'][0];
        if (text == null || text === '') { return; }
        const blob = new Blob([text], {type: 'text/x-python'});
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = fn;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
    """,
))
export_btn = Button(label="Export classifier (.py)", width=180, button_type="success")
export_status_div = Div(text="", width=300)
_export_nonce = [0]


def _on_export_click():
    feat = _current_selected_feat()
    if feat is None:
        export_status_div.text = "<span style='color:#c00'>Select a feature first.</span>"
        return
    if int(state.freq[feat]) == 0:
        export_status_div.text = ("<span style='color:#c00'>Dead feature — "
                                  "nothing to classify.</span>")
        return
    if not state.ds.get('sae_file'):
        export_status_div.text = ("<span style='color:#c00'>No SAE checkpoint "
                                  "registered for this model.</span>")
        return
    try:
        top_idx = int(state.top_img_idx[feat, 0])
        top_image = (os.path.basename(state.image_paths[top_idx])
                     if top_idx >= 0 else "")
        threshold = 0.5 * float(state.top_img_act[feat, 0])
        script, fn = _classifier_export.build_classifier_script(
            feat, state.ds, _registry.hf_data_repo, threshold,
            top_image_basename=top_image,
            hf_images_repo=_registry.hf_images_repo,
            images_tarball=_registry.images_tarball,
        )
    except Exception as e:
        export_status_div.text = (f"<span style='color:#c00'>Export failed: "
                                  f"{str(e)[:120]}</span>")
        return
    _export_nonce[0] += 1
    export_download_source.data = dict(
        text=[script], filename=[fn], nonce=[str(_export_nonce[0])])
    export_status_div.text = ""


export_btn.on_click(_on_export_click)

# Zoom slider — crops the main MEI around the peak patch.
# Defaults to the full grid (no crop = full MEI).
_zoom_max = max(2, int(state.heatmap_patch_grid) if state.heatmap_patch_grid else 16)
zoom_slider = Slider(
    title="MEI zoom window",
    value=_zoom_max, start=1, end=_zoom_max, step=1,
    sizing_mode="stretch_width",
)

# Heatmap opacity slider — controls alpha of the overlay in render_heatmap_overlay
heatmap_alpha_slider = Slider(
    title="Heatmap opacity", value=0.5, start=0.0, end=1.0, step=0.05,
    sizing_mode="stretch_width",
)

# View selector: which image ranking to show in the detail panel.
# Mean / crop options are dropped when their corresponding heatmap tensor is
# missing — without it those views show the same plain images as Top, which
# is misleading.
VIEW_TOP     = "Top (max activation)"
VIEW_MEAN    = "Mean activation"
VIEW_CROP    = "Crop avg (top-8 patches)"
VIEW_COMPARE = "Compare aggregations"

# Short, plain-language description of what each ranking sorts images by,
# shown as a footnote below the detail panel. Makes explicit which per-image
# score drives the ordering.
VIEW_SUBTITLES = {
    VIEW_TOP:  "Sorted by each image's single most-activating patch (max).",
    VIEW_MEAN: "Sorted by each image's mean activation over all patches.",
    VIEW_CROP: "Sorted by the mean of each image's top-8 positive patches "
               "(crop &amp; average).",
}


def _available_view_options():
    """Return the list of view options valid for the active dataset."""
    opts = [VIEW_TOP]
    if state.mean_heatmaps is not None:
        opts.append(VIEW_MEAN)
    if state.crop_heatmaps is not None:
        opts.append(VIEW_CROP)
    opts.append(VIEW_COMPARE)
    return opts


view_select = Select(
    title="Image ranking:",
    value=VIEW_TOP,
    options=_available_view_options(),
    sizing_mode="stretch_width",
)

N_DISPLAY = 9

# Below this patch-activation count a feature fires too rarely for its top
# examples to be representative — we surface a small caution note instead of
# the raw stats. Tunable heuristic.
LOW_ACT_FREQ = 1000


def update_feature_display(feature_idx):
    feat = int(feature_idx)
    ui.render_token += 1
    my_token = ui.render_token
    gemini_status_div.text = ""

    freq_val = state.feature_frequency[feat].item()
    dead = "DEAD FEATURE" if freq_val == 0 else ""

    feat_name = state.feature_names.get(feat, "")
    auto_name = state.auto_interp_names.get(feat, "")
    name_parts = []
    if feat_name:
        name_parts.append(
            f'<div style="color:#1a6faf;font-style:italic;margin:2px 0 3px 0">'
            f'&#x1F3F7;&#xFE0E; {feat_name}'
            f'<span style="font-size:10px;color:#999;margin-left:6px">(manual)</span></div>'
        )
    if auto_name:
        name_parts.append(
            f'<div style="color:#5a9a5a;font-style:italic;margin:2px 0 3px 0">'
            f'&#x1F916; {auto_name}'
            f'<span style="font-size:10px;color:#999;margin-left:6px">(auto-interp)</span></div>'
        )
    name_display = "".join(name_parts)

    # Caution note for rarely-activating (but not dead) features — their top
    # examples may be noisy / unrepresentative.
    low_act_note = ""
    if 0 < freq_val < LOW_ACT_FREQ:
        low_act_note = (
            '<div style="background:#fff8e0;border-left:4px solid #f0a020;'
            'color:#7a5000;padding:6px 10px;border-radius:3px;font-size:12px;'
            'margin:4px 0">&#x26A0;&#xFE0E; Low-activation feature — fires on '
            'few patches, so the examples below may be noisy or '
            'unrepresentative.</div>'
        )

    stats_div.text = f"""
    <h2 style="margin:4px 0">Feature {feat} <span style="color:red">{dead}</span></h2>
    {name_display}
    {low_act_note}
    """
    name_input.value = feat_name

    if freq_val == 0:
        for div in [top_heatmap_div, mean_heatmap_div, crop_heatmap_div,
                    compare_agg_div]:
            div.text = ""
        return

    # Snapshot slider values once on the doc thread; the worker threads must
    # not touch Bokeh widgets (Bokeh state isn't thread-safe).
    alpha_v = heatmap_alpha_slider.value
    zoom_v  = int(zoom_slider.value)

    def _render_one(img_idx_tensor, act_tensor, ranking_idx, heatmap_tensor=None):
        """Render one (img, caption) pair. Pure function — safe to call from
        any thread because it doesn't touch Bokeh widgets or `state` mutators.
        Returns None if the slot has no image (img_i < 0)."""
        img_i = img_idx_tensor[feat, ranking_idx].item()
        if img_i < 0:
            return None
        try:
            # The activation value is the most informative number in the
            # caption, so keep it on both the plain and the overlay variant.
            act_val = float(act_tensor[feat, ranking_idx].item())
            caption = f"act={act_val:.4f}  img {img_i}"
            if heatmap_tensor is not None and state.heatmap_patch_grid > 1:
                hmap = heatmap_tensor[feat, ranking_idx].float().numpy()
                hmap = hmap.reshape(state.heatmap_patch_grid, state.heatmap_patch_grid)
            else:
                hmap = None

            if hmap is None:
                plain = ctx.load_image(img_i).resize((THUMB, THUMB), Image.BILINEAR)
                return (plain, caption)
            overlay = render_heatmap_overlay(
                ctx, img_i, hmap, size=THUMB, alpha=alpha_v, zoom_patches=zoom_v,
            )
            return (overlay, caption)
        except Exception as e:
            ph = Image.new("RGB", (THUMB, THUMB), "gray")
            return (ph, f"Error: {e}")

    def _submit_ranking(idx_t, act_t, hm_t, *, skip_zero_act=False):
        """Submit up to N_DISPLAY render jobs to the shared executor without
        waiting. Returns the list of futures in display order."""
        futures = []
        for j in range(min(N_DISPLAY, idx_t.shape[1])):
            if idx_t[feat, j].item() < 0:
                # No more valid slots for this feature.
                break
            if skip_zero_act and act_t[feat, j].item() == 0:
                continue
            futures.append(_render_executor.submit(_render_one, idx_t, act_t, j, hm_t))
        return futures

    def _await(futures):
        """Collect futures, dropping early-out None results."""
        return [r for r in (f.result() for f in futures) if r is not None]

    def _render():
        # Bail out if the user has already clicked a different feature.
        if ui.render_token != my_token:
            return

        # Submit all 27 (3 × N_DISPLAY) jobs first so the executor can run
        # them in parallel; only then await. Wall time goes from 27×t to
        # roughly ceil(27/workers)×t.
        top_futs  = _submit_ranking(state.top_img_idx,  state.top_img_act,  state.top_heatmaps)
        mean_futs = _submit_ranking(state.mean_img_idx, state.mean_img_act, state.mean_heatmaps)
        crop_futs = _submit_ranking(state.crop_img_idx, state.crop_img_act, state.crop_heatmaps,
                                    skip_zero_act=True)

        heatmap_infos = _await(top_futs)
        mean_hm_infos = _await(mean_futs)
        crop_hm_infos = _await(crop_futs)

        # User clicked a different feature while we were rendering — drop
        # the stale results.
        if ui.render_token != my_token:
            return

        model_lbl = state.ds['label']
        top_heatmap_div.text = make_image_grid_html(
            heatmap_infos, "Top Activation", "#2563a8",
            feat=feat, model_label=model_lbl, subtitle=VIEW_SUBTITLES[VIEW_TOP])
        mean_heatmap_div.text = make_image_grid_html(
            mean_hm_infos, "Mean Activation", "#1a7a4a",
            feat=feat, model_label=model_lbl, subtitle=VIEW_SUBTITLES[VIEW_MEAN])
        crop_heatmap_div.text = make_image_grid_html(
            crop_hm_infos, "Crop Avg (top-8 patches)", "#c2691a",
            feat=feat, model_label=model_lbl, subtitle=VIEW_SUBTITLES[VIEW_CROP])

        # Side-by-side aggregation comparison (paper-ready screenshot view)
        compare_agg_div.text = make_compare_aggregations_html(
            heatmap_infos, mean_hm_infos, feat,
            model_label=state.ds['label'])

        _update_view_visibility()

        # Auto-Gemini: when --auto-gemini is passed and the user has
        # provided a Google API key, fire an auto-interp call for any
        # selected feature that has neither a manual name nor an existing
        # auto-interp label. Off by default — enable per session with
        # `--auto-gemini` on the bokeh-serve command line. Skipped while
        # another Gemini call is in flight (button disabled).
        if (args.auto_gemini
                and _gemini_api_key
                and not gemini_btn.disabled
                and feat not in state.auto_interp_names
                and feat not in state.feature_names):
            _on_gemini_click()

    curdoc().add_next_tick_callback(_render)


# ---------- View visibility ----------
def _update_view_visibility():
    v = view_select.value
    top_heatmap_div.visible  = (v == VIEW_TOP)
    mean_heatmap_div.visible = (v == VIEW_MEAN)
    crop_heatmap_div.visible = (v == VIEW_CROP)
    compare_agg_div.visible  = (v == VIEW_COMPARE)

view_select.on_change('value', lambda attr, old, new: _update_view_visibility())
_update_view_visibility()  # set initial state


def _rerender_current_feature(attr, old, new):
    """Re-render the current feature on slider release.

    Bound to `value_throttled` (not `value`) so dragging the zoom or opacity
    slider only triggers a single re-render of all 27 heatmaps when the user
    releases the handle, instead of one render per drag tick.
    """
    try:
        feat = int(feature_input.value)
    except ValueError:
        return
    if 0 <= feat < state.d_model:
        update_feature_display(feat)

zoom_slider.on_change('value_throttled', _rerender_current_feature)
heatmap_alpha_slider.on_change('value_throttled', _rerender_current_feature)


# ---------- Callbacks ----------
def _zoom_umap_to_feature(feat: int) -> None:
    """Recenter (and zoom in, if currently zoomed out) the UMAP figure on
    the selected feature so the highlighted red dot is easy to find. If
    the user is already zoomed in past the target span and the feature
    is on-screen, the view is left alone — no jarring jumps."""
    feat_list = umap_source.data['feature_idx']
    if feat not in feat_list:
        return
    idx = feat_list.index(feat)
    fx = float(umap_source.data['x'][idx])
    fy = float(umap_source.data['y'][idx])
    if not (np.isfinite(fx) and np.isfinite(fy)):
        return
    xs = np.asarray(umap_source.data['x'], dtype=float)
    ys = np.asarray(umap_source.data['y'], dtype=float)
    x_extent = float(np.nanmax(xs) - np.nanmin(xs))
    y_extent = float(np.nanmax(ys) - np.nanmin(ys))
    if x_extent <= 0 or y_extent <= 0:
        return
    target_x = x_extent * 0.3
    target_y = y_extent * 0.3
    # The figure's ranges are None until the first auto-fit render. If a
    # feature is selected before then (e.g. from a panel callback), there's
    # nothing to recenter against — the initial fit will frame it anyway.
    if None in (umap_fig.x_range.start, umap_fig.x_range.end,
                umap_fig.y_range.start, umap_fig.y_range.end):
        return
    cur_x = umap_fig.x_range.end - umap_fig.x_range.start
    cur_y = umap_fig.y_range.end - umap_fig.y_range.start
    new_x = min(cur_x, target_x)
    new_y = min(cur_y, target_y)
    in_view = (umap_fig.x_range.start <= fx <= umap_fig.x_range.end
               and umap_fig.y_range.start <= fy <= umap_fig.y_range.end)
    if new_x < cur_x - 1e-9 or new_y < cur_y - 1e-9 or not in_view:
        umap_fig.x_range.start = fx - new_x / 2
        umap_fig.x_range.end   = fx + new_x / 2
        umap_fig.y_range.start = fy - new_y / 2
        umap_fig.y_range.end   = fy + new_y / 2


# Direct feature input. Defined here (ahead of the callbacks that read it)
# because `_select_feature` is wired onto `ctx` below and can be invoked by
# panels (e.g. CLIP search) before the rest of the controls block is
# constructed.
feature_input = TextInput(title="Feature Index:", value="", width=120)


def _select_feature(feat):
    """Common entry point for selecting a feature from any UI surface:
    sets the input box, renders the detail panels, syncs the UMAP highlight,
    and points the Cross-SAE Compare 'SAE A' inputs at the current
    feature/dataset. Safe to call from a UMAP-driven callback — re-selecting
    the same index in the source is a no-op."""
    feat = int(feat)
    feature_input.value = str(feat)
    update_feature_display(feat)
    feat_list = umap_source.data['feature_idx']
    if feat in feat_list:
        umap_source.selected.indices = [feat_list.index(feat)]
    _zoom_umap_to_feature(feat)
    # Auto-fill the Cross-SAE Compare 'A' side. SAE B stays sticky so the
    # user can keep comparing against a fixed reference dataset.
    cmp_feat_a.value = str(feat)
    cmp_ds_a.value = str(state.idx)


# Wire the selection hub onto the context so panels (e.g. clip_search,
# patch_explorer) can drive selection without importing the bootstrap.
ctx.select_feature = _select_feature


# A tap on overlapping glyphs adds every stacked feature to
# ``selected.indices``. Doing the narrowing in Python causes a visible
# flash of all hits rendering red before the callback round-trip
# completes; a browser-side JS callback trims to the topmost glyph
# (last in source order = drawn on top) in the same tick the tap
# fires, so the multi-selection state never reaches the canvas.
umap_source.selected.js_on_change('indices', CustomJS(
    args=dict(source=umap_source),
    code="""
        const sel = source.selected.indices;
        if (sel.length > 1) {
            source.selected.indices = [sel[sel.length - 1]];
        }
    """,
))


def on_umap_select(attr, old, new):
    if not new:
        return
    if len(new) > 1:
        # Wait for the JS callback above to trim to a single index; this
        # Python handler will re-fire with the trimmed list.
        return
    feature_idx = int(umap_source.data['feature_idx'][new[0]])
    # If this index is already the displayed feature, the selection was set
    # programmatically (by `_select_feature`, which already rendered, or by
    # the UMAP-type toggle re-resolving the highlight). The red glyph repaints
    # from `selected.indices` on its own; skip the redundant detail re-render.
    try:
        if int(feature_input.value) == feature_idx:
            return
    except (ValueError, TypeError):
        pass
    feature_input.value = str(feature_idx)
    update_feature_display(feature_idx)
    _zoom_umap_to_feature(feature_idx)

umap_source.selected.on_change('indices', on_umap_select)


# UMAP color mode: frequency (default) or Interpretability Index. Hidden when
# the active dataset has no precomputed II.
umap_color_select = Select(
    title="Color by", value="Frequency",
    options=["Frequency", "Interpretability"], width=160,
)
umap_color_select.visible = state.has_ii


def _apply_umap_color_mode(mode: str) -> None:
    mapper = ii_color_mapper if mode == "Interpretability" else color_mapper
    umap_scatter.glyph.fill_color = mapper
    umap_scatter.glyph.line_color = mapper


def on_umap_color_change(attr, old, new):
    _apply_umap_color_mode(new)

umap_color_select.on_change('value', on_umap_color_change)


# UMAP type toggle
umap_type_select = Select(
    title="UMAP Type", value="Activation Pattern",
    options=["Activation Pattern", "Dictionary Geometry"], width=200,
)

def on_umap_type_change(attr, old, new):
    ui.umap_mei_cache.clear()
    ui.umap_last_hover_feats = ()
    hover_target_source.data = dict(html=[''])

    if new == "Activation Pattern":
        mask = state.live_mask
        umap_source.data = _umap_source_data(
            state.umap_backup['act_x'], state.umap_backup['act_y'],
            state.umap_backup['act_feat'], mask,
        )
        umap_fig.title.text = "UMAP of SAE Features (by activation pattern)"
    else:
        mask = state.dict_live_mask
        umap_source.data = _umap_source_data(
            state.umap_backup['dict_x'], state.umap_backup['dict_y'],
            state.umap_backup['dict_feat'], mask,
        )
        umap_fig.title.text = "UMAP of SAE Features (by dictionary geometry)"
    _refit_umap_color(mask)

    # Re-resolve the highlighted feature into the new ordering. The two
    # embeddings have different feature orderings (and lengths), so the row
    # index carried over from the previous view would point at an unrelated
    # feature — the red highlight has to be looked up by feature id, not row.
    feat_list = umap_source.data['feature_idx']
    try:
        cur_feat = int(feature_input.value)
    except (ValueError, TypeError):
        cur_feat = None
    umap_source.selected.indices = (
        [feat_list.index(cur_feat)] if cur_feat in feat_list else [])

umap_type_select.on_change('value', on_umap_type_change)


# Direct feature input (`feature_input` is defined earlier, ahead of
# `_select_feature`).
go_button = Button(label="Go", width=60)
random_btn = Button(label="Random", width=130)
random_unlabeled_btn = Button(label="Random Unlabeled", width=130)

def on_go_click():
    try:
        feat = int(feature_input.value)
    except ValueError:
        stats_div.text = "<h3>Please enter a valid integer</h3>"
        return
    if not (0 <= feat < state.d_model):
        stats_div.text = f"<h3>Feature {feat} out of range (0-{state.d_model-1})</h3>"
        return
    _select_feature(feat)

go_button.on_click(on_go_click)

ui.active_feats = [int(i) for i in range(state.d_model)
                   if state.feature_frequency[i].item() > 0]

# Kick off the initial hover-thumbnail prewarm in a background thread.
_prewarm_hover_cache_async(ctx)


def _on_random():
    if not ui.active_feats:
        return
    _select_feature(random.choice(ui.active_feats))

random_btn.on_click(_on_random)


def _on_random_unlabeled():
    """Pick uniformly from active features that have neither a manual name
    nor a Gemini auto-interp label — useful for chewing through the
    review backlog."""
    pool = [f for f in ui.active_feats
            if f not in state.feature_names and f not in state.auto_interp_names]
    if not pool:
        stats_div.text = "<h3>All active features are already labeled.</h3>"
        return
    _select_feature(random.choice(pool))

random_unlabeled_btn.on_click(_on_random_unlabeled)


# ---------- Sorted feature list + name search ----------
# Widgets, callbacks, and the dataset-switch / update_table_names hooks all
# live in explorer/panels/feature_list.py. The bootstrap holds onto
# `update_table_names` because on_name_change / Gemini's label hook need to
# refresh the table's name column after a label mutation; `on_dataset_changed`
# resets the table on a model switch.
_feature_list = _feature_list_mod.build(ctx, display_name=_display_name)
_update_table_names = _feature_list['update_table_names']


# ---------- Auto-save name on typing ----------
def on_name_change(attr, old, new):
    try:
        feat = int(feature_input.value)
    except ValueError:
        return
    name = new.strip()
    # Selecting a feature programmatically rewrites name_input.value, which
    # fires this handler. Bail if the field already matches what we have on
    # disk for this feature — otherwise every click would write JSON and
    # schedule an HF push.
    if name == state.feature_names.get(feat, ""):
        return
    # Archive any existing entry before overwriting/deleting so the history
    # file retains every label this feature has ever carried.
    old_label = state.feature_names.get(feat)
    if old_label is not None:
        _archive_label(state.feature_names_history,
                       feat, old_label,
                       state.feature_name_authors.get(feat))
    if name:
        state.feature_names[feat] = name
        state.feature_name_authors[feat] = _current_manual_author()
    elif feat in state.feature_names:
        del state.feature_names[feat]
        state.feature_name_authors.pop(feat, None)
    _save_names()
    _update_table_names()

name_input.on_change('value', on_name_change)


# Gemini auto-interp lives entirely in explorer/gemini.py. Its widgets +
# click handler are constructed up near the rest of the detail-panel
# widgets via `_gemini.build(...)`.


# Name-search widgets live in explorer/panels/feature_list.py (built above).


# Summary panel lives in explorer/panels/summary.py.
_summary = _summary_mod.build(ctx)
summary_div = _summary['summary_div']


# ---------- Patch Explorer ----------
# Widgets + handlers + paint-on-drag CustomJS all live in
# explorer/panels/patch_explorer.py. The bootstrap holds onto the
# refresh hook (``rebuild_grid``) so the dataset-switch path can resize
# the patch grid when the new dataset has a different patch_grid.
_patch_explorer = _patch_explorer_mod.build(ctx)
patch_fig          = _patch_explorer['patch_fig']
patch_info_div     = _patch_explorer['patch_info_div']
patch_feat_table   = _patch_explorer['patch_feat_table']
_rebuild_patch_grid = _patch_explorer['rebuild_grid']


# ---------- CLIP Text Search ----------
# Widgets + handlers live in explorer/panels/clip_search.py. The panel owns
# its own reset (visibility + clearing stale results) via on_dataset_changed,
# which the dataset-switch path calls; the bootstrap only needs the layout.
_clip_search = _clip_search_mod.build(ctx, display_name=_display_name)
clip_search_panel    = _clip_search['layout']


# ---------- Layout ----------
# Random / Random Unlabeled stack vertically to the right of Go so the wide
# "Random Unlabeled" button doesn't stick out past the controls row.
controls = row(umap_type_select, umap_color_select,
               feature_input, go_button,
               column(random_btn, random_unlabeled_btn))

name_panel = column(
    name_input,
    row(gemini_btn, export_btn),
    row(gemini_status_div, export_status_div),
)

feature_list_panel = _feature_list['layout']


def _make_collapsible(title, body, initially_open=False):
    """Wrap a widget in a toggle-able collapsible section."""
    btn = Toggle(
        label=("▼  " if initially_open else "▶  ") + title,
        active=initially_open,
        button_type="light",
        width=500,
        height=30,
    )
    body.visible = initially_open
    btn.js_on_click(CustomJS(args=dict(body=body, btn=btn, title=title), code="""
        body.visible = btn.active;
        btn.label = (btn.active ? '▼  ' : '▶  ') + title;
    """))
    return column(btn, body)


summary_section = _make_collapsible("SAE Summary",      summary_div)
patch_section   = _make_collapsible("Patch Explorer",   _patch_explorer['layout'])
clip_section    = _make_collapsible("CLIP Text Search", clip_search_panel)

# Fixed width on the side columns so the stretch_width middle column always
# has its space reserved. Without this, expanding a right-panel collapsible
# (e.g. SAE Summary, whose body is 700px) grows the auto-width right column
# into the middle column and the two overlap.
left_panel = column(
    row(dataset_select, identity_input),
    controls,
    umap_fig,
    feature_list_panel,
    width=720,
)

# Cap the controls row at the image-card width and let the three widgets share
# that width, so it lines up with the SAE images below it.
feature_controls_row = row(
    view_select, zoom_slider, heatmap_alpha_slider,
    sizing_mode="stretch_width", max_width=image_card_width())

middle_panel = column(
    stats_div,
    name_panel,
    feature_controls_row,
    compare_agg_div,
    top_heatmap_div,
    mean_heatmap_div,
    crop_heatmap_div,
    sizing_mode="stretch_width",
    min_width=560,
)

# --- Cross-SAE comparison section ---
# Widgets + handlers live in explorer/panels/cross_sae.py. The bootstrap
# only needs `cmp_ds_a` and `cmp_feat_a` — `_select_feature` auto-fills
# them when a feature is selected.
_cross_sae = _cross_sae_mod.build(ctx, datasets=_all_datasets,
                                  ensure_loaded=_ensure_loaded)
cmp_ds_a   = _cross_sae['cmp_ds_a']
cmp_feat_a = _cross_sae['cmp_feat_a']
cmp_section = _make_collapsible("Cross-SAE Comparison", _cross_sae['layout'])

right_panel = column(summary_section, patch_section, clip_section,
                     cmp_section, width=720)

# Outer row stretches to the viewport; left + right keep their natural
# widths while the middle column absorbs the remaining space and reflows.
layout = row(left_panel, middle_panel, right_panel, sizing_mode="stretch_width")
curdoc().add_root(layout)
# The export-classifier download source isn't referenced by any glyph, so add
# it as a standalone root to ensure it's part of the session document and its
# `js_on_change('data', ...)` Blob-download fires when the click handler sets it.
curdoc().add_root(export_download_source)
curdoc().title = "SAE Feature Explorer"

print("Explorer app ready!")
