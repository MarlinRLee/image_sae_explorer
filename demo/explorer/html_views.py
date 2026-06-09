"""Pure HTML-emitting helpers used by the various Bokeh ``Div`` widgets.

No Bokeh, no ``state``, no I/O — just string formatting plus image
encoding (via :mod:`explorer.images`). Tightly state-coupled HTML helpers
(``_make_summary_html``, ``_feature_block_html``) live alongside the
things they read in the panel modules / the bootstrap.
"""

from . import images as _images
# `_render_overlay_from_ds` is imported lazily inside the cross-SAE function
# to keep the rendering import deferred, so this module stays importable
# from anywhere without pulling cv2/torch onto the path at import time.


# ---- Image-grid card geometry (shared so callers can size siblings) ----
_GRID_COLS = 3
_GRID_GAP = 10     # px between thumbnails
_CARD_PAD_X = 20   # px left/right padding inside the white card


def image_card_width() -> int:
    """Pixel width of the image-grid card produced by
    :func:`make_image_grid_html` (banner + 3-column grid).

    The bootstrap uses this to size the controls row (image ranking / zoom
    / opacity) to the same length as the images below it.
    """
    thumb_w = min(_images.THUMB, 224)
    return _GRID_COLS * thumb_w + (_GRID_COLS - 1) * _GRID_GAP + 2 * _CARD_PAD_X


def make_image_grid_html(images_info, method_name: str, color: str, *,
                         feat: int = None, model_label: str = None,
                         subtitle: str = None) -> str:
    """Render a single-method image grid in the same card+banner style as
    :func:`make_compare_aggregations_html`.

    ``method_name`` is rendered on a full-width colored banner (white text
    on ``color``); the optional ``model_label`` / ``feat`` line is shown as
    a caption *underneath* the grid. ``subtitle`` is a short, plain-language
    description of the ranking regime, rendered in muted italics as a
    footnote *below* the grid and the model/feature caption.

    The grid is a fixed 3 columns (``repeat(3, 1fr)``) that resize with the
    card, so it always reads as a 3×N block and shrinks gracefully on
    smaller screens. The card is capped at the natural 3-thumbnail width so
    the full-width banner lines up exactly with the grid (no over-extend)
    and every tab keeps identical spacing regardless of image count.
    """
    cols = _GRID_COLS
    gap = _GRID_GAP
    pad_x = _CARD_PAD_X
    # Cap the card to the natural width of a 3-wide grid so the banner and
    # grid share the same extent and thumbnails don't balloon on wide screens.
    card_max = image_card_width()

    # Model + feature caption, rendered below the grid (see return).
    caption = ''
    if model_label or feat is not None:
        title_bits = []
        if model_label:
            title_bits.append(model_label)
        if feat is not None:
            title_bits.append(f'Feature {feat}')
        caption = (
            f'<div style="font-size:13px;font-weight:bold;color:#222;'
            f'margin-top:12px;letter-spacing:0.1px">'
            f'{" — ".join(title_bits)}</div>'
        )

    # Full-width banner — matches the grid width exactly (both fill the
    # card's content box), so Top / Mean always line up identically.
    banner = (
        f'<div style="background:{color};color:#ffffff;font-size:13px;font-weight:bold;'
        f'text-align:center;padding:6px 0;border-radius:5px;margin-bottom:12px;'
        f'letter-spacing:0.4px;box-sizing:border-box">{method_name}</div>'
    )
    # The aggregation-method explanation is rendered as a footnote *below* the
    # grid and the model/feature caption (see return), not under the banner.
    subtitle_html = ''
    if subtitle:
        subtitle_html = (
            f'<div style="font-size:11px;color:#777;font-style:italic;'
            f'margin-top:8px;line-height:1.4">{subtitle}</div>'
        )

    if not images_info:
        body = ('<div style="color:#aaa;font-style:italic;font-size:12px;'
                'padding:6px 2px">No examples available</div>')
    else:
        # Fixed 3-column grid: columns share the card width evenly and
        # resize with it; the card's max-width keeps thumbnails near
        # thumb_w on large screens.
        body = (f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
                f'gap:{gap}px">')
        for img, img_caption in images_info:
            url = _images.pil_to_data_url(img)
            parts = img_caption.split('<br>')
            cap_html = '<br>'.join(parts)
            body += (
                f'<div style="text-align:center">'
                f'<img src="{url}"'
                f' style="width:100%;height:auto;aspect-ratio:1/1;'
                f'border:1px solid #ccc;border-radius:3px;display:block"/>'
                f'<div style="font-size:10px;color:#555;margin-top:3px;line-height:1.4">'
                f'{cap_html}</div></div>'
            )
        body += '</div>'

    # Block-level card that fills the host Div (up to card_max) so the grid
    # has room to size into rather than shrink-to-fit inline-block. The
    # model + feature caption goes underneath the grid.
    return (
        f'<div style="font-family:Arial,Helvetica,sans-serif;background:#ffffff;'
        f'padding:16px {pad_x}px 14px {pad_x}px;box-sizing:border-box;'
        f'width:100%;max-width:{card_max}px">'
        + banner + body + caption + subtitle_html +
        '</div>'
    )


def make_compare_aggregations_html(top_infos, mean_infos, feat: int,
                                   n_each: int = 6, model_label: str = None) -> str:
    """Figure-ready side-by-side comparison of the first two aggregation methods.

    Only Top (Max Activation) and Mean Activation are shown so that a
    screenshot of this element stands alone as a clean figure panel.
    """
    col_thumb = min(_images.THUMB, 160)

    # Only the first two methods are shown in the figure
    sections = [
        ("Top Activation",  "#2563a8", top_infos),
        ("Mean Activation", "#1a7a4a", mean_infos),
    ]

    cols_per_row = 2
    section_gap = 24
    outer_pad_x = 20
    strip_w = cols_per_row * col_thumb + (cols_per_row - 1) * 6
    # Natural (un-shrunk) figure width — used only as a cap so the layout
    # never balloons past full size on a wide screen.
    fig_max = (len(sections) * strip_w + (len(sections) - 1) * section_gap
               + 2 * outer_pad_x)

    # Outer container — white background, no border decoration so the figure
    # can be cropped cleanly. It is fluid (``width:100%``) up to its natural
    # size (``max-width:fig_max``), so it shrinks to fit the panel instead of
    # overflowing into the neighbouring column. A subtle bottom-padding keeps
    # images from being clipped.
    html = (
        f'<div style="font-family:Arial,Helvetica,sans-serif;background:#ffffff;'
        f'padding:16px {outer_pad_x}px 14px {outer_pad_x}px;'
        f'width:100%;max-width:{fig_max}px;box-sizing:border-box">'
        f'<div style="font-size:13px;font-weight:bold;color:#222;margin-bottom:14px;'
        f'letter-spacing:0.1px">'
        + (f'{model_label} — ' if model_label else '')
        + f'Feature {feat}</div>'
        f'<div style="display:flex;gap:{section_gap}px;align-items:flex-start">'
    )

    for method_name, color, infos in sections:
        shown = (infos or [])[:n_each]

        # Each section flexes to share the row equally and may shrink below
        # its natural width (``min-width:0`` lets the grid contract).
        html += (
            f'<div style="display:flex;flex-direction:column;flex:1 1 0;min-width:0">'
            f'<div style="background:{color};color:#ffffff;font-size:13px;font-weight:bold;'
            f'text-align:center;padding:6px 0;border-radius:5px;margin-bottom:10px;'
            f'letter-spacing:0.4px;width:100%;box-sizing:border-box">{method_name}</div>'
            f'<div style="display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:6px">'
        )
        if not shown:
            html += '<div style="color:#aaa;font-style:italic;font-size:11px;padding:8px">No images</div>'
        for img, caption in shown:
            url = _images.pil_to_data_url(img)
            parts = caption.split('<br>')
            cap_html = '<br>'.join(parts)
            html += (
                f'<div style="text-align:center">'
                f'<img src="{url}"'
                f' style="width:100%;height:auto;aspect-ratio:1/1;object-fit:cover;'
                f'border:1px solid #ccc;border-radius:3px;display:block"/>'
                f'<div style="font-size:9px;color:#555;margin-top:3px;line-height:1.35">'
                f'{cap_html}</div></div>'
            )
        html += '</div></div>'

    html += '</div></div>'  # close grid-row, outer figure
    return html


def make_cross_sae_comparison_html(ds_a: dict, feat_a: int, ds_b: dict, feat_b: int,
                                   n: int = 4, size: int = 160, alpha: float = 1.0) -> str:
    """Two side-by-side 2x2 grids: left = SAE A / feat_a, right = SAE B / feat_b.

    Reads top images + heatmaps from the two dataset dicts directly, so it
    works across datasets without needing the active state.
    """
    from .rendering import _render_overlay_from_ds  # lazy import (see top)

    def _collect(ds, feat):
        items = []
        for slot in range(min(n, ds['top_img_idx'].shape[1])):
            result = _render_overlay_from_ds(ds, feat, slot, size=size, alpha=alpha)
            if result:
                items.append(result)
            if len(items) == n:
                break
        return items

    items_a = _collect(ds_a, feat_a)
    items_b = _collect(ds_b, feat_b)

    def _strip_dim(label: str) -> str:
        """Strip parenthetical dim info like '(d=32K)' or '(d=32K, k=160)'."""
        out = label
        while '(' in out and ')' in out:
            l, r = out.index('('), out.index(')')
            out = out[:l].rstrip() + out[r + 1:]
        return out.strip(' —').strip()

    def _grid_html(items, model_label: str, feat_num: int, color: str) -> str:
        header = (
            f'<div style="background:{color};color:#fff;text-align:center;'
            f'padding:5px 6px 4px 6px;border-radius:4px;margin-bottom:6px;line-height:1.4">'
            f'<div style="font-size:12px;font-weight:bold">{model_label}</div>'
            f'<div style="font-size:10px;opacity:0.88">Feature {feat_num}</div>'
            f'</div>'
        )
        grid = '<div style="display:grid;grid-template-columns:repeat(2,{s}px);gap:4px">'.format(s=size)
        for img, cap in items:
            url = _images.pil_to_data_url(img)
            grid += (f'<div style="text-align:center">'
                     f'<img src="{url}" width="{size}" height="{size}"'
                     f' style="border:1px solid #ccc;border-radius:3px;display:block"/>'
                     f'<div style="font-size:9px;color:#555;margin-top:2px">{cap}</div></div>')
        grid += '</div>'
        return f'<div style="display:flex;flex-direction:column">{header}{grid}</div>'

    label_a = _strip_dim(ds_a['label'])
    label_b = _strip_dim(ds_b['label'])
    col_a = _grid_html(items_a, label_a, feat_a, "#2563a8")
    col_b = _grid_html(items_b, label_b, feat_b, "#b85c00")

    return (
        '<div style="display:flex;gap:16px;padding:8px;background:#fafafa;'
        'border:1px solid #ddd;border-radius:6px">'
        + col_a + col_b + '</div>'
    )
