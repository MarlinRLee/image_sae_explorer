"""Summary panel — a static-ish HTML overview of the active dataset.

Re-rendered on every dataset switch. The HTML formatter is pure (it reads
``ctx.state`` only) so it's easy to test by passing a fake context.
"""

from bokeh.models import Div


def _make_summary_html(ctx) -> str:
    """Render the summary HTML for the active dataset.

    Reads ``ctx.state`` at call time, so it always reflects the
    currently-applied dataset.
    """
    state = ctx.state
    ds = state.ds
    n_truly_active = int((state.freq > 0).sum())
    n_dead = state.d_model - n_truly_active
    tok_label = ("CLS global" if ds.get('token_type') == 'cls'
                 else f"{state.patch_grid}x{state.patch_grid} = {state.patch_grid**2} patches")
    backbone_label = ds.get('backbone', 'dinov3').upper()
    sae_url = ds.get('sae_url')
    dl_row = (f'<tr><td><b>SAE weights:</b></td>'
              f'<td><a href="{sae_url}" download style="color:#1a6faf">&#x2B07; Download</a></td></tr>'
              if sae_url else '')
    return f"""
<div style="background:#f0f4f8;padding:12px;border-radius:6px;margin-bottom:8px;">
<h2 style="margin:0 0 8px 0">SAE Feature Explorer</h2>
<table style="font-size:13px;">
<tr><td><b>Active model:</b></td><td><b style="color:#1a6faf">{ds['label']}</b></td></tr>
<tr><td><b>Backbone:</b></td><td>{backbone_label}</td></tr>
<tr><td><b>Token type:</b></td><td>{ds.get('token_type','spatial')}</td></tr>
<tr><td><b>Dictionary size:</b></td><td>{state.d_model:,}</td></tr>
<tr><td><b>Active (fired &ge;1):</b></td><td>{n_truly_active:,} ({100*n_truly_active/state.d_model:.1f}%)</td></tr>
<tr><td><b>Dead:</b></td><td>{n_dead:,} ({100*n_dead/state.d_model:.1f}%)</td></tr>
<tr><td><b>Images:</b></td><td>{state.n_images:,}</td></tr>
<tr><td><b>Tokens/image:</b></td><td>{tok_label}</td></tr>
{dl_row}
</table>
</div>"""


def build(ctx):
    """Construct the summary ``Div`` and return ``{layout, refresh,
    on_dataset_changed}``. ``refresh()`` re-renders from current state."""
    summary_div = Div(text=_make_summary_html(ctx), width=700)

    def refresh() -> None:
        summary_div.text = _make_summary_html(ctx)

    def on_dataset_changed(idx: int) -> None:
        refresh()

    return {
        'layout':              summary_div,
        'summary_div':         summary_div,
        'refresh':             refresh,
        'on_dataset_changed':  on_dataset_changed,
    }
