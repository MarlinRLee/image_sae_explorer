"""Sortable feature table + name search panel.

Displays every feature in the active dataset, sorted by frequency
(descending). The user can:
  - click a row to select that feature (drives ``ctx.select_feature``)
  - search by manual or Gemini-auto-interp label substring
  - clear the search to restore the full list

``build()`` returns the refresh hooks (``apply_order`` /
``get_sorted_order`` / ``update_table_names``) so the bootstrap can refresh
the table after dataset switches or label edits.
"""

from typing import Callable, Optional

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, DataTable, Div,
                          HTMLTemplateFormatter, NumberFormatter, TableColumn,
                          TextInput)

# Sentinel for features without an II (dead / <2 MEIs). Sits below the [0,1]
# II range so a high->low sort drops them to the BOTTOM (Bokeh sorts NaN/null
# to the top, which we don't want); the column formatter renders it blank.
_II_NONE = -1.0


def build(ctx, display_name: Callable[[int], str],
          on_feature_pick: Optional[Callable[[int], None]] = None):
    """Construct the feature table + search row.

    Args:
        ctx: the session :class:`~explorer.context.Context`.
        display_name: callable returning the manual-or-auto-interp name
            for a feature id (the bootstrap's ``_display_name``).
        on_feature_pick: callable invoked when the user clicks a row.
            Defaults to ``ctx.select_feature``.

    Returns a dict with the panel layout, the data source, and the
    refresh hooks (``apply_order`` / ``get_sorted_order`` /
    ``update_table_names``) the bootstrap calls after dataset/label changes.
    """
    state = ctx.state
    init_order = np.argsort(-state.freq)

    def _row_data(order):
        """Assemble the table ColumnDataSource dict for a given feature order.
        Single source of truth so the initial build and ``apply_order`` can't
        drift; the ``ii`` column is included only when the dataset has one."""
        s = ctx.state
        d = dict(
            feature_idx=order.tolist(),
            frequency=s.freq[order].tolist(),
            mean_act=s.mean_act[order].tolist(),
            name=[display_name(int(i)) for i in order],
        )
        if s.has_ii:
            # Use the _II_NONE sentinel (not NaN/None) for features without an
            # II so they sort to the bottom on a high->low sort; the column
            # formatter renders the sentinel as a blank cell.
            ii_ord = s.ii[order]
            d['ii'] = [float(v) if np.isfinite(v) else _II_NONE for v in ii_ord]
        return d

    feature_list_source = ColumnDataSource(data=_row_data(init_order))

    columns = [
        TableColumn(field="feature_idx", title="Feature", width=60),
        TableColumn(field="frequency",   title="Freq", width=70,
                    formatter=NumberFormatter(format="0,0")),
        TableColumn(field="mean_act",    title="Mean Act", width=80,
                    formatter=NumberFormatter(format="0.0000")),
    ]
    # Interpretability Index (Klindt et al., 2023), reported on a 0..1
    # coherence scale (1 - LPIPS); higher = more coherent top-MEIs. Sortable
    # like any column. Features without an II carry the _II_NONE sentinel:
    # the formatter shows them blank, but they sort to the bottom high->low.
    if state.has_ii:
        ii_formatter = HTMLTemplateFormatter(
            template='<% if (value >= 0) { %><%= value.toFixed(3) %><% } %>')
        columns.append(TableColumn(field="ii", title="II", width=70,
                                   formatter=ii_formatter))
    columns.append(TableColumn(field="name", title="Name", width=200))

    feature_table = DataTable(
        source=feature_list_source,
        columns=columns,
        width=500, height=500, sortable=True, index_position=None,
    )

    def get_sorted_order():
        """Return the freq-desc index order, filtered by
        ``ctx.ui.search_filter`` (a set of feat ids) when one is set."""
        s = ctx.state
        order = np.argsort(-s.freq)
        f = ctx.ui.search_filter
        if f is not None:
            order = order[np.isin(order, list(f))]
        return order

    def apply_order(order) -> None:
        feature_list_source.data = _row_data(order)

    def update_table_names() -> None:
        """Refresh the name column without changing the row order."""
        apply_order(np.asarray(feature_list_source.data['feature_idx']))

    def _on_table_select(attr, old, new):
        if not new:
            return
        cb = on_feature_pick or ctx.select_feature
        if cb is not None:
            cb(feature_list_source.data['feature_idx'][new[0]])

    feature_list_source.selected.on_change('indices', _on_table_select)

    # Search row
    search_input = TextInput(
        title="Search feature names:",
        placeholder="Type to search...",
        width=220,
    )
    search_btn = Button(label="Search", width=70, button_type="primary")
    clear_search_btn = Button(label="Clear", width=60)
    search_result_div = Div(text="", width=360)

    def _do_search():
        query = search_input.value.strip().lower()
        if not query:
            ctx.ui.search_filter = None
            search_result_div.text = ""
            apply_order(get_sorted_order())
            return
        # Search manual labels AND Gemini auto-interp labels -- auto-interp
        # is by far the larger pool on a typical dataset.
        s = ctx.state
        matches = {i for i, name in s.feature_names.items() if query in name.lower()}
        matches |= {i for i, name in s.auto_interp_names.items() if query in name.lower()}
        ctx.ui.search_filter = matches
        apply_order(get_sorted_order())
        if matches:
            search_result_div.text = (
                f'<span style="color:#1a6faf"><b>{len(matches)}</b> feature(s) matching '
                f'&ldquo;{query}&rdquo;</span>'
            )
        else:
            search_result_div.text = (
                f'<span style="color:#c00">No features named &ldquo;{query}&rdquo;</span>'
            )

    def _do_clear_search():
        search_input.value = ""
        ctx.ui.search_filter = None
        search_result_div.text = ""
        apply_order(get_sorted_order())

    search_btn.on_click(_do_search)
    clear_search_btn.on_click(_do_clear_search)

    search_panel = column(
        row(search_input, search_btn, clear_search_btn),
        search_result_div,
    )
    layout = column(search_panel, feature_table)

    def on_dataset_changed(idx: int) -> None:
        ctx.ui.search_filter = None
        apply_order(get_sorted_order())

    return {
        'layout':                layout,
        'feature_list_source':   feature_list_source,
        'feature_table':         feature_table,
        'search_input':          search_input,
        'search_result_div':     search_result_div,
        'get_sorted_order':      get_sorted_order,
        'apply_order':           apply_order,
        'update_table_names':    update_table_names,
        'on_dataset_changed':    on_dataset_changed,
    }
