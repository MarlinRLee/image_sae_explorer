"""Self-contained Bokeh panels.

Each module in this package exports a ``build(ctx, ...)`` factory that
constructs fresh widgets (per Bokeh session), wires their internal
callbacks, and returns the panel layout plus any widgets the bootstrap
needs to access from elsewhere.

The bootstrap orchestrates: it constructs the per-session
:class:`~explorer.context.Context` (which owns ``state`` / ``ui`` /
``args``), calls each panel's ``build(ctx, ...)``, and assembles the
final layout.
"""
