"""Per-session explorer context.

One :class:`Context` is constructed per Bokeh session in
``demo/explorer_app.py`` and threaded explicitly into every panel
``build(ctx, ...)`` factory and every rendering / activation helper.

It replaces the old ``explorer.runtime`` module, whose attributes were
process-global and therefore *shared* across concurrent browser sessions:
``explorer_app.py`` re-runs per session, so the last session to load would
overwrite ``runtime.state``, and a render worker thread in session A could
then read session B's active dataset. Passing the context explicitly keeps
each session's ``state`` / ``ui`` isolated.

Thread-safety: the heatmap renders run on a shared ``ThreadPoolExecutor``.
The worker closures capture ``ctx`` (a plain object reference) and only
*read* its tensors, so this is safe — unlike ``curdoc()``, which returns
the wrong document off the document thread and so can't be used to resolve
per-session state inside a worker.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .state import _State, _UI


class Context:
    """Everything one browser session needs to render and mutate the demo.

    ``args`` / ``datasets`` are set at construction; ``state`` and ``ui``
    are fresh per session. ``select_feature`` and ``ensure_loaded`` are
    late-bound by the bootstrap once the widgets that implement them exist
    — they are ``None`` until then, and the only readers are UI callbacks,
    which fire long after construction.
    """

    def __init__(self, args, datasets: List[dict]):
        self.args = args
        self.datasets = datasets
        self.ui = _UI()
        self.state = _State(datasets)

        # Late-bound cross-module callables (see class docstring).
        self.select_feature: Optional[Callable[[int], None]] = None
        self.ensure_loaded: Optional[Callable[[int], None]] = None

    def load_image(self, img_idx: int):
        """Open an image by global index using the active dataset's
        ``image_paths``. Lives here (not in :mod:`explorer.images`) because
        it depends on the session's active ``state``."""
        from . import images
        return images._open_image(self.state.image_paths[img_idx])
