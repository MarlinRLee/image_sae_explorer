"""SAE Feature Explorer — modular package backing demo/explorer_app.py.

The entry script composes the pieces: per-session state and CLI parsing
(``state``), the context object threaded into every panel (``context``),
registry/data loading (``registry``, ``loaders``), pure image + HTML helpers
(``images``, ``rendering``, ``html_views``), persistence of feature labels
(``persistence``), on-demand inference (``activations``, ``clip_loader``),
Gemini auto-labeling (``gemini``), classifier export (``classifier_export``),
and the UI panels (``panels/``).
"""
