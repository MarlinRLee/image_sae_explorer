#!/usr/bin/env python
"""Sanity-check a model registry (``configs/models.yaml``) before serving.

Catches the mistakes that otherwise only surface as a stack trace mid-session:

  - malformed YAML / missing required keys / not exactly one ``primary: true``
    (delegated to ``explorer.registry.load_registry``);
  - a registry entry whose ``data_file`` is present locally but is missing one
    of the tensor fields the explorer requires (a precompute bug);
  - ``backbone`` / ``token_type`` recorded in the ``.pt`` disagreeing with the
    registry block (the on-demand classifier/heatmap inference trusts the
    registry, so a mismatch silently produces wrong scores).

Run it on the canonical registry and the offline demo registry::

    python scripts/validate_registry.py
    python scripts/validate_registry.py --registry demo_data/models.yaml \\
        --data-dir demo_data

Exits non-zero if any check fails, so it slots into CI / pre-deploy.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from explorer.registry import load_registry  # noqa: E402

# Tensor / scalar fields every explorer_data*.pt must carry. The JSON name
# sidecars (feature_names, history, ...) are merged in by the loader at serve
# time, not stored in the .pt, so they're intentionally absent here. Keep this
# in sync with scripts/build_demo_data.py:_make_explorer_pt and docs/DATA_FORMAT.md.
_REQUIRED_PT_FIELDS = (
    "image_paths", "d_model", "n_images", "patch_grid", "image_size",
    "top_img_idx", "top_img_act", "mean_img_idx", "mean_img_act",
    "crop_img_idx", "crop_img_act",
    "feature_frequency", "feature_mean_act",
    "umap_coords", "dict_umap_coords",
)


def _check_pt(path: str, entry) -> list[str]:
    """Return a list of problem strings for one loaded .pt (empty = OK)."""
    import torch
    problems: list[str] = []
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        return [f"could not load: {e}"]

    missing = [f for f in _REQUIRED_PT_FIELDS if f not in d]
    if missing:
        problems.append(f"missing fields: {', '.join(missing)}")

    # Registry vs .pt consistency — the inference path trusts the registry.
    pt_backbone = d.get("backbone")
    if pt_backbone is not None and pt_backbone != entry.backbone:
        problems.append(
            f"backbone mismatch: registry={entry.backbone!r} .pt={pt_backbone!r}")
    pt_token = d.get("token_type")
    if pt_token is not None and pt_token != entry.token_type:
        problems.append(
            f"token_type mismatch: registry={entry.token_type!r} .pt={pt_token!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(_HERE, "..", "configs", "models.yaml"),
                    help="Path to the registry YAML (default: configs/models.yaml).")
    ap.add_argument("--data-dir", default=None,
                    help="Directory holding the .pt files. If set, present "
                         "data_files are loaded and their fields checked.")
    args = ap.parse_args()

    print(f"Validating registry: {args.registry}")
    try:
        reg = load_registry(args.registry)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL — {e}")
        return 1
    print(f"  OK — 1 primary ({reg.primary.id}) + {len(reg.compares)} compare(s)")

    ok = True
    for entry in reg.all_entries:
        tag = "primary" if entry.primary else "compare"
        line = f"  [{tag}] {entry.id}: backbone={entry.backbone} layer={entry.layer} token={entry.token_type}"
        if args.data_dir is None:
            print(line + "  (skipped .pt check — no --data-dir)")
            continue
        pt_path = os.path.join(args.data_dir, entry.data_file)
        if not os.path.isfile(pt_path):
            print(line + f"  (no local {entry.data_file} — skipped)")
            continue
        problems = _check_pt(pt_path, entry)
        if problems:
            ok = False
            print(line + "  FAIL")
            for p in problems:
                print(f"      - {p}")
        else:
            print(line + "  OK")

    print("PASS" if ok else "FAIL — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
