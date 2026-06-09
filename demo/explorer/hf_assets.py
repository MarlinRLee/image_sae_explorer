"""HuggingFace asset-download helpers.

Shared by the two code paths that pull a model's precompute sidecars from
the HF dataset repo:

- ``demo/bootstrap_demo.py`` — bulk download at build / first launch.
- ``explorer.loaders.ensure_loaded`` — lazy, on-demand download of a
  *compare* model the first time it is selected from the dropdown.

Keeping the primitive here (rather than in the bootstrap script) lets both
callers share one skip-if-present, warn-on-missing-sidecar implementation.
"""

from __future__ import annotations

import os


def heatmap_filename(data_file: str) -> str:
    """``explorer_data_x.pt`` -> ``explorer_data_x_heatmaps.pt``."""
    return data_file.replace('.pt', '_heatmaps.pt')


def download_if_missing(repo_id: str, filename: str, local_dir: str,
                        token: str | None, *, quiet: bool = False) -> str:
    """Download ``filename`` from the dataset ``repo_id`` into ``local_dir``
    if it isn't already there. Returns the local path.

    Skip-and-print when present so repeat runs are fast and quiet.
    """
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest):
        if not quiet:
            print(f"  [skip] {filename}")
        return dest
    if not quiet:
        print(f"  [pull] {filename}")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset",
        local_dir=local_dir, token=token,
    )


def download_model_assets(repo_id: str, data_file: str, sae_file: str | None,
                          local_dir: str, token: str | None, *,
                          label: str = "", quiet: bool = False) -> str:
    """Download one model's ``.pt`` + optional ``_heatmaps.pt`` + optional
    ``.pth`` SAE checkpoint into ``local_dir``. Returns the local ``.pt`` path.

    The heatmap and SAE sidecars are optional — CLS-token SAEs and minimal
    demos lack heatmaps, and the explorer needs the SAE checkpoint only for
    the summary panel's download link — so a missing one warns and continues.
    """
    os.makedirs(local_dir, exist_ok=True)
    pt_path = download_if_missing(repo_id, data_file, local_dir, token, quiet=quiet)
    try:
        download_if_missing(repo_id, heatmap_filename(data_file), local_dir,
                            token, quiet=quiet)
    except Exception as e:
        if not quiet:
            print(f"  [warn] no heatmap sidecar for {label or data_file}: {e}")
    if sae_file:
        try:
            download_if_missing(repo_id, sae_file, local_dir, token, quiet=quiet)
        except Exception as e:
            if not quiet:
                print(f"  [warn] no SAE checkpoint for {label or data_file}: {e}")
    return pt_path
