#!/usr/bin/env python3
"""
Center UMAP coordinates in explorer_data*.pt files in-place.

Shifts umap_coords and dict_umap_coords so the mean of live (non-NaN)
points is (0, 0).  Skips heatmap / patch_acts sidecars and files that
have already been centered (mean < 1e-6).

Usage:
    python scripts/center_umap_inplace.py [--dir explorer_data]
"""

import argparse
import glob
import os

import numpy as np
import torch


def center(coords: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    arr = coords.numpy().copy()
    live = ~np.isnan(arr[:, 0])
    mean = arr[live].mean(axis=0)
    arr[live] -= mean
    return torch.from_numpy(arr), mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="explorer_data",
                        help="Directory containing explorer_data*.pt files")
    args = parser.parse_args()

    pattern = os.path.join(args.dir, "**", "explorer_data*.pt")
    files = sorted(f for f in glob.glob(pattern, recursive=True)
                   if "_heatmaps" not in f and "_patch_acts" not in f)

    if not files:
        print(f"No matching files found in {args.dir}")
        return

    for path in files:
        d = torch.load(path, map_location="cpu", weights_only=False)

        if "umap_coords" not in d:
            print(f"  SKIP (no umap_coords): {path}")
            continue

        # Check if already centered
        uc = d["umap_coords"].numpy()
        live = ~np.isnan(uc[:, 0])
        existing_mean = np.abs(uc[live].mean(axis=0)).max()
        if existing_mean < 1e-6:
            print(f"  SKIP (already centered): {os.path.basename(path)}")
            continue

        d["umap_coords"],      mean_act  = center(d["umap_coords"])
        d["dict_umap_coords"], mean_dict = center(d["dict_umap_coords"])

        torch.save(d, path)
        print(f"  Centered: {os.path.basename(path)}"
              f"  (act shift: [{mean_act[0]:+.3f}, {mean_act[1]:+.3f}],"
              f"  dict shift: [{mean_dict[0]:+.3f}, {mean_dict[1]:+.3f}])")

    print("Done.")


if __name__ == "__main__":
    main()
