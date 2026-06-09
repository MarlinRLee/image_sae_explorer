"""
Post-hoc Interpretability Index (II) enrichment for explorer_data.pt files.

Implements the metric from Klindt et al. (2023), "Identifying Interpretable
Visual Features in Artificial and Biological Neural Systems" (see
Klindt et al., 2023). For a direction in activation space -- here, an
SAE feature -- the II is the average pairwise image similarity among its top-M
Maximally Exciting Images (MEIs):

    II(u) = mean_{j<k} sim(x_j, x_k)

We use LPIPS (Zhang et al., 2018), the paper's validated human proxy. LPIPS is
a perceptual *distance* (lower = more similar), and the paper reports negative
II values, so we store

    II = - mean pairwise LPIPS over the top-M MEIs

meaning higher (closer to 0) = more coherent / interpretable. Dead or near-dead
features (fewer than 2 valid MEIs) get II = NaN.

This script does NOT re-run the backbone or the SAE -- it only needs the
existing explorer_data.pt (for image paths and top-MEI indices) and LPIPS.

Usage
-----
    python add_interpretability_index.py \
        --data ../explorer_data_d32000_k160_val.pt \
        --image-dir /path/to/val \
        --interp-m 5

The enriched file is saved to --output-path (defaults to overwriting --data
with a backup copy at <data>.bak), adding:
    'interp_index'        : Tensor (n_features,) float32   (NaN for dead feats)
    'interp_index_m'      : int                            (M actually used)
    'interp_index_metric' : str   e.g. "neg_lpips_alex"
"""

import argparse
import itertools
import os
import shutil

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_lpips_tensor(path, size):
    """Load an image as an LPIPS input tensor: (3, size, size), RGB, in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # (H, W, 3) in [0, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1)           # (3, H, W)
    return t * 2.0 - 1.0                                  # -> [-1, 1]


def main():
    parser = argparse.ArgumentParser(
        description="Add the LPIPS Interpretability Index to explorer_data.pt")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to explorer_data.pt")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output path (default: overwrite --data, keeping .bak)")
    parser.add_argument("--interp-m", type=int, default=5,
                        help="Number of top MEIs to average over (paper uses M=5)")
    parser.add_argument("--lpips-net", type=str, default="alex",
                        choices=["alex", "vgg", "squeeze"],
                        help="Backbone for LPIPS (paper uses AlexNet/VGG)")
    parser.add_argument("--resize", type=int, default=64,
                        help="Side length to resize MEIs to before LPIPS "
                             "(small keeps it fast; paper used small images)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of image pairs per LPIPS forward pass")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating a .bak copy before overwriting")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Primary image directory for resolving bare filenames")
    parser.add_argument("--extra-image-dir", type=str, action="append", default=[],
                        help="Additional image directory (repeatable)")
    args = parser.parse_args()

    image_bases = [b for b in ([args.image_dir] + args.extra_image_dir) if b]

    def resolve_path(p):
        if os.path.isabs(p) or not image_bases:
            return p
        for base in image_bases:
            full = os.path.join(base, p)
            if os.path.exists(full):
                return full
        return os.path.join(image_bases[0], p)  # fallback

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load explorer data ---
    print(f"Loading explorer data from {args.data}...")
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    image_paths = [resolve_path(p) for p in data['image_paths']]
    d_model = data['d_model']
    top_img_idx = data['top_img_idx']       # (n_features, n_top)
    n_top_stored = top_img_idx.shape[1]
    m = min(args.interp_m, n_top_stored)
    print(f"  d_model={d_model}, n_images={data['n_images']}, "
          f"top-{n_top_stored} stored, using M={m}")

    # --- Load LPIPS ---
    import lpips
    print(f"Loading LPIPS (net={args.lpips_net})...")
    loss_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    # --- Per-feature valid MEI image indices ---
    mei_idx = [
        [int(top_img_idx[feat, j].item())
         for j in range(m) if int(top_img_idx[feat, j].item()) >= 0]
        for feat in range(d_model)
    ]

    # Cache image tensors (dedupe across features; many images are MEIs for
    # several features). Keyed by global image index.
    tensor_cache = {}

    def get_tensor(img_idx):
        t = tensor_cache.get(img_idx)
        if t is None:
            try:
                t = _load_lpips_tensor(image_paths[img_idx], args.resize)
            except Exception:
                t = False  # mark as unloadable so we don't retry
            tensor_cache[img_idx] = t
        return t if t is not False else None

    ii = np.full(d_model, np.nan, dtype=np.float32)
    dist_sums = np.zeros(d_model, dtype=np.float64)
    pair_counts = np.zeros(d_model, dtype=np.int64)

    # Pair buffer batched across features for GPU efficiency.
    buf_a, buf_b, buf_feat = [], [], []

    def flush():
        if not buf_a:
            return
        a = torch.stack(buf_a).to(device)
        b = torch.stack(buf_b).to(device)
        with torch.inference_mode():
            d = loss_fn(a, b).view(-1).cpu().numpy().astype(np.float64)
        for feat, dist in zip(buf_feat, d):
            dist_sums[feat] += dist
            pair_counts[feat] += 1
        buf_a.clear(); buf_b.clear(); buf_feat.clear()

    print(f"Computing pairwise LPIPS over MEIs for {d_model} features...")
    for feat in range(d_model):
        idxs = mei_idx[feat]
        if len(idxs) < 2:
            continue
        for ia, ib in itertools.combinations(idxs, 2):
            ta, tb = get_tensor(ia), get_tensor(ib)
            if ta is None or tb is None:
                continue
            buf_a.append(ta); buf_b.append(tb); buf_feat.append(feat)
            if len(buf_a) >= args.batch_size:
                flush()
        if (feat + 1) % 2000 == 0:
            print(f"  [{feat + 1}/{d_model}] features queued "
                  f"({len(tensor_cache)} images cached)", flush=True)
    flush()

    # II = - mean pairwise LPIPS  (only where we got >= 1 pair)
    have = pair_counts > 0
    ii[have] = -(dist_sums[have] / pair_counts[have]).astype(np.float32)

    n_finite = int(have.sum())
    if n_finite:
        vals = ii[have]
        print(f"  Finite II for {n_finite}/{d_model} features. "
              f"min={vals.min():.4f} median={np.median(vals):.4f} max={vals.max():.4f}")
    else:
        print("  WARNING: no features produced a finite II.")

    # --- Save ---
    output_path = args.output_path or args.data
    if output_path == args.data and not args.no_backup:
        bak_path = args.data + ".bak"
        print(f"Creating backup at {bak_path}...")
        shutil.copy2(args.data, bak_path)

    data['interp_index'] = torch.from_numpy(ii)               # float32 (n_features,)
    data['interp_index_m'] = m
    data['interp_index_metric'] = f"neg_lpips_{args.lpips_net}"

    print(f"Saving enriched explorer data to {output_path}...")
    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
