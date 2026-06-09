"""
Interpretability Index (II) computed separately under each MEI aggregation.

The stored ``interp_index`` in explorer_data is computed from the MAX-activation
MEIs only (``top_img_idx``). This script computes the same LPIPS II
(Klindt et al., 2023 — see interpretability_metric.pdf) under all three
aggregations the explorer tracks, by drawing each feature's top-M MEIs from the
corresponding ranking:

    max   -> top_img_idx     (single most-activating patch)
    mean  -> mean_img_idx    (mean activation over all patches)
    crop  -> crop_img_idx    (mean of the top-8 positive patches)

    II(u) = - mean_{j<k} LPIPS(x_j, x_k)   over the top-M MEIs   (higher = more
                                                                  coherent)

It plots the three II distributions overlaid for comparison ("which aggregation
yields more interpretable features?") and saves the per-feature arrays.

Heavy-ish (LPIPS over ~M-choose-2 pairs x d_model x 3) — run via sbatch
(run_ii_by_aggregation.sh). The image tensor cache is shared across the three
aggregations, so the marginal cost of the 2nd/3rd method is just the extra pairs.
"""

import argparse
import itertools
import os

import numpy as np
import torch
from PIL import Image

AGGS = (("max", "top_img_idx"), ("mean", "mean_img_idx"), ("crop", "crop_img_idx"))
COLORS = {"max": "#2563a8", "mean": "#1a7a4a", "crop": "#c2691a"}


def _load_lpips_tensor(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def main():
    p = argparse.ArgumentParser(description="II under each MEI aggregation")
    p.add_argument("--data", required=True)
    p.add_argument("--interp-m", type=int, default=5)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--resize", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--image-dir", default=None)
    p.add_argument("--extra-image-dir", action="append", default=[])
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)

    bases = [b for b in ([args.image_dir] + args.extra_image_dir) if b]

    def resolve(pth):
        if os.path.isabs(pth) or not bases:
            return pth
        for base in bases:
            full = os.path.join(base, pth)
            if os.path.exists(full):
                return full
        return os.path.join(bases[0], pth)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading explorer data from {args.data}...")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    image_paths = [resolve(p) for p in data["image_paths"]]
    d_model = data["d_model"]
    n_top = data["top_img_idx"].shape[1]
    m = min(args.interp_m, n_top)
    print(f"  d_model={d_model}, n_images={data['n_images']}, top-{n_top} stored, M={m}")

    import lpips
    print(f"Loading LPIPS (net={args.lpips_net})...")
    loss_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    # Image tensors are shared across aggregations (same image pool).
    tensor_cache = {}

    def get_tensor(i):
        t = tensor_cache.get(i)
        if t is None:
            try:
                t = _load_lpips_tensor(image_paths[i], args.resize)
            except Exception:
                t = False
            tensor_cache[i] = t
        return t if t is not False else None

    results = {}
    for agg, key in AGGS:
        idx_tensor = data[key]                      # (d_model, n_top)
        mei_idx = [
            [int(idx_tensor[f, j].item())
             for j in range(m) if int(idx_tensor[f, j].item()) >= 0]
            for f in range(d_model)
        ]
        dist_sums = np.zeros(d_model, dtype=np.float64)
        pair_counts = np.zeros(d_model, dtype=np.int64)
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

        print(f"[{agg}] computing pairwise LPIPS over {key} MEIs...")
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
            if (feat + 1) % 5000 == 0:
                print(f"    [{feat + 1}/{d_model}] ({len(tensor_cache)} imgs cached)", flush=True)
        flush()

        ii = np.full(d_model, np.nan, dtype=np.float32)
        have = pair_counts > 0
        ii[have] = -(dist_sums[have] / pair_counts[have]).astype(np.float32)
        results[agg] = ii
        v = ii[have]
        print(f"  [{agg}] finite II {int(have.sum())}/{d_model}  "
              f"min={v.min():.4f} median={np.median(v):.4f} mean={v.mean():.4f} max={v.max():.4f}")

    # --- Save arrays ---
    res_path = os.path.join(out_dir, "ii_by_aggregation.pt")
    torch.save({agg: torch.from_numpy(results[agg]) for agg in results}
               | {"interp_index_m": m, "lpips_net": args.lpips_net}, res_path)
    print(f"  wrote {res_path}")

    # --- Plot overlaid histograms ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite_all = np.concatenate([results[a][np.isfinite(results[a])] for a, _ in AGGS])
    lo, hi = np.percentile(finite_all, 0.5), np.percentile(finite_all, 99.5)
    bins = np.linspace(lo, hi, 60)

    plt.figure(figsize=(7, 4.5))
    for agg, _ in AGGS:
        v = results[agg][np.isfinite(results[agg])]
        plt.hist(v, bins=bins, alpha=0.45, color=COLORS[agg],
                 label=f"{agg} (median {np.median(v):.3f})")
        plt.axvline(np.median(v), color=COLORS[agg], ls="--", lw=1.2)
    plt.xlabel("Interpretability Index  (−mean pairwise LPIPS over top-%d MEIs)" % m)
    plt.ylabel("features")
    plt.title("II by MEI aggregation (higher = more coherent MEIs)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "ii_by_aggregation_hist.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  wrote {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
