"""
Psychophysics difficulty sweep on COHERENT units, at high resolution, under the
three similarity levels of Klindt et al. (2023) — Low (Color), Mid (LPIPS),
High (CLIP semantic). Main DINOv3 dataset only.

This mirrors the paper's setup more closely than psychophysics_eval.py:
  - "grab coherent units": the feature sample is restricted to high-II features
    (the interpretable ones), instead of a uniform sample over all features.
  - high resolution: LPIPS / color read images at --resize (e.g. 224, vs 64).
  - three similarity metrics for the 2AFC decision (paper Sec. 2.2):
        low   = color   : -L2 over per-image spatial-average RGB
        mid   = lpips   : -LPIPS(alex)        (the paper's validated proxy)
        high  = clip    : cosine over CLIP image embeddings (semantic; our stand-in
                          for the paper's class-label metric, which needs labels
                          we don't have for the COCO/val pool)

Reuses the pool pass, trial selection, and 2AFC scoring from psychophysics_eval.
Heavy (re-runs DINOv3+SAE for activations, + CLIP pass, + LPIPS at high res) —
run via sbatch (run_psychophysics_sim_metrics.sh).
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psychophysics_eval import (compute_pool_activations, select_psych_trials,
                                afc_correct, LpipsSim)

AGGREGATIONS = ("max", "mean", "crop")
METRICS = ("color", "lpips", "clip")           # low / mid / high
METRIC_LABEL = {"color": "low (color)", "lpips": "mid (LPIPS)", "high": "high (CLIP)",
                "clip": "high (CLIP)"}
METRIC_COLOR = {"color": "#c2691a", "lpips": "#1a7a4a", "clip": "#2563a8"}


# ---------------------------------------------------------------------------
# Per-image feature vectors for the low (color) and high (CLIP) metrics
# ---------------------------------------------------------------------------

class VectorSim:
    """max similarity of a query image to a reference set, from precomputed
    per-image vectors. ``kind='clip'`` uses cosine; ``kind='color'`` uses
    negative L2 (so higher = more similar, matching -LPIPS)."""

    def __init__(self, vec_by_idx, kind):
        self.v = vec_by_idx
        self.kind = kind

    def max_sim(self, q, refs):
        qv = self.v.get(int(q))
        if qv is None:
            return np.nan
        best = -np.inf
        for r in refs:
            r = int(r)
            if r == q:
                continue
            rv = self.v.get(r)
            if rv is None:
                continue
            if self.kind == "clip":
                s = float(np.dot(qv, rv) /
                          ((np.linalg.norm(qv) + 1e-8) * (np.linalg.norm(rv) + 1e-8)))
            else:  # color: negative L2 distance
                s = float(-np.linalg.norm(qv - rv))
            if s > best:
                best = s
        return best if best > -np.inf else np.nan


def compute_color_vecs(pool_paths, needed, size=32):
    """Per-image spatial-average RGB (3-dim) for the requested pool indices."""
    out = {}
    for i in needed:
        try:
            im = Image.open(pool_paths[i]).convert("RGB").resize((size, size))
            out[i] = np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(0) / 255.0
        except Exception:
            pass
    return out


def compute_clip_vecs(pool_paths, needed, device, model_id, batch=64):
    """CLIP image embeddings (pooled projection) for the requested pool indices."""
    from transformers import CLIPModel, CLIPProcessor
    print(f"  loading CLIP {model_id} for high-level metric...")
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)
    out = {}
    buf_idx, buf_img = [], []

    def flush():
        if not buf_img:
            return
        inp = proc(images=buf_img, return_tensors="pt").to(device)
        with torch.inference_mode():
            # Version-robust (transformers 5.x changed get_image_features to
            # return an output object): pool the vision tower and apply the
            # CLIP visual projection to get image embeddings in the shared space.
            vis = model.vision_model(pixel_values=inp["pixel_values"])
            emb = model.visual_projection(vis.pooler_output).cpu().numpy()
        for j, e in zip(buf_idx, emb):
            out[j] = e.astype(np.float32)
        buf_idx.clear(); buf_img.clear()

    for i in needed:
        try:
            buf_img.append(Image.open(pool_paths[i]).convert("RGB"))
            buf_idx.append(i)
        except Exception:
            continue
        if len(buf_img) >= batch:
            flush()
    flush()
    return out


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Coherent-unit psychophysics across sim metrics")
    p.add_argument("--data", required=True)
    p.add_argument("--sae-path", required=True)
    p.add_argument("--backbone", default="dinov3")
    p.add_argument("--layer", type=int, default=None)        # None = final layer
    p.add_argument("--token-type", default="spatial")
    p.add_argument("--d-model", type=int, default=32000)
    p.add_argument("--top-k", type=int, default=160)
    p.add_argument("--image-dir", default=None)
    p.add_argument("--extra-image-dir", action="append", default=[])
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--pool-size", type=int, default=6000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    # coherent-unit selection
    p.add_argument("--ii-min", type=float, default=-0.42,
                   help="restrict the feature sample to II >= this (coherent units)")
    p.add_argument("--n-features-sample", type=int, default=1500)
    p.add_argument("--n-refs", type=int, default=9)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--difficulties", type=float, nargs="+",
                   default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    # similarity
    p.add_argument("--resize", type=int, default=224, help="image side for LPIPS/color")
    p.add_argument("--lpips-net", default="alex")
    p.add_argument("--lpips-batch", type=int, default=256)
    p.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    ii = data.get("interp_index")
    if ii is None:
        raise SystemExit("Primary file has no 'interp_index' (run run_interp_index.sh first).")
    ii = ii.numpy()
    freq = data["feature_frequency"].numpy()

    print("Computing per-feature activations over the image pool...")
    pool_paths, acts = compute_pool_activations(args, device)
    n_active = int((acts["max"] > 0).any(axis=0).sum())
    print(f"  features active anywhere in pool: {n_active}/{args.d_model}")
    if n_active < 0.2 * args.d_model:
        raise SystemExit(f"Activation collapse ({n_active}/{args.d_model}); check --layer.")

    # --- Coherent units: high-II, live, then sample ---
    coherent = np.where(np.isfinite(ii) & (ii >= args.ii_min) & (freq > 0))[0]
    if args.n_features_sample and args.n_features_sample < coherent.size:
        feat_sample = np.sort(rng.choice(coherent, size=args.n_features_sample, replace=False))
    else:
        feat_sample = coherent
    print(f"Coherent units: {coherent.size} with II>={args.ii_min}; "
          f"sweeping {feat_sample.size} of them.")

    # --- Build trials for EACH aggregation + gather referenced pool images ---
    trials, needed = {}, set()
    for agg in AGGREGATIONS:
        A = acts[agg]
        for f in feat_sample:
            col = A[:, f].astype(np.float32)
            for diff in args.difficulties:
                ref, pos, neg = select_psych_trials(col, args.n_refs, diff, args.n_trials, rng)
                trials[(agg, int(f), diff)] = (ref, pos, neg)
                for x in list(ref) + list(pos) + list(neg):
                    needed.add(int(x))
    needed = sorted(needed)
    print(f"  {len(trials)} trial cells ({len(AGGREGATIONS)} aggs), "
          f"{len(needed)} unique pool images referenced.")

    # --- Build the three similarity engines ---
    engines = {}
    print("Building similarity engines...")
    color_vecs = compute_color_vecs(pool_paths, needed)
    engines["color"] = VectorSim(color_vecs, "color")
    clip_vecs = compute_clip_vecs(pool_paths, needed, device, args.clip_model)
    engines["clip"] = VectorSim(clip_vecs, "clip")
    lp = LpipsSim(pool_paths, args.lpips_net, args.resize, device, args.lpips_batch)
    pair_set = set()
    for (agg, f, diff), (ref, pos, neg) in trials.items():
        for q in list(pos) + list(neg):
            for r in ref:
                pair_set.add((int(q), int(r)))
    print(f"  LPIPS (resize={args.resize}): {len(pair_set)} pairs...")
    lp.precompute_pairs(pair_set)
    engines["lpips"] = lp

    # --- Score every (aggregation x metric) over the difficulty sweep ---
    diffs = list(args.difficulties)
    accuracy = {(agg, m): np.full((feat_sample.size, len(diffs)), np.nan)
                for agg in AGGREGATIONS for m in METRICS}
    for fi, f in enumerate(feat_sample):
        for agg in AGGREGATIONS:
            for di, diff in enumerate(diffs):
                ref, pos, neg = trials[(agg, int(f), diff)]
                if not pos or not neg:
                    continue
                for m in METRICS:
                    eng = engines[m]
                    scores = [afc_correct(eng.max_sim(pq, ref), eng.max_sim(nq, ref))
                              for pq, nq in zip(pos, neg)]
                    scores = [s for s in scores if s == s]
                    if scores:
                        accuracy[(agg, m)][fi, di] = float(np.mean(scores))

    _save_and_plot(out_dir, args, feat_sample, ii, accuracy, diffs)


def _save_and_plot(out_dir, args, feat_sample, ii, accuracy, diffs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    d = np.array(diffs)
    ii_s = ii[feat_sample]

    # 3 panels (one per aggregation), each with the 3 similarity-metric curves.
    fig, axes = plt.subplots(1, len(AGGREGATIONS), figsize=(15, 4.7), sharey=True)
    print("\nDifficulty sweep on coherent units — 9 approaches (aggregation x sim metric):")
    print("   " + " ".join(f"{x:>5.2f}" for x in d) + "   easiest")
    corr = {}
    for ax, agg in zip(axes, AGGREGATIONS):
        for m in METRICS:
            acc = accuracy[(agg, m)]
            mean = np.nanmean(acc, axis=0)
            se = np.nanstd(acc, axis=0) / np.sqrt(np.maximum(np.sum(~np.isnan(acc), axis=0), 1))
            ax.errorbar(d, mean, yerr=se, marker="o", capsize=3,
                        color=METRIC_COLOR[m], label=METRIC_LABEL[m])
            acc_mean = np.nanmean(acc, axis=1)
            ok = np.isfinite(ii_s) & np.isfinite(acc_mean)
            rho, pv = spearmanr(ii_s[ok], acc_mean[ok]) if ok.sum() > 10 else (np.nan, np.nan)
            corr[(agg, m)] = {"rho": float(rho), "p": float(pv), "n": int(ok.sum())}
            print(f"  {agg:>4} x {METRIC_LABEL[m]:>12}: " +
                  " ".join(f"{x:.3f}" for x in mean) + f"   {mean[0]:.3f}")
        ax.axhline(0.5, ls="--", c="grey", label="chance")
        ax.invert_xaxis()
        ax.set_title(f"{agg} aggregation")
        ax.set_xlabel("Difficulty")
    axes[0].set_ylabel("Psychophysics accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Coherent units (II>={args.ii_min}, n={feat_sample.size}), "
                 f"resize={args.resize} — 9 approaches (3 aggregations x 3 sim metrics)")
    plt.tight_layout()
    figp = os.path.join(out_dir, "psychophysics_sim_metrics_9.png")
    plt.savefig(figp, dpi=150); plt.close()
    print(f"  wrote {figp}")

    res = os.path.join(out_dir, "psychophysics_sim_metrics_9.pt")
    torch.save({"feature_sample": feat_sample, "difficulties": d,
                "accuracy": {f"{agg}_{m}": accuracy[(agg, m)]
                             for agg in AGGREGATIONS for m in METRICS},
                "ii_corr": {f"{agg}_{m}": corr[(agg, m)]
                            for agg in AGGREGATIONS for m in METRICS},
                "args": vars(args)}, res)
    print(f"  wrote {res}")


if __name__ == "__main__":
    main()
