"""
Precompute Ising-coupling feature groups for the explorer's manifold panel.

Motivated by Bhalla, Fel et al., "Do Sparse Autoencoders Capture Concept
Manifolds?" (2026): SAEs do not encode a concept manifold in a single atom —
they *tile* it across many co-firing features. The functional signal that
recovers which atoms tile the same manifold is the pairwise **Ising coupling**
between feature activations (positive = joint "capture", negative = mutually
exclusive "tiling"), NOT decoder-direction cosine similarity (which the paper
shows fails to recover the block structure).

This script runs the backbone + SAE over the explorer's image set once and:

  1. Selects a live feature subset (drops dead / ultra-rare atoms).
  2. Accumulates Ising sufficient statistics over every spatial token —
     magnetizations m_i = <s_i> and second moments <s_i s_j> with
     s_i = 2·1[z_i > 0] − 1.
  3. Estimates couplings J via the mean-field inverse of the connected
     correlation matrix (Cocco & Monasson, 2009; cited by the paper), plus
     reconstructed fields h_i.
  4. Builds a Ward **linkage** over the coupling profiles so the explorer can
     cut the dendrogram at *any* number of clusters dynamically.
  5. Stores a per-image sample (pooled codes + mean-patch backbone features)
     so each cluster can open a PCA projection of its manifold — both the
     "project onto span(D_S)" view (paper's most faithful) and the raw
     feature-coordinate view.

Output: ``<explorer-data-without-ext>_ising.pt`` (auto-discovered by the
explorer's loader, exactly like the ``_heatmaps.pt`` sidecar).

Usage:
    python precompute_ising.py \
        --explorer-data explorer_data/explorer_data_dinov2_layer11_d10000_k100_val.pt \
        --sae-path      models/dinov2_l11_spatial/sae_1_SI-SAE_d10000_k100_per_init0.02_state_dict.pth \
        --backbone dinov2 --layer 11 --token-type spatial \
        --image-dir /scratch.global/lee02328/val \
        --extra-image-dir /scratch.global/lee02328/coco/val2017 \
        --recursive --interleave-classes
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from precompute_utils import ImageFolder, extract_tokens, load_sae  # noqa: E402


# ---------------------------------------------------------------------------
# Numeric core — factored out so it can be unit-tested / validated on cached
# activation shards without spinning up the backbone (see validate_ising.py).
# ---------------------------------------------------------------------------

def select_features(freq, min_frequency, max_features):
    """Pick the live feature subset to model.

    ``freq`` is the per-feature activation count (explorer ``feature_frequency``).
    Returns global feature indices (int64), sorted by descending frequency,
    keeping only features with ``freq >= min_frequency`` and capping the count
    at ``max_features`` (the Ising inverse is O(F^3), so F must stay bounded).
    """
    freq = np.asarray(freq, dtype=np.float64)
    live = np.where(freq >= min_frequency)[0]
    order = live[np.argsort(-freq[live])]
    if max_features is not None and len(order) > max_features:
        order = order[:max_features]
    return np.sort(order).astype(np.int64)


class IsingAccumulator:
    """Stream binary spin statistics over tokens for a fixed feature subset.

    For each token we form spins s in {-1,+1}^F from the (already subset-
    restricted) code z, and accumulate the count, the spin sum S1 = Σ s, and
    the second-moment matrix S2 = Σ s sᵀ. These are the sufficient statistics
    for the magnetizations and connected correlations.
    """

    def __init__(self, n_features, device):
        self.F = n_features
        self.device = device
        self.n = 0
        self.S1 = torch.zeros(n_features, dtype=torch.float64, device=device)
        self.S2 = torch.zeros(n_features, n_features, dtype=torch.float64, device=device)

    def update(self, z_subset):
        """``z_subset``: (n_tokens, F) code restricted to the modelled features."""
        spins = torch.where(z_subset > 0,
                            torch.ones_like(z_subset),
                            -torch.ones_like(z_subset)).to(torch.float64)
        self.n += spins.shape[0]
        self.S1 += spins.sum(dim=0)
        self.S2 += spins.t() @ spins

    def finalize(self, ridge=1e-2):
        """Return (m, C, J, h).

        m : (F,)   magnetizations <s_i>
        C : (F,F)  connected correlation <s_i s_j> − m_i m_j (ridge-regularized)
        J : (F,F)  mean-field couplings  J = −C⁻¹  (zero diagonal)
        h : (F,)   reconstructed fields  h_i = atanh(m_i) − Σ_j J_ij m_j
        """
        if self.n == 0:
            raise RuntimeError("IsingAccumulator.finalize called with no samples")
        m = (self.S1 / self.n)
        C = (self.S2 / self.n) - torch.outer(m, m)
        # Ridge toward the diagonal keeps C invertible when some features are
        # near-deterministic or sample counts are modest.
        C = C + ridge * torch.eye(self.F, dtype=C.dtype, device=C.device)
        Cinv = torch.linalg.inv(C)
        J = -Cinv
        J.fill_diagonal_(0.0)
        J = 0.5 * (J + J.t())  # symmetrize away numerical asymmetry
        m_clamped = torch.clamp(m, -0.999, 0.999)
        h = torch.atanh(m_clamped) - J @ m
        return (m.cpu().numpy(), C.cpu().numpy(),
                J.cpu().numpy(), h.cpu().numpy())


def build_linkage(J, method='ward'):
    """Hierarchical linkage over coupling profiles → dendrogram for dynamic K.

    Each feature is represented by its signed coupling row J_i,:; Ward linkage
    on the Euclidean distance between rows groups features that interact
    similarly with the population (the functional definition of a shared
    manifold). The returned ((F-1), 4) matrix lets the explorer call
    ``scipy.cluster.hierarchy.fcluster`` at any cut height instantly.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    d = pdist(J.astype(np.float64), metric='euclidean')
    return linkage(d, method=method)


def accumulate_from_shards(acc, shard_paths, sae, feat_idx_t, device, chunk):
    """Stream cached SAE-input token shards through the encoder into ``acc``.

    Each shard is a (N, d_hidden) fp16 tensor of the exact features the SAE was
    trained on (raw layer-11 tokens here), so we just chunk it through the SAE
    and feed the subset codes to the Ising accumulator. This is the full-
    trainset coupling estimate — far more tokens than an image pass, and it
    needs no backbone.
    """
    for si, sp in enumerate(shard_paths):
        toks = torch.load(sp, map_location='cpu', weights_only=False)
        n = toks.shape[0]
        with torch.inference_mode():
            for start in range(0, n, chunk):
                blk = toks[start:start + chunk].to(device, dtype=torch.float32,
                                                   non_blocking=True)
                _, z, _ = sae(blk)
                acc.update(z.index_select(1, feat_idx_t))
        del toks
        print(f"  shard {si + 1}/{len(shard_paths)}: {acc.n} tokens", flush=True)
    return acc


# ---------------------------------------------------------------------------
# Backbone streaming pass (the actual precompute job)
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--explorer-data", required=True,
                   help="Existing explorer_data*.pt sidecar (for image_paths, "
                        "feature_frequency, and the decoder).")
    p.add_argument("--sae-path", required=True)
    p.add_argument("--output", default=None,
                   help="Output path (default: <explorer-data>_ising.pt).")
    p.add_argument("--backbone", default="dinov2")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--token-type", default="spatial")
    p.add_argument("--image-dir", required=True,
                   help="Image dir streamed through the backbone for the "
                        "per-image manifold sample (and for the coupling "
                        "estimate when --token-shards is not given).")
    p.add_argument("--extra-image-dir", default=None)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--token-shards", default=None,
                   help="Glob of cached SAE-input token shards "
                        "(e.g. '.../layer_11/shard_*.pt', each (N, d_hidden) "
                        "fp16). When set, the Ising couplings are estimated by "
                        "streaming ALL of these through the SAE encoder — no "
                        "backbone needed and orders of magnitude more tokens "
                        "than an image pass. The backbone still runs over "
                        "--image-dir only to collect the per-image sample.")
    p.add_argument("--shard-chunk", type=int, default=32768,
                   help="Rows per SAE forward when streaming token shards.")
    p.add_argument("--patch-norm-stats", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--d-model", type=int, default=None,
                   help="SAE width (default: read from explorer data).")
    p.add_argument("--top-k", type=int, default=None,
                   help="SAE top-k (default: parsed from the SAE filename).")
    # Ising / clustering knobs
    p.add_argument("--min-frequency", type=float, default=200,
                   help="Drop features that fire in fewer than this many tokens.")
    p.add_argument("--max-features", type=int, default=4000,
                   help="Cap the modelled feature count (Ising inverse is O(F^3)).")
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="Diagonal ridge added to the correlation matrix before inversion.")
    p.add_argument("--linkage-method", default="ward")
    p.add_argument("--sample-images", type=int, default=4000,
                   help="Images to store pooled codes + backbone feats for "
                        "(drives the per-cluster PCA manifold view).")
    p.add_argument("--max-images", type=int, default=None,
                   help="Process at most this many images (for quick test runs).")
    return p.parse_args(argv)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load explorer sidecar (image alignment + frequency + decoder) ---
    print(f"Loading explorer data: {args.explorer_data}")
    ed = torch.load(args.explorer_data, map_location='cpu', weights_only=False)
    image_paths_ref = ed['image_paths']
    d_model = args.d_model or int(ed['d_model'])
    freq = ed['feature_frequency'].numpy() if torch.is_tensor(ed['feature_frequency']) \
        else np.asarray(ed['feature_frequency'])
    decoder = ed.get('dictionary')
    if decoder is not None and torch.is_tensor(decoder):
        decoder = decoder.float().numpy()          # (d_model, d_hidden)

    feat_idx = select_features(freq, args.min_frequency, args.max_features)
    F = len(feat_idx)
    print(f"Modelling {F} / {d_model} features "
          f"(freq >= {args.min_frequency}, cap {args.max_features})")
    feat_idx_t = torch.from_numpy(feat_idx).to(device)

    # --- Backbone + SAE ---
    from backbone_runners import load_batched_backbone
    from precompute_utils import parse_top_k_from_path
    print(f"Loading backbone={args.backbone} layer={args.layer} ...")
    get_hidden, d_brain, n_reg, transform = load_batched_backbone(
        args.backbone, args.layer, device)
    top_k = args.top_k or parse_top_k_from_path(args.sae_path)
    sae = load_sae(args.sae_path, d_brain, d_model, top_k, device)
    print(f"  SAE: d_model={d_model}, top_k={top_k}, d_hidden={d_brain}")

    patch_norm_mean = patch_norm_std = None
    if args.patch_norm_stats:
        st = torch.load(args.patch_norm_stats, map_location=device, weights_only=True)
        patch_norm_mean = st['positional_mean'].float().to(device)
        patch_norm_std = st['std'].float().to(device)

    # --- Dataset ---
    # Map processed images back to their index in the explorer's image_paths
    # by basename, so the per-image sample stays aligned even if the image
    # directories or their ordering have changed since the sidecar was built.
    from torch.utils.data import DataLoader
    name_to_global = {os.path.basename(p): i for i, p in enumerate(image_paths_ref)}
    roots = [args.image_dir] + ([args.extra_image_dir] if args.extra_image_dir else [])
    dataset = ImageFolder(roots, recursive=args.recursive, transform_fn=transform)
    print(f"Found {len(dataset.paths)} images across {len(roots)} dir(s); "
          f"explorer references {len(image_paths_ref)} images.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    acc = IsingAccumulator(F, device)
    coupling_from_shards = bool(args.token_shards)
    if coupling_from_shards:
        shard_paths = sorted(glob.glob(args.token_shards))
        if not shard_paths:
            raise SystemExit(f"--token-shards matched no files: {args.token_shards}")
        print(f"Estimating couplings from {len(shard_paths)} token shard(s) "
              f"through the SAE encoder (chunk={args.shard_chunk})...")
        accumulate_from_shards(acc, shard_paths, sae, feat_idx_t, device,
                               args.shard_chunk)

    n_sample = min(args.sample_images, len(dataset.paths))
    sample_codes = np.zeros((n_sample, F), dtype=np.float16)
    sample_feats = np.zeros((n_sample, d_brain), dtype=np.float16)
    sample_idx = np.full(n_sample, -1, dtype=np.int64)
    n_stored = 0

    if coupling_from_shards:
        print(f"Backbone pass over images only to collect up to {n_sample} "
              f"manifold-sample images (couplings already estimated)...")
    else:
        print(f"Streaming tokens (sampling up to {n_sample} images for the manifold view)...")
    n_images_seen = 0
    with torch.inference_mode():
        for bi, (imgs, paths) in enumerate(loader):
            bs = imgs.shape[0]
            imgs = imgs.to(device, non_blocking=True)
            hidden = get_hidden(imgs)
            tokens = extract_tokens(hidden, args.backbone, args.token_type, n_reg)
            if patch_norm_mean is not None:
                tokens = (tokens - patch_norm_mean) / (patch_norm_std + 1e-8)
            n_patches = tokens.shape[1]
            flat = tokens.reshape(-1, d_brain)
            _, z, _ = sae(flat)                                  # (bs*n_patches, d_model)
            z_sub = z.index_select(1, feat_idx_t)                # (bs*n_patches, F)
            if not coupling_from_shards:
                acc.update(z_sub)

            # Per-image sample for the PCA manifold view, keyed to the
            # explorer's global image index so the filmstrip is clickable.
            z_img = z_sub.reshape(bs, n_patches, F)
            pooled = z_img.amax(dim=1)                           # (bs, F) max over patches
            mean_feat = tokens.mean(dim=1)                       # (bs, d_hidden)
            for i in range(bs):
                gi = name_to_global.get(os.path.basename(paths[i]))
                if gi is not None and n_stored < n_sample:
                    sample_codes[n_stored] = pooled[i].cpu().numpy().astype(np.float16)
                    sample_feats[n_stored] = mean_feat[i].cpu().numpy().astype(np.float16)
                    sample_idx[n_stored] = gi
                    n_stored += 1
            n_images_seen += bs

            if (bi + 1) % 25 == 0:
                print(f"  {n_images_seen} images, {acc.n} tokens, "
                      f"{n_stored} sampled", flush=True)
            # In shard mode the couplings are already done; once the manifold
            # sample is full there is nothing left to gain from more images.
            if coupling_from_shards and n_stored >= n_sample:
                break
            if args.max_images is not None and n_images_seen >= args.max_images:
                break

    print(f"Estimating couplings from {acc.n} tokens (ridge={args.ridge})...")
    m, C, J, h = acc.finalize(ridge=args.ridge)
    print(f"  coupling stats: |J| mean={np.abs(J).mean():.4g} max={np.abs(J).max():.4g}, "
          f"positive fraction={(J > 0).sum() / (J.size - F):.3f}")
    print(f"Building {args.linkage_method} linkage over coupling profiles...")
    Z = build_linkage(J, method=args.linkage_method)

    # Trim sample to the rows we actually filled (short test runs).
    valid = sample_idx >= 0
    out = {
        'ising_feature_indices': torch.from_numpy(feat_idx),                # (F,) global ids
        'ising_couplings':       torch.from_numpy(J.astype(np.float16)),    # (F,F)
        'ising_fields':          torch.from_numpy(h.astype(np.float32)),    # (F,)
        'ising_magnetization':   torch.from_numpy(m.astype(np.float32)),    # (F,)
        'ising_linkage':         torch.from_numpy(Z),                       # (F-1,4)
        'sample_image_indices':  torch.from_numpy(sample_idx[valid]),       # (S,)
        'sample_codes':          torch.from_numpy(sample_codes[valid]),     # (S,F)
        'sample_backbone_feats': torch.from_numpy(sample_feats[valid]),     # (S,d_hidden)
        'n_tokens':              acc.n,
        'min_frequency':         args.min_frequency,
        'max_features':          args.max_features,
        'ridge':                 args.ridge,
        'linkage_method':        args.linkage_method,
        'd_model':               d_model,
        'd_hidden':              d_brain,
    }
    if decoder is not None:
        out['decoder_subset'] = torch.from_numpy(decoder[feat_idx].astype(np.float32))  # (F,d_hidden)

    out_path = args.output or (os.path.splitext(args.explorer_data)[0] + '_ising.pt')
    torch.save(out, out_path)
    print(f"Saved Ising sidecar → {out_path}  "
          f"(F={F}, sample={int(valid.sum())} images)")


if __name__ == "__main__":
    main()
