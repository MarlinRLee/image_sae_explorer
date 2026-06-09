"""Signed Ising coupling matrices under each grouping K (paper Fig. 17 / 4C).

Within a *single* SAE, the pairwise Ising couplings J_ij over binarized codes
capture the co-activation structure between features (Bhalla, Fel et al. 2026,
Sec 4.1). Reordering J so that the atoms of each Ising group are contiguous makes
the grouping show up as block-diagonal structure:

    red  (J > 0)  atoms co-fire        -> "capture" block
    blue (J < 0)  atoms mutually exclude -> "shattering" block
    mixed sign within a block            -> "dilution" (no coherent group)

This script reads the ``_ising.pt`` sidecar and, for a sweep of cluster counts K
(cuts of the precomputed dendrogram), plots the reordered *signed* J with the K
block boundaries overlaid. A grouping is visually "good" at the K where the
blocks are tight and sign-coherent (clean red or clean blue squares on the
diagonal) rather than smeared or fragmented -- exactly the low-K (shattered) /
intermediate-K (clean blocks) / high-K (over-fragmented) progression of Fig. 17.

Because most of the modeled atoms couple only weakly (and universal atoms wash
the picture out, Sec 4), the matrix is restricted to the most strongly-coupled
atoms by default (``--top-atoms``), which concentrates the block signal the way
the synthetic 512-atom panels do. Use ``--top-atoms 0`` for the full matrix.

Usage:
    python scripts/plot_ising_grouping_matrices.py \
        --ising explorer_data/explorer_data_dinov2_layer11_d10000_k100_val_ising.pt \
        --k 8 16 32 64 128 --top-atoms 800
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, leaves_list

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _as_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def block_boundaries(labels_in_order):
    """Positions where the cluster label changes along an ordering."""
    return np.where(np.diff(labels_in_order) != 0)[0] + 1


def signed_cohesion(J_sub, labels_sub):
    """Median over blocks of the signed cohesion rho(G) (paper Defn. 6).

    rho(G) = mean sign(J_ab) over off-diagonal pairs in a block; +1 = all
    co-fire (capture), -1 = all mutually exclude (shattering), ~0 = mixed
    (dilution). Returns the median |rho| across blocks with >= 2 atoms -- a
    coarse "how sign-coherent are the groups" score (higher = cleaner).
    """
    vals = []
    for c in np.unique(labels_sub):
        idx = np.where(labels_sub == c)[0]
        if len(idx) < 2:
            continue
        blk = J_sub[np.ix_(idx, idx)]
        off = ~np.eye(len(idx), dtype=bool)
        s = np.sign(blk[off])
        if s.size:
            vals.append(abs(s.mean()))
    return float(np.median(vals)) if vals else float("nan")


def mag_cohesion(blk, n):
    """Magnitude-weighted signed cohesion  sum(J) / sum(|J|)  over off-diag.

    Unlike the bare-sign rho, strong couplings dominate, so a group with a few
    big positive couplings and many weak noisy-sign ones still reads as coherent.
    """
    off = blk[~np.eye(n, dtype=bool)]
    denom = np.abs(off).sum()
    return float(off.sum() / denom) if denom else 0.0


def spectral_gap(codes_sub, maxdim):
    """Largest consecutive eigenvalue ratio in a group's code-PCA spectrum.

    The paper validates a candidate manifold group by a *sharp PCA spectral
    gap* in its code vectors (App. E): a clear drop separates a few signal
    components from the noise floor, indicating genuine low-dimensional
    structure. Returns (gap, intrinsic_dim) where gap = max_i lambda_i/lambda_{i+1}
    over the leading ``maxdim`` components.
    """
    X = codes_sub.astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    if X.shape[1] < 2 or not np.any(X):
        return 0.0, 1
    s = np.linalg.svd(X, full_matrices=False)[1]
    lam = s[s > 0] ** 2
    if len(lam) < 2:
        return 0.0, 1
    upto = min(len(lam) - 1, maxdim)
    ratios = lam[:upto] / lam[1:upto + 1]
    d = int(np.argmax(ratios))
    return float(ratios[d]), d + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ising", required=True, help="_ising.pt sidecar path")
    ap.add_argument("--k", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--top-atoms", type=int, default=800,
                    help="restrict to this many most strongly-coupled atoms "
                         "(0 = use all modeled atoms)")
    ap.add_argument("--clip-pct", type=float, default=98.0,
                    help="percentile of |J| for the symmetric color limit")
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--min-block-size", type=int, default=3,
                    help="only draw boundaries around blocks with >= this many "
                         "atoms (avoids the line-wash at high K)")
    ap.add_argument("--zoom-k", type=int, default=64,
                    help="K at which to also render a per-cluster zoom of the "
                         "largest groups (0 = skip)")
    ap.add_argument("--zoom-n", type=int, default=12,
                    help="number of largest groups to show in the zoom panel")
    ap.add_argument("--output", default="figures/ising_coupling_by_grouping.png")
    ap.add_argument("--curve", action="store_true",
                    help="also sweep K and plot group-quality vs K")
    ap.add_argument("--curve-k", type=int, nargs="+",
                    default=[4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                             384, 512, 768, 1000],
                    help="K values for the quality-vs-K curve")
    ap.add_argument("--coh-tau", type=float, default=0.7,
                    help="|signed cohesion| threshold for a group to count as "
                         "coherent (capture if >= +tau)")
    ap.add_argument("--coh-min", type=int, default=5,
                    help="min group size for the coherence curve (guards against "
                         "tiny groups being coherent by chance)")
    ap.add_argument("--gap-thresh", type=float, default=3.0,
                    help="min PCA eigenvalue-ratio for a group to pass spectral-"
                         "gap validation")
    ap.add_argument("--gap-maxdim", type=int, default=12,
                    help="leading components searched for the spectral gap")
    args = ap.parse_args()

    print(f"loading {args.ising} ...")
    ising = torch.load(args.ising, map_location="cpu", weights_only=False)
    Z = _as_np(ising["ising_linkage"])
    J = _as_np(ising["ising_couplings"]).astype(np.float32)
    np.fill_diagonal(J, 0.0)
    n = J.shape[0]
    print(f"  J: {J.shape}, linkage: {Z.shape}")

    # Optionally restrict to the most strongly-coupled atoms (concentrates the
    # block signal; universal / weakly-coupled atoms otherwise wash it out).
    strength = np.abs(J).sum(1)
    if args.top_atoms and args.top_atoms < n:
        keep = np.sort(np.argsort(-strength)[:args.top_atoms])
        sub = "top-%d coupled atoms" % args.top_atoms
    else:
        keep = np.arange(n)
        sub = "all %d atoms" % n
    print(f"  using {len(keep)} atoms ({sub})")

    # Dendrogram leaf order, then restrict to kept atoms preserving that order.
    leaf_order = leaves_list(Z)
    order = leaf_order[np.isin(leaf_order, keep)]
    J_ord = J[np.ix_(order, order)]

    vlim = float(np.percentile(np.abs(J_ord[J_ord != 0]), args.clip_pct)) or 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    def draw_boundaries(ax, labels_in_order):
        """Draw block separators, but only around blocks >= min-block-size."""
        labs = labels_in_order
        edges = [0, *block_boundaries(labs), len(labs)]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a >= args.min_block_size:
                for p in (a, b):
                    ax.axhline(p - 0.5, color="k", lw=0.5, alpha=0.6)
                    ax.axvline(p - 0.5, color="k", lw=0.5, alpha=0.6)

    ks = args.k
    ncols = min(args.ncols, len(ks))
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 5.4 * nrows),
                             squeeze=False)

    for ax, K in zip(axes.ravel(), ks):
        labels_full = fcluster(Z, int(K), criterion="maxclust")
        labels_ord = labels_full[order]
        n_blocks = len(np.unique(labels_ord))
        coh = signed_cohesion(J_ord, labels_ord)

        im = ax.imshow(J_ord, cmap="RdBu_r", norm=norm,
                       interpolation="nearest", aspect="equal")
        draw_boundaries(ax, labels_ord)
        ax.set_title(f"K = {K}   ({n_blocks} groups)\n"
                     f"median block sign-coherence = {coh:.2f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes.ravel()[len(ks):]:
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    cbar.set_label("Ising coupling  J   (red = co-fire,  blue = mutual exclusion)")
    fig.suptitle(f"Signed SAE feature coupling reordered by Ising grouping "
                 f"({sub}, leaf-ordered)\n"
                 f"clean block-diagonal sign structure = good grouping",
                 fontsize=14)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    # ---- Per-cluster zoom: the largest groups at a single K -----------------
    if args.zoom_k:
        labels_full = fcluster(Z, int(args.zoom_k), criterion="maxclust")
        ids, counts = np.unique(labels_full, return_counts=True)
        big = ids[np.argsort(-counts)][:args.zoom_n]
        zc = min(4, len(big))
        zr = int(np.ceil(len(big) / zc))
        figz, axesz = plt.subplots(zr, zc, figsize=(3.0 * zc, 3.2 * zr),
                                   squeeze=False)
        for ax, cid in zip(axesz.ravel(), big):
            mem = np.where(labels_full == cid)[0]
            mem = leaf_order[np.isin(leaf_order, mem)]  # dendrogram order
            blk = J[np.ix_(mem, mem)]
            off = blk[~np.eye(len(mem), dtype=bool)]
            pos = float((off > 0).mean()) if off.size else 0.0
            if pos >= 0.70:
                reg = "capture"
            elif pos <= 0.35:
                reg = "shatter"
            else:
                reg = "dilution"
            vl = float(np.percentile(np.abs(off), 98)) if off.size else 1.0
            vl = vl or 1.0
            ax.imshow(blk, cmap="RdBu_r",
                      norm=TwoSlopeNorm(vmin=-vl, vcenter=0, vmax=vl),
                      interpolation="nearest")
            ax.set_title(f"{len(mem)} atoms · {reg}\n{pos:.0%} positive",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axesz.ravel()[len(big):]:
            ax.axis("off")
        figz.suptitle(f"Within-group signed coupling — {args.zoom_n} largest "
                      f"groups at K={args.zoom_k}\n"
                      f"(uniform red=capture, uniform blue=shatter, "
                      f"mixed=dilution)", fontsize=12)
        outz = out.with_name(out.stem + f"_zoom_k{args.zoom_k}" + out.suffix)
        figz.savefig(outz, dpi=150, bbox_inches="tight")
        print(f"wrote {outz}")

    # ---- Group-quality vs K curve (computed over ALL modeled atoms) ---------
    if args.curve:
        tau, mg = args.coh_tau, args.coh_min
        gthr, gmax = args.gap_thresh, args.gap_maxdim
        codes = _as_np(ising["sample_codes"]).astype(np.float32) \
            if "sample_codes" in ising else None
        if codes is None:
            print("  (no sample_codes -> skipping spectral-gap validation)")
        ks_c = sorted(args.curve_k)
        # Three nested criteria, increasingly strict:
        #   A sign-capture      rho_sign >= tau
        #   B + magnitude-coh   rho_mag  >= tau
        #   C + spectral gap    code-PCA gap >= gthr
        largest, fracA, fracB, fracC, nA, nC = [], [], [], [], [], []
        for K in ks_c:
            labels = fcluster(Z, int(K), criterion="maxclust")
            ids, counts = np.unique(labels, return_counts=True)
            largest.append(counts.max() / n)
            aA = aB = aC = cA = cC = 0
            for c, cnt in zip(ids, counts):
                if cnt < mg:
                    continue
                idx = np.where(labels == c)[0]
                blk = J[np.ix_(idx, idx)].astype(np.float64)
                rho_s = np.sign(blk[~np.eye(cnt, dtype=bool)]).mean()
                if rho_s < tau:                 # not a sign-capture group
                    continue
                aA += cnt; cA += 1
                if mag_cohesion(blk, cnt) < tau:
                    continue
                aB += cnt
                if codes is not None:
                    gap, _ = spectral_gap(codes[:, idx], gmax)
                    if gap < gthr:
                        continue
                aC += cnt; cC += 1
            fracA.append(aA / n); fracB.append(aB / n); fracC.append(aC / n)
            nA.append(cA); nC.append(cC)
            print(f"  K={K:5d}  blob={largest[-1]:.3f}  signCap={fracA[-1]:.3f}"
                  f"  +magCoh={fracB[-1]:.3f}  +specGap={fracC[-1]:.3f}"
                  f"  (nCap {cA}->{cC})")

        figc, ax1 = plt.subplots(figsize=(8.5, 5))
        ax1.plot(ks_c, fracA, "o-", color="#9ca3af",
                 label=f"sign-capture  (ρ_sign≥{tau}, size≥{mg})")
        ax1.plot(ks_c, fracB, "s-", color="#2563eb",
                 label=f"+ magnitude-coherent  (ρ_mag≥{tau})")
        if codes is not None:
            ax1.plot(ks_c, fracC, "D-", color="#1a7f37",
                     label=f"+ PCA spectral-gap valid  (gap≥{gthr})")
        ax1.set_xlabel("K (number of Ising groupings)")
        ax1.set_ylabel("fraction of modeled atoms in qualifying groups")
        ax1.set_xscale("log")
        ax1.set_xticks(ks_c); ax1.set_xticklabels(ks_c, rotation=45, fontsize=8)
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(ks_c, largest, "^:", color="#b91c1c",
                 label="largest cluster (dilution blob)")
        ax2.set_ylabel("largest cluster / all atoms", color="#b91c1c")
        ax2.tick_params(axis="y", labelcolor="#b91c1c")

        final = fracC if codes is not None else fracB
        best = ks_c[int(np.argmax(final))]
        ax1.axvline(best, color="#1a7f37", ls=":", lw=1, alpha=0.6)
        l1, b1 = ax1.get_legend_handles_labels()
        l2, b2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, b1 + b2, fontsize=8, loc="upper left")
        figc.suptitle(f"Ising-group quality vs K under stricter validators "
                      f"(co-activation, {n} atoms)\n"
                      f"validated-capture coverage peaks at K≈{best}", fontsize=12)
        outc = out.with_name(out.stem + "_quality_curve" + out.suffix)
        figc.savefig(outc, dpi=150, bbox_inches="tight")
        print(f"wrote {outc}")


if __name__ == "__main__":
    main()
