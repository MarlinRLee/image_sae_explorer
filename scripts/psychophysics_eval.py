"""
In-silico psychophysics validation of the Interpretability Index, plus a
2AFC separability ("diversity") metric — Klindt et al. (2023).

Follows Klindt et al., 2023 (Sections 2.3 / 3.1 / 3.3 and App. D).

What this computes
------------------
1. IN-SILICO PSYCHOPHYSICS ACCURACY (Eq. 2-3). For an SAE feature u with top-9
   reference MEIs x_1..x_9, the model decides a 2-alternative-forced-choice
   trial (a positive query that activates u vs. a negative query that does not)
   by picking whichever query is most similar to the reference set:

       sim(x, MEI(u)) = max_{k=1..9} sim(x, x_k),   sim = -LPIPS

   Acc(u) = fraction of trials the positive query is chosen. We sweep task
   DIFFICULTY (paper Fig. 5): at difficulty 1.0 the positive query is drawn
   from the strongest activations (easy); toward 0.5 it is drawn from
   progressively weaker (more central) activations (hard). Higher II should
   track higher accuracy — that correlation is the validation.

2. AGGREGATION COMPARISON. A feature's per-image activation can be summarized
   three ways (as in precompute_explorer_data.py): MAX patch, MEAN patch, and
   CROP-MEAN (mean of the top-CROP_K positive patches). We rank/select MEIs and
   run the whole psychophysics sweep under each, to see which aggregation yields
   more interpretable / separable features.

3. DIVERSITY (2AFC SEPARABILITY, App. D cross-feature idea). For a pair of
   features (A, B), build a reference set that is half A's MEIs and half B's,
   then ask whether a held-out A-image and a held-out B-image can be sorted
   back to the correct half by max LPIPS-similarity. Pair separability = how
   cleanly A and B separate; a feature's diversity = mean separability against
   its partners. Computed only over features with II above a threshold (the
   interpretable ones are the ones worth asking about), per the requested scope.

Faithful sweep: this RE-RUNS the backbone + SAE over an image pool to get
per-feature activation rankings (we don't store minimally-activating images),
so it needs the SAE .pth weights. Heavy — run via sbatch (run_psychophysics.sh).

Outputs (to --out-dir, default ../figures):
    psychophysics_difficulty_by_aggregation.png   accuracy vs difficulty, 3 curves
    ii_vs_psychophysics_<agg>.png                 validation scatter + Spearman
    diversity_hist_<agg>.png                       per-feature diversity distribution
    psychophysics_results.pt                       all per-feature arrays + metadata
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from backbone_runners import load_batched_backbone
from precompute_utils import RESOLUTION, INPUT_SIZE, ImageFolder, extract_tokens, load_sae

CROP_K = 8                      # must match precompute_explorer_data.py
AGGREGATIONS = ("max", "mean", "crop")
DIFFICULTIES_DEFAULT = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


# ===========================================================================
# Pure logic (unit-tested in __main__ self-test; no torch/LPIPS needed)
# ===========================================================================

def select_psych_trials(acts, n_refs, difficulty, n_trials, rng,
                        band=0.05, neg_frac=0.10):
    """Pick reference + (positive, negative) query indices for one feature.

    acts        : 1D array of the feature's activation over the image pool.
    n_refs      : size of the reference MEI set (top activations).
    difficulty  : in (0.5, 1.0]; the positive query's target activation
                  quantile (1.0 = strongest/easy, 0.5 = median/hard).
    band        : half-width of the quantile window the positive is sampled from.
    neg_frac    : negatives are sampled from the bottom this-fraction of acts
                  (near-non-activating images).

    Returns (ref_idx, pos_idx[list], neg_idx[list]); pos/neg disjoint from refs.
    Returns (ref_idx, [], []) when the feature can't supply a clean trial.
    """
    order = np.argsort(-acts)                       # strongest first
    ref_idx = order[:n_refs]
    pos_pool_order = order[n_refs:]                  # candidates exclude refs
    pos_acts = acts[pos_pool_order]
    # Positive candidates must actually activate the feature.
    active = pos_pool_order[pos_acts > 0]
    if active.size == 0:
        return ref_idx, [], []

    # Quantile window over the ACTIVE candidates (rank 1.0 = strongest).
    n_act = active.size
    ranks = 1.0 - np.arange(n_act) / max(n_act - 1, 1)   # active is already desc
    lo, hi = difficulty - band, difficulty + band
    in_band = active[(ranks >= lo) & (ranks <= hi)]
    if in_band.size == 0:                                # nearest fallback
        in_band = active[[int(np.argmin(np.abs(ranks - difficulty)))]]

    # Negatives: bottom neg_frac of the pool (weakest activations).
    n_neg_pool = max(int(len(acts) * neg_frac), 1)
    neg_candidates = order[-n_neg_pool:]

    pos_idx = rng.choice(in_band, size=n_trials,
                         replace=in_band.size < n_trials).tolist()
    neg_idx = rng.choice(neg_candidates, size=n_trials,
                         replace=neg_candidates.size < n_trials).tolist()
    return ref_idx, pos_idx, neg_idx


def afc_correct(pos_score, neg_score):
    """2AFC: the model is correct when the positive query is more similar to
    the reference set than the negative. Ties count as 0.5 (chance)."""
    if pos_score > neg_score:
        return 1.0
    if pos_score < neg_score:
        return 0.0
    return 0.5


def separability(qA_to_A, qA_to_B, qB_to_A, qB_to_B):
    """2AFC separability for a feature pair: a query is assigned to whichever
    half (A or B) its max-similarity is higher. Returns fraction of the two
    queries assigned to their true source (1.0, 0.5, or 0.0)."""
    a_ok = afc_correct(qA_to_A, qA_to_B)     # A-query should prefer A-half
    b_ok = afc_correct(qB_to_B, qB_to_A)     # B-query should prefer B-half
    return 0.5 * (a_ok + b_ok)


# ===========================================================================
# LPIPS similarity over pooled images (lazy tensor cache + batched pairs)
# ===========================================================================

class LpipsSim:
    """max LPIPS-similarity (= -LPIPS distance) of a query image to a set of
    reference images, over the shared image pool. Image tensors are loaded and
    cached on demand; pairwise distances are computed in batches and memoized
    by unordered index pair so reused (query, ref) pairs are never recomputed."""

    def __init__(self, pool_paths, net, resize, device, batch_size=512):
        import lpips
        self.paths = pool_paths
        self.loss_fn = lpips.LPIPS(net=net).to(device).eval()
        self.resize = resize
        self.device = device
        self.batch_size = batch_size
        self._img = {}          # pool idx -> (3,H,W) tensor in [-1,1] or None
        self._dist = {}         # frozenset({i,j}) -> float LPIPS distance

    def _tensor(self, idx):
        if idx not in self._img:        # not attempted yet
            from PIL import Image
            try:
                im = (Image.open(self.paths[idx]).convert("RGB")
                      .resize((self.resize, self.resize), Image.Resampling.BILINEAR))
                arr = np.asarray(im, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
            except Exception:
                t = None
            self._img[idx] = t
        return self._img[idx]

    def precompute_pairs(self, pairs):
        """Compute and cache LPIPS for an iterable of (i, j) pool-index pairs."""
        todo = []
        for i, j in pairs:
            key = frozenset((i, j))
            if i == j or key in self._dist:
                continue
            if self._tensor(i) is None or self._tensor(j) is None:
                self._dist[key] = np.nan
            else:
                todo.append((i, j))
        for s in range(0, len(todo), self.batch_size):
            chunk = todo[s:s + self.batch_size]
            a = torch.stack([self._img[i] for i, _ in chunk]).to(self.device)
            b = torch.stack([self._img[j] for _, j in chunk]).to(self.device)
            with torch.inference_mode():
                d = self.loss_fn(a, b).view(-1).cpu().numpy()
            for (i, j), dist in zip(chunk, d):
                self._dist[frozenset((i, j))] = float(dist)

    def max_sim(self, query, refs):
        """max over refs of -LPIPS(query, ref); NaN refs ignored."""
        best = -np.inf
        for r in refs:
            if r == query:
                continue
            dist = self._dist.get(frozenset((query, r)), np.nan)
            if dist == dist:                      # not NaN
                best = max(best, -dist)
        return best if best > -np.inf else np.nan


# ===========================================================================
# Backbone + SAE pass: per-feature activations over the pool, 3 aggregations
# ===========================================================================

def compute_pool_activations(args, device):
    """Run backbone+SAE over the image pool; return (pool_paths, acts) where
    acts is {agg: float16 array (n_pool, d_model)} for max/mean/crop."""
    from torch.utils.data import DataLoader

    _get_hidden, d_brain, n_reg, _transform = load_batched_backbone(
        args.backbone, args.layer, device)
    sae = load_sae(args.sae_path, d_brain, args.d_model, args.top_k, device)
    print(f"  backbone={args.backbone} layer={args.layer} d_brain={d_brain} "
          f"n_reg={n_reg}; SAE d_model={args.d_model} top_k={args.top_k}")

    roots = [d for d in ([args.image_dir] + args.extra_image_dir) if d]
    dataset = ImageFolder(roots, recursive=args.recursive, transform_fn=_transform)
    if args.pool_size and len(dataset.paths) > args.pool_size:
        # Deterministic evenly-spaced subsample across the corpus.
        step = len(dataset.paths) / args.pool_size
        keep = [dataset.paths[int(i * step)] for i in range(args.pool_size)]
        dataset.paths = keep
    n_pool = len(dataset.paths)
    print(f"  image pool: {n_pool} images across {len(roots)} dir(s)")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    acts = {a: np.zeros((n_pool, args.d_model), dtype=np.float16) for a in AGGREGATIONS}
    pool_paths, off = [], 0
    with torch.inference_mode():
        for imgs, paths in loader:
            pool_paths.extend(list(paths))
            bs = imgs.shape[0]
            hidden = _get_hidden(imgs.to(device, non_blocking=True))
            tokens = extract_tokens(hidden, args.backbone, args.token_type, n_reg)
            n_patches = tokens.shape[1]
            z = sae(tokens.reshape(-1, d_brain))[1].reshape(bs, n_patches, args.d_model)
            zc = z.cpu().numpy()                                  # (bs, P, d)
            acts["max"][off:off + bs]  = zc.max(axis=1).astype(np.float16)
            acts["mean"][off:off + bs] = zc.mean(axis=1).astype(np.float16)
            k = min(CROP_K, n_patches)
            topk = np.partition(zc, n_patches - k, axis=1)[:, n_patches - k:, :]
            pos = np.maximum(topk, 0.0)
            cnt = (topk > 0).sum(axis=1)
            acts["crop"][off:off + bs] = np.where(
                cnt > 0, pos.sum(axis=1) / np.maximum(cnt, 1), 0.0).astype(np.float16)
            off += bs
            if (off // args.batch_size) % 20 == 0:
                print(f"    [{off}/{n_pool}] images", flush=True)
    return pool_paths, acts


# ===========================================================================
# Orchestration
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="In-silico psychophysics + diversity")
    p.add_argument("--data", required=True, help="primary explorer_data*.pt")
    p.add_argument("--sae-path", required=True, help="SAE .pth weights")
    p.add_argument("--backbone", default="dinov3", choices=["dinov3", "clip", "dinov2"])
    p.add_argument("--layer", type=int, default=None,
                   help="Intermediate backbone layer to read (hidden_states[layer]). "
                        "OMIT for the final layer (last_hidden_state) -- this is what "
                        "the primary dinov3_l24_spatial model was built on. Passing an "
                        "intermediate layer here feeds the SAE pre-final-norm features "
                        "(out-of-distribution) and collapses its top-k encoder.")
    p.add_argument("--token-type", default="spatial", choices=["spatial", "cls", "all"])
    p.add_argument("--d-model", type=int, default=32000)
    p.add_argument("--top-k", type=int, default=160)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--extra-image-dir", action="append", default=[])
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True,
                   help="Recurse into image-dir subdirectories (--no-recursive to disable)")
    p.add_argument("--pool-size", type=int, default=5000,
                   help="cap on pool images (evenly subsampled)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    # psychophysics
    p.add_argument("--n-features-sample", type=int, default=3000,
                   help="features sampled for the validation (0 = all live)")
    p.add_argument("--n-refs", type=int, default=9)
    p.add_argument("--n-trials", type=int, default=10, help="trials/feature/difficulty")
    p.add_argument("--difficulties", type=float, nargs="+", default=list(DIFFICULTIES_DEFAULT))
    # diversity
    p.add_argument("--ii-top-n", type=int, default=800,
                   help="diversity over the top-N features by II (0 disables)")
    p.add_argument("--ii-min", type=float, default=None,
                   help="alternative: diversity over features with II >= this")
    p.add_argument("--diversity-partners", type=int, default=200,
                   help="random partners per feature within the II subset")
    # lpips / output
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--resize", type=int, default=64)
    p.add_argument("--lpips-batch", type=int, default=512)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load primary data (need II + image pool sizing) ---
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    d_model = data["d_model"]
    ii = data.get("interp_index")
    if ii is None:
        raise SystemExit(
            "Primary file has no 'interp_index'. Run add_interpretability_index.py "
            "on it first (the diversity subset and the validation correlation need II).")
    ii = ii.numpy() if hasattr(ii, "numpy") else np.asarray(ii)
    freq = data["feature_frequency"].numpy()

    # --- Backbone+SAE pass over the pool ---
    print("Computing per-feature activations over the image pool...")
    pool_paths, acts = compute_pool_activations(args, device)

    # Sanity: a faithful pass fires a large fraction of the dictionary across a
    # natural-image pool. If --layer / normalization don't match how the data and
    # SAE were built, the SAE gets out-of-distribution inputs and its top-k encoder
    # collapses to a few dominant features -- which silently produces a degenerate
    # validation (only the most frequent features ever form a trial). Fail loudly.
    n_active = int((acts["max"] > 0).any(axis=0).sum())
    print(f"  features active anywhere in the {len(pool_paths)}-image pool: "
          f"{n_active}/{d_model}")
    if n_active < 0.2 * d_model:
        raise SystemExit(
            f"Activation collapse: only {n_active}/{d_model} features fire on the "
            f"pool. The SAE is receiving out-of-distribution inputs. Check that "
            f"--layer matches how the data/SAE were built: the primary "
            f"dinov3_l24_spatial model uses the FINAL layer (last_hidden_state), "
            f"i.e. NO --layer flag -- not --layer 24 (that reads the pre-final-norm "
            f"block output).")

    # --- LPIPS engine over the pool ---
    sim = LpipsSim(pool_paths, args.lpips_net, args.resize, device, args.lpips_batch)

    # --- Choose feature samples ---
    live = np.where(freq > 0)[0]
    if args.n_features_sample and args.n_features_sample < live.size:
        feat_sample = np.sort(rng.choice(live, size=args.n_features_sample, replace=False))
    else:
        feat_sample = live
    print(f"Psychophysics over {feat_sample.size} features, "
          f"{len(args.difficulties)} difficulties, {args.n_trials} trials each.")

    # ---- Build all psychophysics trials, gather LPIPS pairs, then score ----
    # trials[(agg, feat, diff)] = (ref_idx, pos_list, neg_list)
    trials, pair_set = {}, set()
    for agg in AGGREGATIONS:
        A = acts[agg]
        for f in feat_sample:
            col = A[:, f].astype(np.float32)
            for diff in args.difficulties:
                ref, pos, neg = select_psych_trials(col, args.n_refs, diff,
                                                    args.n_trials, rng)
                trials[(agg, int(f), diff)] = (ref, pos, neg)
                for q in pos + neg:
                    for r in ref:
                        pair_set.add((int(q), int(r)))
    print(f"  unique (query,ref) pairs to LPIPS: {len(pair_set)}")
    sim.precompute_pairs(pair_set)

    # accuracy[agg] : (n_feat, n_diff)
    accuracy = {a: np.full((feat_sample.size, len(args.difficulties)), np.nan) for a in AGGREGATIONS}
    for fi, f in enumerate(feat_sample):
        for di, diff in enumerate(args.difficulties):
            for agg in AGGREGATIONS:
                ref, pos, neg = trials[(agg, int(f), diff)]
                if not pos or not neg:
                    continue
                scores = [afc_correct(sim.max_sim(pq, ref), sim.max_sim(nq, ref))
                          for pq, nq in zip(pos, neg)]
                scores = [s for s in scores if s == s]
                if scores:
                    accuracy[agg][fi, di] = float(np.mean(scores))

    # ---- Diversity: 2AFC separability among high-II features ----
    if args.ii_min is not None:
        subset = np.where(ii >= args.ii_min)[0]
    elif args.ii_top_n:
        finite = np.where(np.isfinite(ii))[0]
        subset = finite[np.argsort(-ii[finite])[:args.ii_top_n]]
    else:
        subset = np.array([], dtype=int)
    subset = np.intersect1d(subset, live)
    print(f"Diversity over {subset.size} high-II features "
          f"({args.diversity_partners} partners each).")

    diversity = {a: np.full(d_model, np.nan, dtype=np.float32) for a in AGGREGATIONS}
    if subset.size >= 2:
        half = args.n_refs // 2
        # Precompute, per agg, each subset feature's ordered MEI pool indices.
        meis = {a: {int(f): np.argsort(-acts[a][:, f].astype(np.float32))
                    for f in subset} for a in AGGREGATIONS}
        # Gather pairs across all partner relationships, then LPIPS once.
        partner_lists, dpairs = {}, set()
        for f in subset:
            others = subset[subset != f]
            k = min(args.diversity_partners, others.size)
            partners = rng.choice(others, size=k, replace=False)
            partner_lists[int(f)] = partners
            for agg in AGGREGATIONS:
                rf = meis[agg][int(f)]
                for b in partners:
                    rb = meis[agg][int(b)]
                    qA, qB = int(rf[args.n_refs]), int(rb[args.n_refs])  # held-out
                    for r in list(rf[:half]) + list(rb[:half]):
                        dpairs.add((qA, int(r))); dpairs.add((qB, int(r)))
        sim.precompute_pairs(dpairs)

        for agg in AGGREGATIONS:
            for f in subset:
                rf = meis[agg][int(f)]
                refA = rf[:half]
                per = []
                for b in partner_lists[int(f)]:
                    rb = meis[agg][int(b)]
                    refB = rb[:half]
                    qA, qB = int(rf[args.n_refs]), int(rb[args.n_refs])
                    s = separability(sim.max_sim(qA, refA), sim.max_sim(qA, refB),
                                     sim.max_sim(qB, refA), sim.max_sim(qB, refB))
                    per.append(s)
                if per:
                    diversity[agg][int(f)] = float(np.mean(per))

    # ---- Save + figures ----
    _save_and_plot(out_dir, args, feat_sample, ii, accuracy, diversity, subset)


def _save_and_plot(out_dir, args, feat_sample, ii, accuracy, diversity, subset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    diffs = np.array(args.difficulties)

    # 1) Accuracy vs difficulty, one curve per aggregation (paper Fig. 5).
    plt.figure(figsize=(6, 4.5))
    for agg in AGGREGATIONS:
        m = np.nanmean(accuracy[agg], axis=0)
        se = np.nanstd(accuracy[agg], axis=0) / np.sqrt(np.sum(~np.isnan(accuracy[agg]), axis=0))
        plt.errorbar(diffs, m, yerr=se, marker="o", capsize=3, label=agg)
    plt.axhline(0.5, ls="--", c="grey", label="chance")
    plt.gca().invert_xaxis()
    plt.xlabel("Difficulty (positive-query activation quantile)")
    plt.ylabel("Psychophysics accuracy")
    plt.title("In-silico psychophysics by MEI aggregation")
    plt.legend(); plt.tight_layout()
    f1 = os.path.join(out_dir, "psychophysics_difficulty_by_aggregation.png")
    plt.savefig(f1, dpi=150); plt.close()
    print(f"  wrote {f1}")

    # 2) Validation: II vs psychophysics accuracy, per agg. We correlate II
    #    with the MEAN accuracy across difficulties (area under the difficulty
    #    curve): single-difficulty accuracy saturates at ceiling (paper Sec 2.3),
    #    so the mean carries more signal. (ρ at the easiest point is also logged.)
    ii_s = ii[feat_sample]
    corr = {}
    for agg in AGGREGATIONS:
        acc_mean = np.nanmean(accuracy[agg], axis=1)   # over difficulties
        acc0 = accuracy[agg][:, 0]                      # easiest difficulty
        ok = np.isfinite(ii_s) & np.isfinite(acc_mean)
        if ok.sum() > 10:
            rho, pval = spearmanr(ii_s[ok], acc_mean[ok])
            ok0 = np.isfinite(ii_s) & np.isfinite(acc0)
            rho0, _ = spearmanr(ii_s[ok0], acc0[ok0]) if ok0.sum() > 10 else (np.nan, np.nan)
            corr[agg] = {"rho_mean": float(rho), "p_mean": float(pval),
                         "rho_easiest": float(rho0), "n": int(ok.sum())}
            plt.figure(figsize=(5, 4))
            plt.scatter(ii_s[ok], acc_mean[ok], s=6, alpha=0.3)
            plt.xlabel("Interpretability Index (II)")
            plt.ylabel(f"Mean psychophysics accuracy ({agg})")
            plt.title(f"II vs accuracy  (Spearman ρ={rho:.3f}, p={pval:.1e}, n={ok.sum()})")
            plt.tight_layout()
            fp = os.path.join(out_dir, f"ii_vs_psychophysics_{agg}.png")
            plt.savefig(fp, dpi=150); plt.close()
            print(f"  wrote {fp}  (ρ_mean={rho:.3f}, ρ_easiest={rho0:.3f})")

    # 3) Diversity histograms per aggregation.
    for agg in AGGREGATIONS:
        vals = diversity[agg][np.isfinite(diversity[agg])]
        if vals.size:
            plt.figure(figsize=(5, 3.5))
            plt.hist(vals, bins=40)
            plt.xlabel("2AFC separability (diversity)")
            plt.ylabel("features")
            plt.title(f"Diversity among high-II features ({agg}), n={vals.size}")
            plt.tight_layout()
            fh = os.path.join(out_dir, f"diversity_hist_{agg}.png")
            plt.savefig(fh, dpi=150); plt.close()
            print(f"  wrote {fh}")

    res = os.path.join(out_dir, "psychophysics_results.pt")
    torch.save({
        "feature_sample": feat_sample,
        "difficulties": diffs,
        "accuracy": {a: accuracy[a] for a in AGGREGATIONS},
        "diversity": {a: diversity[a] for a in AGGREGATIONS},
        "ii_subset": subset,
        "spearman_ii_vs_acc": corr,
        "args": vars(args),
    }, res)
    print(f"  wrote {res}")
    print("\nValidation (II vs mean psychophysics accuracy across difficulties):")
    for agg, c in corr.items():
        print(f"  {agg:>4}: Spearman ρ={c['rho_mean']:.3f}  p={c['p_mean']:.2e}  "
              f"(n={c['n']}; ρ@easiest={c['rho_easiest']:.3f})")


# ===========================================================================
# Self-test of the pure logic (no torch/LPIPS/GPU): python psychophysics_eval.py --self-test
# ===========================================================================

def _self_test():
    rng = np.random.default_rng(0)
    n = 2000
    # A feature that activates strongly on a clear cluster of images.
    acts = np.zeros(n); acts[:50] = np.linspace(5, 1, 50); acts[50:200] = rng.uniform(0, 1, 150)
    ref, pos, neg = select_psych_trials(acts, n_refs=9, difficulty=1.0, n_trials=10, rng=rng)
    assert len(ref) == 9 and len(pos) == 10 and len(neg) == 10
    assert not np.intersect1d(ref, pos).size, "positives must exclude refs"
    assert acts[pos].mean() > acts[neg].mean(), "easy positives stronger than negatives"
    # Harder difficulty -> weaker positives.
    _, pos_hard, _ = select_psych_trials(acts, 9, 0.5, 10, rng)
    assert acts[pos_hard].mean() < acts[pos].mean(), "harder => weaker positive queries"

    # 2AFC decision + separability.
    assert afc_correct(0.9, 0.1) == 1.0 and afc_correct(0.1, 0.9) == 0.0
    assert afc_correct(0.5, 0.5) == 0.5
    # Distinct features separate cleanly; identical ones don't.
    # args: (qA_to_A, qA_to_B, qB_to_A, qB_to_B) — clean split: A prefers A, B prefers B.
    assert separability(0.9, 0.1, 0.1, 0.9) == 1.0
    assert separability(0.5, 0.5, 0.5, 0.5) == 0.5
    assert separability(0.1, 0.9, 0.9, 0.1) == 0.0   # both queries cross-assigned

    # End-to-end scoring with a synthetic embedding similarity: images have 2D
    # embeddings; sim = -distance. A feature's MEIs cluster near a center, so a
    # positive (near center) beats a negative (random) -> accuracy ~1 at easy.
    emb = rng.normal(size=(n, 2)) * 3
    emb[:50] = rng.normal(loc=[10, 10], scale=0.3, size=(50, 2))   # the cluster
    emb[50:200] = rng.normal(loc=[10, 10], scale=1.5, size=(150, 2))
    def msim(q, refs):
        return max(-np.linalg.norm(emb[q] - emb[r]) for r in refs if r != q)
    correct = []
    for _ in range(200):
        ref, pos, neg = select_psych_trials(acts, 9, 1.0, 1, rng)
        if pos and neg:
            correct.append(afc_correct(msim(pos[0], ref), msim(neg[0], ref)))
    acc = np.mean(correct)
    assert acc > 0.8, f"easy-difficulty accuracy should be high, got {acc:.2f}"
    print(f"self-test OK (easy-difficulty synthetic accuracy = {acc:.2f})")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
