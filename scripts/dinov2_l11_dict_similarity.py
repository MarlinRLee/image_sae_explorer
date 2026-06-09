"""
Compute pairwise cosine similarity between the two DINOv2 layer-11 SAE dictionaries
(spatial vs patchnorm), cluster by similarity, and display as a heatmap.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

BASE = "/projects/standard/boleydl/shared/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE"
SPATIAL_PATH   = f"{BASE}/models/dinov2_l11_spatial/sae_1_SI-SAE_d10000_k100_per_init0.02_state_dict.pth"
PATCHNORM_PATH = f"{BASE}/models/dinov2_l11_patchnorm/sae_1_SAE_d10000_k100_state_dict.pth"
OUT_PATH       = f"{BASE}/figures/dinov2_l11_dict_similarity_heatmap.png"
HIST_PATH      = f"{BASE}/figures/dinov2_l11_dict_similarity_max_hist.png"

print("Loading dictionaries...")
# _weights stored as (10000, 768) = D.T  where D is the decoder [768, 10000]
# Each row is one dictionary atom.
w_spatial   = torch.load(SPATIAL_PATH,   map_location='cpu')['dictionary._weights']   # (10000, 768)
w_patchnorm = torch.load(PATCHNORM_PATH, map_location='cpu')['dictionary._weights']   # (10000, 768)

# D = w.T  shape [768, 10000].  Normalize columns of D == normalize rows of w.
D_spatial   = torch.nn.functional.normalize(w_spatial,   dim=1).T   # (768, 10000)
D_patchnorm = torch.nn.functional.normalize(w_patchnorm, dim=1).T   # (768, 10000)

print("Computing 10k x 10k cosine similarity matrix (D_s.T @ D_p)...")
sim = (D_spatial.T @ D_patchnorm).numpy()   # (10000, 10000)

# --- Hierarchical clustering on each axis independently ---
print("Clustering rows (spatial features)...")
row_dist = pdist(sim, metric='euclidean')
row_link = linkage(row_dist, method='ward')
row_order = leaves_list(row_link)

print("Clustering cols (patchnorm features)...")
col_dist = pdist(sim.T, metric='euclidean')
col_link = linkage(col_dist, method='ward')
col_order = leaves_list(col_link)

sim_ordered = sim[np.ix_(row_order, col_order)]
print(f"Similarity range: min={sim.min():.4f}, max={sim.max():.4f}, "
      f"mean={sim.mean():.4f}, std={sim.std():.4f}")

# --- Plot ---
print("Plotting heatmap...")
fig, ax = plt.subplots(figsize=(10, 9))
vlim = np.percentile(np.abs(sim_ordered), 99)
im = ax.imshow(sim_ordered, aspect='auto', cmap='RdBu_r',
               vmin=-vlim, vmax=vlim, interpolation='nearest')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Cosine similarity')
ax.set_xlabel("DINOv2 L11 PatchNorm features (clustered)", fontsize=12)
ax.set_ylabel("DINOv2 L11 Spatial features (clustered)", fontsize=12)
ax.set_title("Pairwise cosine similarity: Spatial vs PatchNorm SAE dictionaries\n"
             "(DINOv2 layer 11, d=10 000, hierarchical clustering)", fontsize=12)
ax.set_xticks([])
ax.set_yticks([])

import os
os.makedirs(f"{BASE}/figures", exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved to {OUT_PATH}")

# --- Histogram of best-match cosine similarity per spatial feature ---
print("Plotting max-similarity histogram...")
max_sim = sim.max(axis=1)   # best patchnorm match for each spatial feature
print(f"Max-sim per feature: min={max_sim.min():.4f}, median={np.median(max_sim):.4f}, "
      f"mean={max_sim.mean():.4f}, max={max_sim.max():.4f}")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(max_sim, bins=100, color='steelblue', edgecolor='none')
ax2.set_xlabel("Max cosine similarity to any PatchNorm feature", fontsize=12)
ax2.set_ylabel("Number of Spatial features", fontsize=12)
ax2.set_title("Best-match similarity: Spatial → PatchNorm SAE dictionaries\n"
              "(DINOv2 layer 11, d=10 000)", fontsize=12)
ax2.axvline(np.median(max_sim), color='red', linestyle='--', label=f"Median = {np.median(max_sim):.3f}")
ax2.legend()
plt.tight_layout()
plt.savefig(HIST_PATH, dpi=150, bbox_inches='tight')
print(f"Saved to {HIST_PATH}")
