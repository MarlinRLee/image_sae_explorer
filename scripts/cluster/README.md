# Cluster (SLURM) wrappers

SLURM submission scripts for the training / precompute / analysis pipeline.
They were written for the authors' cluster (MSI) and contain
site-specific values — partition names, `module`/conda activation lines,
`/scratch.global/...` data paths, and log locations. **They are not part of
the portable demo**; `scripts/run_explorer_local.sh` and
`scripts/precompute_all.sh` are all you need to run the explorer.

To use these on your own cluster, edit the `#SBATCH` headers and the
environment-setup block at the top of each script (conda env, base paths)
before submitting.

| Script | Submits |
|---|---|
| `grab_coco.sh` | Download + unpack COCO val/train images |
| `run_extract_layer.sh` | `extract_activations.py` — backbone activation shards |
| `run_train.sh` | `src/main.py` — SAE training |
| `run_precompute.sh` | `precompute_explorer_data.py` + `precompute_heatmaps.py` |
| `run_heatmaps.sh` | `precompute_heatmaps.py` only |
| `run_clip_embeddings.sh` | `add_clip_embeddings.py` — CLIP enrichment of a `.pt` |
| `run_interp_index.sh` | `add_interpretability_index.py` |
| `run_ii_by_aggregation.sh` | `compute_ii_by_aggregation.py` |
| `run_auto_interp.sh` / `run_auto_interp_budget.sh` | Gemini auto-labeling (`auto_interp_*.py`) |
| `run_psychophysics.sh` / `run_psychophysics_sim_metrics.sh` | `psychophysics_*.py` evals |
| `submit_ising.sh` | `precompute_ising.py` |
| `submit_hf_upload.sh` | Upload sidecars (+ optionally weights/labels/thumbnails) to HF — see its header for the label-safety modes |
