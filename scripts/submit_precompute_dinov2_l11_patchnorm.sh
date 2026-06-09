#!/bin/bash -l
# Submit precomputation for DINOv2 layer-11 patch-norm SAE.
#
# Backbone : DINOv2 ViT-B/14, intermediate layer 11 (pre-LN, block hook)
# Norm     : per-patch-position normalization (positional_mean / std from
#            train_patch_norm_preln dataset_stats.pt)
# SAE      : d=10000, k=100  (plain SAE, no smart-init)
# Output   : explorer_data_dinov2_l11_patchnorm_d10000_k100_val.pt

SAE_PATH="/users/9/lee02328/Ada_Comp/arch_SAE/trained_models/sae_1_SAE_d10000_k100_state_dict.pth"
OUTPUT_PATH="/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/explorer_data/explorer_data_dinov2_l11_patchnorm_d10000_k100_val.pt"
PATCH_NORM_STATS="/scratch.global/lee02328/data/train_patch_norm_preln/layer_11/dataset_stats.pt"

sbatch \
    --job-name=precompute_dinov2_l11_patchnorm \
    --output=/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE/logs/precompute_dinov2_l11_patchnorm_%j.out \
    --export=ALL,\
SAE_PATH="$SAE_PATH",\
OUTPUT_PATH="$OUTPUT_PATH",\
D_MODEL=10000,\
TOP_K=100,\
BACKBONE=dinov2,\
LAYER=11,\
PATCH_NORM_STATS="$PATCH_NORM_STATS" \
    scripts/run_precompute.sh
