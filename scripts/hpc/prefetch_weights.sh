#!/bin/bash
# ---------------------------------------------------------------------------
# Cache timm's pretrained encoder weights ON THE LOGIN NODE.
#
# configs/model/default.yaml sets `pretrained: true`, so the encoder fetches its
# weights from the HuggingFace hub when the model is built. Compute nodes have
# no outbound internet, so an uncached model makes a training job die at
# construction. Run this once per encoder you intend to train with:
#
#     bash $VSC_DATA/projects/cinder/scripts/hpc/prefetch_weights.sh
#     bash $VSC_DATA/projects/cinder/scripts/hpc/prefetch_weights.sh tiny_vit_21m_384.dist_in22k_ft_in1k
#
# env.sh points HF_HOME and TORCH_HOME at $VSC_SCRATCH/cache, so the download
# lands where the job will look for it -- and off the home quota.
#
# NOTE: scratch is auto-purged; re-run this after a purge, same as the datasets.
# ---------------------------------------------------------------------------
set -euo pipefail

# Default is the encoder in configs/model/default.yaml. Keep in sync, or pass
# the model name explicitly.
MODELS=("${@:-convnext_tiny.dinov3_lvd1689m}")

# This runs on the login node (zen2), so it must not use litleo's zen4 build --
# importing torch from it dies with SIGILL. accelgor's stack is zen3 but still
# executes here, the same reason prefetch_wheels.sh uses it. The weights are
# plain files, so the cache serves every cluster regardless of which stack
# downloaded them.
export CINDER_CLUSTER="${CINDER_CLUSTER:-accelgor}"

source "${CINDER_HPC:-$VSC_DATA/projects/cinder/scripts/hpc}/env.sh"

echo "cache  : $HF_HOME"
for model in "${MODELS[@]}"; do
    echo "--- $model ---"
    python -c "
import sys, timm
name = sys.argv[1]
m = timm.create_model(name, pretrained=True)
print('  cached: %s (%.1fM params)' % (name, sum(p.numel() for p in m.parameters()) / 1e6))
" "$model"
done

echo
du -sh "$HF_HOME" "$TORCH_HOME" 2>/dev/null || true
