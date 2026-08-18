#!/bin/bash
#SBATCH --job-name=cinder-train
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --output=/data/gent/520/vsc52033/projects/cinder/logs/%x-%j.out
#SBATCH --error=/data/gent/520/vsc52033/projects/cinder/logs/%x-%j.err
#
# Train cinder on one of the three datasets.
#
#   sbatch --clusters=accelgor --gpus-per-node=1 jobs/train.slurm GlaS
#   sbatch --clusters=accelgor --gpus-per-node=1 jobs/train.slurm FIVES
#   sbatch --clusters=accelgor --gpus-per-node=2 jobs/train.slurm ADE20K
#
# Extra args go straight to Hydra:
#   sbatch --clusters=accelgor jobs/train.slurm GlaS --trainer.max_iters=40000
#
# GPU count is read from the allocation, so one script covers 1-GPU and DDP;
# scripts/train.sh dispatches plain python vs torchrun accordingly.
#
# #SBATCH directives cannot expand variables, hence the absolute log paths.
# Cluster choice: litleo (H100) is capped at 1 GPU/job by its QoS, so DDP must
# go to accelgor (A100, 12 cores/GPU) or joltik (V100, 8 cores/GPU); match
# --cpus-per-task to cores/GPU x GPUs.
#
set -euo pipefail

DATASET="${1:?usage: sbatch train.slurm <GlaS|FIVES|ADE20K> [hydra args...]}"
shift

source "$VSC_DATA/projects/cinder/scripts/hpc/env.sh"

# Lowercased for Hydra, which selects configs/{dataset,trainer}/<name>.yaml;
# the directories on scratch keep the datasets' own capitalisation.
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"

# Dataset root layouts are set by fetch_datasets.sh; metrics follow the
# conventions in scripts/train.sh (region metrics for binary segmentation,
# IoU for ADE20K's 151 classes).
case "$DATASET" in
    glas)   DATA_DIR="$CINDER_DATA/GlaS";                        METRIC="[dice,hd95]" ;;
    fives)  DATA_DIR="$CINDER_DATA/FIVES";                       METRIC="[dice,hd95]" ;;
    ade20k) DATA_DIR="$CINDER_DATA/ADE20K";                      METRIC="iou" ;;
    *)      echo "unknown dataset '$DATASET' (expected GlaS|FIVES|ADE20K)" >&2; exit 2 ;;
esac

[ -d "$DATA_DIR" ] || { echo "missing dataset at $DATA_DIR -- run fetch_datasets.sh $DATASET on the login node" >&2; exit 1; }

NGPU=$(nvidia-smi --list-gpus | wc -l)
TAG="slurm-${SLURM_JOB_ID}"

# Hydra nests its run under $CINDER_LOGS/<dataset>/<tag>/<model>/<timestamp>
# (configs/hydra/default.yaml), so this parent is predictable at submit time
# and is where provenance for the job belongs.
RUN_DIR="$CINDER_LOGS/$DATASET/$TAG"
mkdir -p "$RUN_DIR"

# Collect Slurm's own stdout/stderr into the run directory too.
#
# #SBATCH --output cannot point here directly: it is expanded before the job
# starts, so it knows neither $DATASET nor the job id it would need for the
# directory name, and Slurm will not create missing parents. So let Slurm open
# the files at the top of logs/ and rename them into place now. $CINDER_LOGS and
# $RUN_DIR are on one filesystem, so this is a plain rename: Slurm's already-open
# descriptors follow the files, and output written from here on still arrives.
# A job that dies before this point simply leaves its logs at the top level.
for _stream in out err; do
    _src="$CINDER_LOGS/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.${_stream}"
    if [ -f "$_src" ]; then
        mv "$_src" "$RUN_DIR/"
    fi
done
unset _stream _src

echo "=========================================================="
echo " cinder | $DATASET | ${NGPU} GPU(s)"
echo " cluster : ${SLURM_CLUSTER_NAME}   node: $(hostname)"
echo " job     : ${SLURM_JOB_ID}"
echo " data    : $DATA_DIR"
echo " out     : $RUN_DIR"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo
cinder_provenance "$RUN_DIR"
echo

if [ "$NGPU" -gt 1 ]; then
    # Single-node DDP: keep NCCL on shared memory / P2P, skip InfiniBand probing.
    export NCCL_IB_DISABLE=1
    export NCCL_DEBUG=WARN
fi

"$CINDER_ROOT/scripts/train.sh" \
    --gpus="$NGPU" \
    --dataset="$DATASET" \
    --trainer="$DATASET" \
    --metric="$METRIC" \
    --path.dataset_dir="$DATA_DIR" \
    --dataset.num_workers=$(( SLURM_CPUS_PER_TASK / NGPU )) \
    --tag="$TAG" \
    "$@"

echo "=== finished at $(date -Is) ==="
