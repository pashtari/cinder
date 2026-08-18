#!/bin/bash
#SBATCH --job-name=cinder-venv
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/gent/520/vsc52033/projects/cinder/logs/%x-%j.out
#SBATCH --error=/data/gent/520/vsc52033/projects/cinder/logs/%x-%j.err
#
# Build the venv -- once per cluster architecture.
#
#   sbatch --clusters=litleo   jobs/setup_venv.slurm
#   sbatch --clusters=accelgor jobs/setup_venv.slurm
#
# Must run as a job, not on the login node: login nodes are zen2 and litleo's
# binaries are zen4, so importing torch there dies with SIGILL.
#
set -euo pipefail

source "$VSC_DATA/projects/cinder/scripts/hpc/env.sh"

echo "cluster : ${SLURM_CLUSTER_NAME}"
echo "arch    : ${VSC_ARCH_LOCAL}${VSC_ARCH_SUFFIX}"
echo "venv    : ${CINDER_VENV}"

# --system-site-packages so the venv inherits the CUDA-enabled module builds of
# torch/torchvision/timm instead of pulling CPU wheels from PyPI.
[ -f "$CINDER_VENV/bin/activate" ] || python -m venv --system-site-packages "$CINDER_VENV"
# shellcheck disable=SC1091
source "$CINDER_VENV/bin/activate"

# Compute nodes have NO outbound internet -- pip runs fully offline against the
# wheels staged on the login node by prefetch_wheels.sh.
WHEELS="$CINDER_HPC/wheels"
if [ -z "$(ls -A "$WHEELS" 2>/dev/null)" ]; then
    echo "ERROR: no staged wheels at $WHEELS" >&2
    echo "       run on the LOGIN NODE first: bash \$CINDER_HPC/prefetch_wheels.sh" >&2
    exit 1
fi

echo "--- installing venv-only dependencies ---"
python -m pip install --no-index --find-links "$WHEELS" --no-deps \
    -r "$CINDER_HPC/requirements-hpc.txt"

# --no-deps is mandatory here, not an optimisation. The module builds satisfy
# cinder's floors (torch 2.1.2 >= 2.1, torchvision 0.16 >= 0.16), but pip does
# not treat them as a ceiling: given a free hand it resolves to the newest
# pytorch-ignite, pulls torch 2.13 and the entire nvidia-* CUDA stack over the
# module build, and then fails anyway for want of a network. requirements-hpc.txt
# pins the versions that match this stack.
echo "--- installing cinder (editable, no-deps) ---"
python -m pip install --no-build-isolation --no-deps -e "$CINDER_ROOT"

echo "--- verifying ---"
python "$CINDER_HPC/smoke_test.py"
