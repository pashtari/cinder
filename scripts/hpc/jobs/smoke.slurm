#!/bin/bash
#SBATCH --job-name=cinder-smoke
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/data/gent/520/vsc52033/projects/cinder/logs/%x-%j.out
#SBATCH --error=/data/gent/520/vsc52033/projects/cinder/logs/%x-%j.err
#
#   sbatch --clusters=litleo jobs/smoke.slurm
#
# For the 2-GPU / NCCL path, use a cluster that allows it (litleo's QoS caps
# every job at 1 GPU):
#   sbatch --clusters=accelgor --gpus-per-node=2 --cpus-per-task=24 jobs/smoke.slurm
#
set -euo pipefail
source "$VSC_DATA/projects/cinder/scripts/hpc/env.sh"

nvidia-smi
echo
python "$CINDER_HPC/smoke_test.py"
