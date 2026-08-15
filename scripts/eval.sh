#!/bin/bash
# Evaluate a trained CINDER checkpoint.
#
# Usage:
#   eval.sh [--gpus=SPEC] [--key=value ...]
#
# --gpus=SPEC:
#   (omit)        CPU only
#   --gpus=1      single GPU (device 0)
#   --gpus=[1]    specific GPU
#   --gpus=2      2 GPUs (torchrun)
#   --gpus=[0,1]  specific GPUs (torchrun)
#   --gpus=-1     all available GPUs (torchrun)
#
# All other --key=value args are forwarded to Hydra (with -- stripped).
# A checkpoint must be supplied via --handler.checkpoint.load_from=...
#
# Example:
#   eval.sh --gpus=1 --dataset=fives --model=cinder_futon --metric='[dice,hd95]' --path.dataset_dir=/scratch/data/FIVES --handler.checkpoint.load_from=path/to/checkpoint.pt
#
# GlaS — report Test A and Test B separately (run once per section):
#   eval.sh --gpus=1 --dataset=glas --model=cinder_futon --metric='[dice,hd95]' --path.dataset_dir=/scratch/data/GlaS --dataset.test_section=testA --handler.checkpoint.load_from=path/to/checkpoint.pt
#   eval.sh --gpus=1 --dataset=glas --model=cinder_futon --metric='[dice,hd95]' --path.dataset_dir=/scratch/data/GlaS --dataset.test_section=testB --handler.checkpoint.load_from=path/to/checkpoint.pt

MODULE="cinder.engine.eval"
USAGE="$(sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//')"

source "$(dirname "${BASH_SOURCE[0]}")/_launch.sh"
