#!/bin/bash
# Train CINDER.
#
# Usage:
#   train.sh [--gpus=SPEC] [--key=value ...]
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
#
# Examples:
#   train.sh --gpus=1 --dataset=fives --trainer=fives --model=cinder_futon --path.dataset_dir=/scratch/data/FIVES --metric='[dice,hd95]'
#   train.sh --gpus=2 --dataset=glas --trainer=glas --model=cinder_futon --path.dataset_dir=/scratch/data/GlaS --metric='[dice,hd95]'
#   train.sh --gpus=-1 --dataset=ade20k --trainer=ade20k --model=cinder_futon --path.dataset_dir=/scratch/data/ADE20K --metric=iou

MODULE="cinder.engine.train"
USAGE="$(sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//')"

source "$(dirname "${BASH_SOURCE[0]}")/_launch.sh"
