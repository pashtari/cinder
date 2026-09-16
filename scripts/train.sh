#!/bin/bash
# Train CINDER.
#
# Usage:
#   scripts/train.sh [--gpus=SPEC] [--key=value ...]
#
# Other arguments are Hydra overrides, with or without the leading "--". The
# trainer config follows the dataset.
#
# Examples:
#   scripts/train.sh --gpus=1 --dataset=glas --path.dataset_dir=/path/to/GlaS
#   scripts/train.sh --gpus=2 --dataset=fives --path.dataset_dir=/path/to/FIVES
#   scripts/train.sh --gpus=-1 --dataset=ade20k --path.dataset_dir=/path/to/ADE20K --metric=iou

MODULE="cinder.engine.train"
source "$(dirname "${BASH_SOURCE[0]}")/_launch.sh"
