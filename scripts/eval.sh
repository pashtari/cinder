#!/bin/bash
# Evaluate a trained CINDER checkpoint.
#
# Usage:
#   scripts/eval.sh [--gpus=SPEC] --handler.checkpoint.load_from=CKPT [--key=value ...]
#
# Other arguments are Hydra overrides, with or without the leading "--". Pass
# the model config the checkpoint was trained with.
#
# Examples:
#   scripts/eval.sh --gpus=1 --dataset=fives --model=futongate_relu \
#       --path.dataset_dir=/path/to/FIVES --handler.checkpoint.load_from=/path/to/checkpoint.pt
#   scripts/eval.sh --gpus=1 --dataset=ade20k --metric=iou --inferer=tta \
#       --path.dataset_dir=/path/to/ADE20K --handler.checkpoint.load_from=/path/to/checkpoint.pt
#
# GlaS reports Test A and Test B separately with object metrics; pixel Dice and
# HD95 are not comparable with published GlaS results:
#   scripts/eval.sh --gpus=1 --dataset=glas --dataset.test_split=testA \
#       --metric='[object_f1,object_dice,object_hausdorff]' \
#       --path.dataset_dir=/path/to/GlaS --handler.checkpoint.load_from=/path/to/checkpoint.pt

MODULE="cinder.engine.eval"
source "$(dirname "${BASH_SOURCE[0]}")/_launch.sh"
