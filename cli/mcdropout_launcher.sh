#!/bin/bash

set -e


CUDA_VISIBLE_DEVICES=4,5 python -m BNNBench.trainer.mcdropout_trainer \
  --src-dir ./datasets/cd105/A/train/ \
  --tgt-dir ./datasets/cd105/B/train/ \
  --ckpt-file ./ckpt/cd105/model_mcdropout \
  --batch-size 8 \
  --img-size 1024 \
  --total-epochs 150 