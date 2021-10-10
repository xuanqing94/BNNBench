#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0,1 python -m BNNBench.trainer.svi_trainer \
      --src-dir ./datasets/cd105/A/train/ \
      --tgt-dir ./datasets/cd105/B/train/ \
      --ckpt-file ./ckpt/cd105/model_svi \
      --batch-size 8 \
      --img-size 1024 \
      --beta 1.0e-6 \
      --total-epochs 150