#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=0,1 python -m BNNBench.trainer.ensemble_local_trainer \
  --src-dir ./datasets/Enza-OOD-f2/A/VSparse/ \
  --tgt-dir ./datasets/Enza-OOD-f2/B/VSparse/ \
  --seed-model ./ckpt/Enza-OOD-f2/VSparse/model_0 \
  --coef 5 \
  --ckpt-file ./ckpt/Enza-OOD-f2/VSparse/local_0 \
  --batch-size 8 \
  --const-lr-epochs 10 \
  --total-epochs 20 \
  --img-size 1024


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m BNNBench.trainer.ensemble_local_trainer \
#   --src-dir ./datasets/cd105/A/train/ \
#   --tgt-dir ./datasets/cd105/B/train/ \
#   --seed-model ./ckpt/cd105/model_0 \
#   --coef 5 \
#   --ckpt-file ./ckpt/cd105/local_0 \
#   --batch-size 8 \
#   --const-lr-epochs 10 \
#   --total-epochs 20 \
#   --img-size 1024
