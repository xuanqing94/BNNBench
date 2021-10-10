#!/bin/bash

set -e


CUDA_VISIBLE_DEVICES=4,5 python -m BNNBench.trainer.sgld_trainer \
  --src-dir ./datasets/Enza-OOD-f2/A/VDense/ \
  --tgt-dir ./datasets/Enza-OOD-f2/B/VDense/ \
  --ckpt-file ./ckpt/Enza-OOD-f2/VDense/model_sgld \
  --lr 1.0e-2 \
  --batch-size 8 \
  --img-size 1024 \
  --total-epochs 150 