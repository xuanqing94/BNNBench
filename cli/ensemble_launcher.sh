#!/bin/bash

set -e

for model_id in {4..5}; do
    CUDA_VISIBLE_DEVICES=4,5 python -m BNNBench.trainer.ensemble_trainer \
      --src-dir ./datasets/Enza-OOD-f2/A/VSparse/ \
      --tgt-dir ./datasets/Enza-OOD-f2/B/VSparse/ \
      --ckpt-file ./ckpt/Enza-OOD-f2/VSparse/model_${model_id} \
      --batch-size 12 \
      --img-size 1024 \
      --total-epochs 150 
      #--coef 15
done
