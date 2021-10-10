#!/bin/bash

set -e

#for split in VSparse Sparse Dense VDense; do
  mkdir -p ./results/svi_cd105_impurities
  CUDA_VISIBLE_DEVICES=0,1,4,5 python -m BNNBench.inference.svi_inference \
    --ckpt-file ./ckpt/cd105/model_svi \
    --n-fwd 5 \
    --img-size 1024 \
    --batch-size 64 \
    --test-src-dir ./datasets/impurities_bbox_test/A/test \
    --test-tgt-dir ./datasets/impurities_bbox_test/B/test \
    --result-dir ./results/svi_cd105_impurities
#done