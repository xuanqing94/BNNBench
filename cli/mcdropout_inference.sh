#!/bin/bash

# for split in VSparse Sparse Dense VDense; do
#   mkdir -p ./results/mcdropout_Enza-OOD-f2_VSparse/$split
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m BNNBench.inference.mcdropout_inference \
#     --ckpt-file ./ckpt/Enza-OOD-f2/VSparse/model_mcdropout \
#     --n-fwd 15 \
#     --img-size 1024 \
#     --batch-size 64 \
#     --test-src-dir ./datasets/Enza-OOD-f2/A/$split/ \
#     --test-tgt-dir ./datasets/Enza-OOD-f2/B/$split/ \
#     --result-dir ./results/mcdropout_Enza-OOD-f2_VSparse/$split
# done


mkdir -p ./results/mcdropout_cd105_impurities
CUDA_VISIBLE_DEVICES=0,1,4,5 python -m BNNBench.inference.mcdropout_inference \
  --ckpt-file ./ckpt/cd105/model_mcdropout \
  --n-fwd 15 \
  --img-size 1024 \
  --batch-size 64 \
  --test-src-dir ./datasets/impurities_bbox_test/A/test \
  --test-tgt-dir ./datasets/impurities_bbox_test/B/test \
  --result-dir ./results/mcdropout_cd105_impurities
