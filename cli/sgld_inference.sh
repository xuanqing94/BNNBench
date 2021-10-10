#!/bin/bash

set -e

#for split in VSparse Sparse Dense VDense; do
#   mkdir -p ./results/sgld_cd105_Enza_patch
#   CUDA_VISIBLE_DEVICES=0,1,4,5 python -m BNNBench.inference.sgld_inference \
#     --ckpt-file ./ckpt/Enza-OOD-f2/VSparse/model_sgld \
#     --lr 1.0e-6 \
#     --n-samples 5 \
#     --img-size 1024 \
#     --batch-size 8 \
#     --mixing-epochs 1 \
#     --train-src-dir ./datasets/Enza-OOD-f2/A/VSparse/ \
#     --train-tgt-dir ./datasets/Enza-OOD-f2/B/VSparse/ \
#     --test-src-dir ./datasets/Enza-OOD-f2/A/$split/ \
#     --test-tgt-dir ./datasets/Enza-OOD-f2/B/$split/ \
#     --result-dir ./results/sgld_cd105_Enza_patch
# #done

for split in VDense Dense Sparse VSparse; do
  mkdir -p ./results/sgld_Enza-OOD-f2_VDense/$split
  CUDA_VISIBLE_DEVICES=4,5 python -m BNNBench.inference.sgld_inference \
    --ckpt-file ./ckpt/Enza-OOD-f2/VDense/model_sgld \
    --lr 1.0e-6 \
    --n-samples 5 \
    --img-size 1024 \
    --batch-size 8 \
    --mixing-epochs 1 \
    --train-src-dir ./datasets/Enza-OOD-f2/A/VDense/ \
    --train-tgt-dir ./datasets/Enza-OOD-f2/B/VDense/ \
    --test-src-dir ./datasets/Enza-OOD-f2/A/$split/ \
    --test-tgt-dir ./datasets/Enza-OOD-f2/B/$split/ \
    --result-dir ./results/sgld_Enza-OOD-f2_VDense/$split
done