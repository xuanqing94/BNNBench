#!/bin/bash

# for split in VSparse Sparse Dense VDense; do
#   mkdir -p ./results/ensemble_Enza-OOD-f2_VDense/$split
#   CUDA_VISIBLE_DEVICES=0,1 python -m BNNBench.inference.ensemble_inference \
#     --ckpt-file ./ckpt/Enza-OOD-f2/VDense/model_* \
#     --img-size 1024 \
#     --batch-size 12 \
#     --test-src-dir ./datasets/Enza-OOD-f2/A/$split/ \
#     --test-tgt-dir ./datasets/Enza-OOD-f2/B/$split/ \
#     --result-dir ./results/ensemble_Enza-OOD-f2_VDense/$split
# done


for split in VDense Dense Sparse VSparse; do
mkdir -p ./results/ensemble_local_Enza-OOD-f2_VDense/$split
CUDA_VISIBLE_DEVICES=0,1 python -m BNNBench.inference.ensemble_inference \
  --ckpt-file ./ckpt/Enza-OOD-f2/VDense/local_0* \
  --img-size 1024 \
  --batch-size 48 \
  --test-src-dir ./datasets/Enza-OOD-f2/A/$split/ \
  --test-tgt-dir ./datasets/Enza-OOD-f2/B/$split/ \
  --result-dir ./results/ensemble_local_Enza-OOD-f2_VDense/$split
done