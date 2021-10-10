#!/bin/env bash


for split in VDense Dense Sparse VSparse; do
mkdir -p ./results/batchensemble_Enza-OOD-f2_VSparse/$split
CUDA_VISIBLE_DEVICES=0 python -m BNNBench.inference.batchensemble_inference \
  --ckpt-file ./ckpt/Enza-OOD-f2/VSparse/model_batchensemble \
  --n-models 6 \
  --img-size 1024 \
  --batch-size 2 \
  --test-src-dir ./datasets/Enza-OOD-f2/A/$split/ \
  --test-tgt-dir ./datasets/Enza-OOD-f2/B/$split/ \
  --result-dir ./results/batchensemble_Enza-OOD-f2_VSparse/$split
done