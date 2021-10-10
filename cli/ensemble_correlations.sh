#!/bin/bash


mkdir -p ./results/snapshot_cd105_Enza_patch/
CUDA_VISIBLE_DEVICES=0,1 python -m BNNBench.inference.ensemble_inference \
  --ckpt-file ./ckpt/cd105/snapshot_{1..5} \
  --img-size 1024 \
  --batch-size 48 \
  --test-src-dir ./datasets/cd105_Enza_patch/A/test \
  --test-tgt-dir ./datasets/cd105_Enza_patch/B/test \
  --result-dir ./results/snapshot_cd105_Enza_patch/
