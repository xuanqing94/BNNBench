#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -m BNNBench.inference.ensemble_inference \
  --ckpt-file ./ckpt/Enza-OOD-f2/VDense/local_0* \
  --img-size 1024 \
  --batch-size 48 \
  --test-src-dir ./datasets/Enza-OOD-f2/A/$split/ \
  --test-tgt-dir ./datasets/Enza-OOD-f2/B/$split/ \
  --result-dir ./results/ensemble_local_Enza-OOD-f2_VDense/$split
