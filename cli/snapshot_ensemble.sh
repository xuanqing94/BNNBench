#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=4,5 python -m BNNBench.trainer.snapshot_ensemble_trainer \
    --src-dir ./datasets/Enza-OOD-f2/A/VDense/ \
    --tgt-dir ./datasets/Enza-OOD-f2/B/VDense/ \
    --ckpt-file ./ckpt/Enza-OOD-f2/VDense/snapshot \
    --img-size 1024 \
    --batch-size 8 \
    --total-epochs 150 \
    --n-models 6
