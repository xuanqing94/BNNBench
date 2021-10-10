#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=1 python -m BNNBench.trainer.batchensemble_trainer \
    --src-dir ./datasets/Enza-OOD-f2/A/VDense/ \
    --tgt-dir ./datasets/Enza-OOD-f2/B/VDense/ \
    --n-models 6 \
    --ckpt-file ./ckpt/Enza-OOD-f2/VDense/model_batchensemble \
    --batch-size 6 \
    --img-size 1024 \
    --total-epochs 150

