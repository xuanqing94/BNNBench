for i in 0 1 2 3 4; do
CUDA_VISIBLE_DEVICES=$i python -m BNNBench.inference.batchensemble_inference \
    --n-models 6 --ckpt-file ./checkpoint/batchensemble/vgg16_cifar10/ckpt_combo.pth \
    --data cifar10-c --level $i > cifar10_batchensemble_level${i}.txt &
done
