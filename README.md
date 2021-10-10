# Bayesian neural network benchmark suite focusing on biomedical datasets

## Install our software
Run following commands in bash shell:

```bash
pip install -r requirements.txt
pip install -e .
```


## Run benchmarks

+ Ensemble method:

```bash
python -m BNNBench.trainer.ensemble_trainer \
  --train-src-dir ./datasets/\
  --train-tgt-dir \
  --test-src-dir \
  --test-tgt-dir \
  --prior-std \
  --train-batch-size \
  --test-batch-size \
  --img-size
```