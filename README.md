# Soft Kernel Interpolation

We provide an implementation of a scalable and high-dimensional Gaussian Process (GP) with *Soft Kernel Interpolation* (SoftKI).


## Quick Start

1. Create environment

```
conda create --name softki python=3.12
pip install -e .
```

2. Get data

```
./download_data.sh
```

3. Run `softki` on `pol` dataset (from UCI).

```
python train.py \
    --model.name softki \
    --model.num_inducing 512 \
    --model.device cuda:0 \
    --model.use_qr \
    --model.use_scale \
    --model.mll_approx hutchinson \
    --data_dir data/uci_datasets/uci_datasets \
    --dataset.name pol \
    --training.seed 6535 \
    --training.epochs 50 \
    --training.learning_rate 0.01 \
```

### Variables / Arguments Explanation

| Name | Description |
| :------------ |  :----------- |
| `model.name` | Specifies which model `softki`, `svgp`, `sgpr`, `exact`. |
| `model.num_inducing` | Number of inducing points to use. |
| `model.device` |  Which GPU device to use (e.g., `cuda:0`). |
| `model.use_qr` |  Flag to use qr solver for SoftKI. |
| `model.use_scale` | Flag to use scale kernel. |
| `model.mll_approx` | Set to `hutchinson` to. |
| `data_dir` |  Path to data (e.g., `data/uci_datasets/uci_datasets`). |
| `dataset.name ` |  Name of dataset (see scripts for names) |
| `training.seed` |  Set random seed to use. |
| `training.epochs` | Number of epochs to train for. |
| `training.learning_rate` |  Hyper-parameter optimization learning rate. |



### Manual Install

```
conda create --name softki python=3.12

pip install torch
pip install tqdm requests wandb
pip install scipy scikit-learn pandas matplotlib omegaconf
pip install gpytorch 
pip install seaborn
pip install -e .
```


## Methods

This repository contains implementations of a few GP methods.

1. `gp/exact` contains an Exact GP trained with conjugate gradient descent.
2. `gp/softki` contains SoftKI (our method).
3. `gp/sgpr` contains a [Sparse GP (often abbreviated SGPR)](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf).
4. `gp/svgp` containst a [Stochastic Variational Inference GP (often abbreviated SVGP)](https://arxiv.org/pdf/1309.6835).


## Benchmark

Run
```
./benchmark.sh
```
to test our method.


## Experiments

See [scripts/README.md](scripts/README.md) for how to replicate experiments.


## Licence

This repository is released under the [Apache 2.0 license](LICENSE) file.
