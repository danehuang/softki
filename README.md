# Soft Kernel Interpolation

We provide an implementation of a scalable and high-dimensional Gaussian Process (GP) with *Soft Kernel Interpolation* (SoftKI).



## Quick Start

1. Create environment

```
conda create --name softgp python=3.12
pip install -r requirements.txt
pip install -e .
```

2. Get data and run benchmarks

```
./run_all.sh
```


## Environment

### Quick Install

```
conda create --name softki python=3.12
pip install -e .
```

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


## Get data

### UCI Data

```
cd data
python get_uci.py
```

### MD22 Data

```
cd data
python get_md22.py
```

## Run

Run `soft-gp` on `pol` dataset

```
python train.py \
    --model.name soft-gp \
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





## Methods

1. `gp/exact_gp` contains an Exact GP trained with conjugate gradient descent.
2. `gp/soft_gp` contains SoftKI (our method).
3. `gp/sv_gp` contains a [Sparse GP (often abbreviated SGPR)](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf).
4. `gp/svi_gp` containst a [Stochastic Variational Inference GP (often abbreviated SVGP)](https://arxiv.org/pdf/1309.6835).


## Licence

This repository is released under the [Apache 2.0 license](LICENSE) file.
