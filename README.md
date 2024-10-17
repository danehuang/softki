# Soft Kernel Interpolation



## Environnment

```
conda create --name softgp python=3.12

pip install torch torchvision torchaudio
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

```
python train.py
```

## Reproducing Experiments

### UCI

1. Make sure you have downloaded UCI data. Run GPs `./run_uci.sh`

2. Run analysis `./analysis/comparison.ipynb`

### MD22

1. Make sure you have downloaded MD22 data.  Run GPs `./run_md22.sh`

2. Run analysis `./analysis/md22.ipynb`

### Noise

1. Run noise experiments `./run_noise.sh`

2. Run analysis `./analysis/noise.ipynb`

### Noise

1. Run inducing experiments `./run_inducing.sh`

2. Run analysis `./analysis/inducing.ipynb`

