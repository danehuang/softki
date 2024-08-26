# Soft GP



## Environnment

```
conda create --name softgp python=3.12

pip install torch torchvision torchaudio
pip install tqdm requests wandb
pip install scipy scikit-learn pandas matplotlib omegaconf
pip install gpytorch 
pip install -e .
```


## Get data

```
cd data
python get_uci.py
```

## Run

```
cd gp
python train.py
```