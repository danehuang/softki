# System imports
import time
from typing import *

# Gpytorch / Torch
import gpytorch
import gpytorch.constraints
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
import torch
from torch.utils.data import DataLoader

# Other
from omegaconf import OmegaConf
from tqdm import tqdm 

try:
    import wandb 
except:
    pass

# Our imports
from gp.exact.mll import CGDMLL
from gp.exact.model import ExactGPModel
from gp.util import dynamic_instantiation, flatten_dict, flatten_dataset, split_dataset


# =============================================================================
# Configuration
# =============================================================================

CONFIG = OmegaConf.create({
    'model': {
        'name': 'exact',
        'kernel': {
            '_target_': 'RBFKernel'
        },
        'use_scale': True,
        'noise': 0.5,
        'noise_constraint': 1e-1,
        'learn_noise': True,
        'dtype': 'float32',
        'device': 'cpu',
        'max_cg_iters': 50,
        'cg_tolerance': 1e-3,
    },
    'dataset': {
        'name': 'elevators',
        'train_frac': 0.9,
        'val_frac': 0.0,
    },
    'training': {
        'seed': 42,
        'learning_rate': 0.1,
        'epochs': 50,
    },
    'wandb': {
        'watch': True,
        'group': 'test',
        'entity': 'bogp',
        'project': 'softki',
    }
})


# =============================================================================
# Train / Eval
# =============================================================================

def train_gp(config, train_dataset, test_dataset):
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_scale, dtype, device, noise, noise_constraint, learn_noise, max_cg_iters, cg_tolerance = (
        dynamic_instantiation(config.model.kernel),
        config.model.use_scale,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.noise_constraint,
        config.model.learn_noise,
        config.model.max_cg_iters,
        config.model.cg_tolerance,
    )

    # Unpack training configuration
    seed, epochs, lr = (
        config.training.seed,
        config.training.epochs,
        config.training.learning_rate,
    )

    # Set wandb
    if config.wandb.watch:
        # Create wandb config with training/model config
        config_dict = flatten_dict(OmegaConf.to_container(config, resolve=True))

        # Create name
        rname = f"exact_{dataset_name}_{noise}_{seed}"
        
        # Initialize wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=rname,
            config=config_dict
        )
    
    print("Setting dtype to ...", dtype)
    torch.set_default_dtype(dtype)

    # Dataset preparation
    train_x, train_y = flatten_dataset(train_dataset)
    train_x = train_x.to(dtype=dtype, device=device)
    train_y = train_y.to(dtype=dtype, device=device)

    # Model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(noise_constraint)).to(device=device)
    likelihood.noise = torch.tensor([noise]).to(device=device)
    model = ExactGPModel(train_x, train_y, likelihood, kernel=kernel, use_scale=use_scale).to(device=device)
    mll = CGDMLL(likelihood, model, max_cg_iters=max_cg_iters, cg_tolerance=cg_tolerance)

    # Training parameters
    model.train()
    likelihood.train()

    # Set optimizer
    if learn_noise:
        params = model.parameters()
        hypers = likelihood.parameters()
    else:
        params = model.parameters()
        hypers = []
    optimizer = torch.optim.Adam([
        {'params': params},
        {'params': hypers}
    ], lr=lr)
    lr_sched = lambda epoch: 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)
    
    # Training loop
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    for epoch in pbar:
        t1 = time.perf_counter()

        # Load batch
        optimizer.zero_grad()
        output = likelihood(model(train_x))
        loss = -mll(output, train_y)
        loss.backward()

        # step optimizers and learning rate schedulers
        optimizer.step()
        scheduler.step()
        t2 = time.perf_counter()

        # Log
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        pbar.set_postfix(MLL=f"{-loss.item()}")

        # Evaluate
        test_rmse, test_nll = eval_gp(model, likelihood, test_dataset, device=device)
        model.train()
        likelihood.train()

        if config.wandb.watch:
            results = {
                "loss": loss,
                "test_nll": test_nll,
                "test_rmse": test_rmse,
                "epoch_time": t2 - t1,
                "noise": model.get_noise(),
                "lengthscale": model.get_lengthscale(),
                "outputscale": model.get_outputscale(),
            }
            wandb.log(results)
        
    return model, likelihood


def eval_gp(model, likelihood, test_dataset, device="cuda:0"):
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Testing loop
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    squared_errors = []
    nlls = []
    for test_x, test_y in tqdm(test_loader):
        output = likelihood(model(test_x.to(device=device)))
        means = output.mean.cpu()
        stds = output.variance.sqrt().cpu()
        nll = -torch.distributions.Normal(means, stds).log_prob(test_y).mean()
        se = torch.sum((means - test_y)**2)
        squared_errors += [se]
        nlls += [nll]
    rmse = torch.sqrt(torch.sum(torch.tensor(squared_errors)) / len(test_dataset))
    nll = torch.sum(torch.tensor(nll))

    print("RMSE", rmse, rmse.dtype, "NLL", nll, "NOISE", model.get_noise().item(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale())
    return rmse, nll


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from data.get_uci import ElevatorsDataset

    # Get dataset
    dataset = ElevatorsDataset("../../data/uci_datasets/uci_datasets/elevators/data.csv")
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_frac=CONFIG.dataset.train_frac,
        val_frac=CONFIG.dataset.val_frac    
    )

    # Test
    model, likelihood = train_gp(CONFIG, train_dataset, test_dataset)