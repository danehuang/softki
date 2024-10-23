# System/Library imports
from typing import *
import time

# Common data science imports
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
except:
    pass

# GPytorch and linear_operator
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, MaternKernel

# Our imports
from gp.svi_gp.svi_gp import GPModel
from gp.util import dynamic_instantiation, flatten_dict, flatten_dataset, split_dataset, filter_param, heatmap


# =============================================================================
# Configuration
# =============================================================================

CONFIG = OmegaConf.create({
    'model': {
        'name': 'svi-gp',
        'kernel': {
            '_target_': 'RBFKernel'
        },
        'use_scale': True,
        'num_inducing': 1024,
        'induce_init': 'kmeans',
        'noise': 0.5,
        'noise_constraint': 1e-1,
        'learn_noise': True,
        'dtype': 'float32',
        'device': 'cpu',
    },
    'dataset': {
        'name': 'elevators',
        'train_frac': 0.9,
        'val_frac': 0.0,
        'num_workers': 0,
    },
    'training': {
        'seed': 42,
        'batch_size': 1024,
        'learning_rate': 0.01,
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
# Train
# =============================================================================

def train_gp(config: DictConfig, train_dataset: Dataset, test_dataset: Dataset):
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_scale, num_inducing, dtype, device, noise, noise_constraint, learn_noise = (
        dynamic_instantiation(config.model.kernel),
        config.model.use_scale,
        config.model.num_inducing,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.noise_constraint,
        config.model.learn_noise,
    )

    # Unpack training configuration
    seed, batch_size, epochs, lr = (
        config.training.seed,
        config.training.batch_size,
        config.training.epochs,
        config.training.learning_rate,
    )

    # Set wandb
    if config.wandb.watch:
        # Create wandb config with training/model config
        config_dict = flatten_dict(OmegaConf.to_container(config, resolve=True))

        # Create name
        rname = f"svigp_{dataset_name}_{num_inducing}_{batch_size}_{noise}_{seed}"
        
        # Initialize wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=rname,
            config=config_dict
        )

    # Set dtype
    print("Setting dtype to ...", dtype)
    torch.set_default_dtype(dtype)

    # Initialize inducing points with kmeans
    train_x, train_y = flatten_dataset(train_dataset)
    kmeans = KMeans(n_clusters=num_inducing)
    kmeans.fit(train_x)
    centers = kmeans.cluster_centers_
    inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)

    # Model
    # inducing_points = torch.rand(num_inducing, train_dataset.dim).to(device=device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(noise_constraint)).to(device=device)
    likelihood.noise = torch.tensor([noise]).to(device=device)
    model = GPModel(kernel, inducing_points=inducing_points, use_scale=use_scale).to(device=device)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_dataset))

    # Set optimizers
    model.train()
    likelihood.train()
    variational_optimizer = torch.optim.Adam([{'params': model.variational_parameters()}], lr=lr)
    lr_sched = lambda epoch: 1.0
    variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)
    
    if learn_noise:
        hypers = model.hyperparameters()
        params = likelihood.parameters()
    else:
        hypers = model.hyperparameters()
        params = filter_param(likelihood.named_parameters(), "noise_covar.raw_noise")
    hyperparameter_optimizer = torch.optim.Adam([
        {"params": hypers},
        {"params": params},
    ], lr=lr)
    hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)

    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.dataset.num_workers)
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    for epoch in pbar:
        t1 = time.perf_counter()
        minibatch_iter = train_loader

        losses = []; nlls = []
        # Perform an epoch of fitting hyperparameters (including inducing points)
        for x_batch, y_batch in minibatch_iter:
            # Load batch
            x_batch = x_batch.to(device=device)
            y_batch = y_batch.to(device=device)

            # Perform optimization
            variational_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            output = likelihood(model(x_batch))
            loss = -mll(output, y_batch)
            nlls += [-loss.item()]
            losses += [loss.item()]
            loss.backward()

            # step optimizers and learning rate schedulers
            variational_optimizer.step()
            variational_scheduler.step()
            hyperparameter_optimizer.step()
            hyperparameter_scheduler.step()

            # Log
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix(MLL=f"{-loss.item()}")
        t2 = time.perf_counter()
        
        # Evaluate
        test_rmse, test_nll = eval_gp(model, likelihood, test_dataset, device=device) 
        model.train()
        likelihood.train()

        # Log
        if config.wandb.watch:
            z = model.variational_strategy.inducing_points
            K_zz = model.covar_module(z).evaluate()
            K_zz = K_zz.detach().cpu().numpy()
            custom_bins = [0, 1e-20, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 20]
            hist = np.histogram(K_zz.flatten(), bins=custom_bins)
            results = {
                "loss": torch.tensor(losses).mean(),
                "test_nll": test_nll,
                "test_rmse": test_rmse,
                "epoch_time": t2 - t1,
                "noise": likelihood.noise_covar.noise.cpu(),
                "lengthscale": model.get_lengthscale(),
                "outputscale": model.get_outputscale(),
                # "K_zz_bins": wandb.Histogram(np_histogram=np_hist),
                "K_zz_norm_2": np.linalg.norm(K_zz, ord='fro'),
                "K_zz_norm_1": np.linalg.norm(K_zz, ord=1),
                "K_zz_norm_inf": np.linalg.norm(K_zz, ord=np.inf),
            }
            for cnt, edge in zip(hist[0], hist[1]):
                results[f"K_zz_bin_{edge}"] = cnt

            if epoch % 10 == 0 or epoch == epochs - 1:
                img = heatmap(K_zz)
                results.update({
                    "inducing_points": wandb.Histogram(z.detach().cpu().numpy()),
                    "K_zz": wandb.Image(img)
                })

                artifact = wandb.Artifact(f"inducing_points_{rname}_{epoch}", type="parameters")
                np.save("array.npy", z.detach().cpu().numpy()) 
                artifact.add_file("array.npy")
                wandb.log_artifact(artifact)
            
            wandb.log(results)

    return model, likelihood


def eval_gp(model, likelihood, test_dataset, device="cuda:0", num_workers=4):
    # Set into eval mode
    model.eval()
    likelihood.eval()
    
    # Testing loop
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=num_workers)
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
    nll = torch.sum(torch.tensor(nlls))

    print("RMSE", rmse, rmse.dtype, "NLL", nll, "NOISE", likelihood.noise_covar.noise.cpu().item(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale())
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
    
    # Evaluate
    eval_gp(model, likelihood, test_dataset, device=CONFIG.model.device)
