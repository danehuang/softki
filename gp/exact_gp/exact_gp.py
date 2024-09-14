import time
from typing import *
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import wandb 
from tqdm import tqdm 

from omegaconf import OmegaConf

import torch
from torch.utils.data import TensorDataset, DataLoader

import gpytorch
import gpytorch.constraints
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.means import ZeroMean
# from mll_approximations.Hutchinsons.other_mlls import CGDMLL

# Our imports
from gp.util import dynamic_instantiation, flatten_dict, unflatten_dict, flatten_dataset, split_dataset, filter_param, heatmap


def conjugate_gradient(A, b, max_iter=20, tolerance=1e-5, preconditioner=None):
    if preconditioner is None:
        preconditioner = torch.eye(b.size(0), device=b.device) 
    
    x = torch.zeros_like(b)
    r = b - A.matmul(x)
    z = preconditioner.matmul(r) 
    p = z.clone()
    rz_old = torch.dot(r.view(-1), z.view(-1))

    for i in range(max_iter):
        Ap = A.matmul(p)
        alpha = rz_old / torch.dot(p.view(-1), Ap.view(-1))
        x = x + alpha * p
        r = r - alpha * Ap
        z = preconditioner.matmul(r)  
        rz_new = torch.dot(r.view(-1), z.view(-1))
        if torch.sqrt(rz_new) < tolerance:
            break
        p = z + (rz_new / rz_old) * p
        rz_old = rz_new

    return x


class CGDMLL(gpytorch.mlls.ExactMarginalLogLikelihood):
    def __init__(self, likelihood, model, max_cg_iters=50, cg_tolerance=1e-5):
        super().__init__(likelihood=likelihood, model=model)
        self.max_cg_iters = max_cg_iters
        self.cg_tolerance = cg_tolerance

    def forward(self, function_dist, target):
        function_dist = self.likelihood(function_dist)
        mean = function_dist.mean
        cov_matrix = function_dist.lazy_covariance_matrix.evaluate()

        residual = target - mean

        # Select the solver method
        solve = conjugate_gradient(cov_matrix, residual, max_iter=self.max_cg_iters, tolerance=self.cg_tolerance)
        mll = -0.5 * (residual.squeeze() @ solve).sum() - torch.logdet(cov_matrix)
        return mll


# =============================================================================
# Exact GP
# =============================================================================

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel=MaternKernel(nu=1.5), use_scale=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = ZeroMean()
        self.use_scale = use_scale
        if self.use_scale:
            self.covar_module = ScaleKernel(kernel)
        else:
            self.covar_module = kernel
       
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def get_noise(self) -> float:
        return self.likelihood.noise_covar.noise.cpu()

    def get_lengthscale(self) -> float:
        if self.use_scale:
            return self.covar_module.base_kernel.lengthscale.cpu()
        else:
            return self.covar_module.lengthscale.cpu()

    def get_outputscale(self) -> float:
        if self.use_scale:
            return self.covar_module.outputscale.cpu()
        else:
            return 1.


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
    # inducing_points = train_x[:num_inducing, :].clone() # torch.rand(num_inducing, D).cuda()
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(noise_constraint)).to(device=device)
    likelihood.noise = torch.tensor([noise]).to(device=device)
    model = ExactGPModel(train_x, train_y, likelihood, kernel=kernel, use_scale=use_scale).to(device=device)
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
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
        'train_frac': 4/9,
        'val_frac': 3/9,
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
        'project': 'soft-gp-3',
    }
})


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


# def train_gp(
#         dataset_name,
#         train_dataset,
#         test_dataset,
#         D: int,
#         kernel="rbf",
#         lr=0.01,
#         epochs=50,
#         learn_hyper=True,
#         device="cuda:0",
#         watch=False,
#         group="test",
#         project="isgp",
#         dtype=torch.float32,
#         train_frac=4/9,
#         val_frac=3/9,
#         seed=42,
#         solver="cg",
#         max_cg_iters=50,
#         cg_tolerance=1e-5,
#         learn_noise=True,
#         mll_noise=1e-1,
#         trace=False,
#     ):
#     print("Setting dtype to ...", dtype)
#     torch.set_default_dtype(dtype)

#     if watch:
#         config = {
#             "model": "exact",
#             "dataset_name": dataset_name,
#             "dim": D,
#             "device": device,
#             "lr": lr,
#             "epochs": epochs,
#             "kernel": kernel,
#             "learn_hyper": learn_hyper,
#             "dtype": dtype, 
#             "seed": seed,
#             "train_frac": train_frac,
#             "val_frac": val_frac,
#             "solver": solver,
#             "max_cg_iters": max_cg_iters,
#             "cg_tolerance": cg_tolerance,
#             "learn_noise": learn_noise,
#         }
#         wandb.init(project=project, entity="bogp", group=group, config=config)

#     # Dataset preparation
#     train_x, train_y = zip(*[(torch.tensor(features).to(device), torch.tensor(labels).to(device)) for features, labels in train_dataset])
#     train_x = torch.stack(train_x).squeeze(-1).to(dtype=dtype)
#     train_y = torch.stack(train_y).squeeze(-1).to(dtype=dtype)

#     if kernel == "rbf":
#         k = RBFKernel()
#     elif kernel == "rbf-ard":
#         k = RBFKernel(ard_num_dims=D)
#     elif kernel == "matern":
#         k = MaternKernel(nu=1.5)
#     elif kernel == "matern_ard":
#         k = MaternKernel(nu=1.5, ard_num_dims=D, lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0))
#     elif kernel == "matern0.5":
#         k = MaternKernel(nu=0.5)
#     else:
#         raise ValueError(f"Kernel {kernel} not supported ...")

#     # Model
#     likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
#     likelihood.noise = torch.tensor([mll_noise]).to(device=device)
#     model = ExactGPModel(train_x, train_y, likelihood, kernel=k).to(device)

#     if solver == "cg":
#         solve = "Naive_cg"
#     elif solver == "cholesky":
#         solve = "Naive_choleksy"
#     elif solver == "cg-reorth":
#         solve = "Naive_cg_reorth"
#     else:
#         raise ValueError(f"Solver {solver} not supported ...")
#     mll = CGDMLL(likelihood, model, solve=solve, max_cg_iters=50, cg_tolerance=cg_tolerance)

#     # Training parameters
#     model.train()
#     likelihood.train()
#     def filter_param(named_params, name):
#         params = []
#         for n, param in named_params:
#             if n == name:
#                 continue
#             params += [param]
#         return params
#     if learn_noise:
#         params = model.parameters()
#     else:
#         params = filter_param(model.named_parameters(), "likelihood.noise_covar.raw_noise")
#     optimizer = torch.optim.Adam([{'params': params}], lr=lr)
#     lr_sched = lambda epoch: 1.0
#     if learn_noise:
#         hyperparams = model.hyperparameters()
#     else:
#         hyperparams = filter_param(model.named_hyperparameters(), "likelihood.noise_covar.raw_noise")
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)
#     hyperparameter_optimizer = torch.optim.Adam([{'params': hyperparams}], lr=lr)
#     hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)
#     # print("Parameters", list(model.named_parameters()))
#     # print("HYPERS", list(model.named_hyperparameters()))
#     total_step = 0
    
#     # Training loop
#     pbar = tqdm(range(epochs), desc="Optimizing MLL")
#     cov_mats = []
#     for i in pbar:
#         t1 = time.perf_counter()
#         output = likelihood(model(train_x))
#         loss = -mll(output, train_y)
#         loss.backward()

#         # step optimizers and learning rate schedulers
#         optimizer.step()
#         scheduler.step()
#         if learn_hyper:
#             hyperparameter_optimizer.step()
#             hyperparameter_scheduler.step()
#         t2 = time.perf_counter()


#         if trace:
#             K_xx = model.covar_module(train_x[:64]).evaluate()
#             if i % 10 == 0 or i == epochs - 1:
#                 fig = plt.figure()
#                 cov_mats += [K_xx.detach().cpu().numpy()]
#                 plt.imshow(np.log(1e-10 + K_xx), cmap='viridis', vmin=-10, vmax=0)
#                 plt.colorbar()
#                 plt.axis('off')
#                 plt.savefig(f"./figures/trace/{dataset_name}_exact_{i}.png", bbox_inches='tight')
#                 # plt.imsave(f"./figures/trace/exact_{dataset_name}_{i}.png", np.log(1e-10+K_xx.detach().cpu().numpy()))

#         test_rmse, mean_variance = eval_gp(model, likelihood, test_dataset, device=device)
#         model.train()
#         likelihood.train()

#         if watch:
#             wandb.log({
#                 "loss": loss.item(),
#                 "mean_variance": mean_variance,
#                 "test_rmse": test_rmse,
#                 "epoch_time": t2 - t1,
#                 "noise": model.likelihood.noise_covar.noise.cpu(),
#                 "lengthscale": model.covar_module.lengthscale.cpu(),
#             })
#         if total_step % 50 == 0:
#             print(f"Epoch: {i}; total_step: {total_step}, loss: {loss.item()}, mean_variance: {mean_variance}")

#     if trace:
#         np.savez(f"./figures/trace/{dataset_name}_exact_covmats.npz", *cov_mats)

#     return model, likelihood


# def eval_gp(model, likelihood, test_dataset, device="cuda:0"):
#     model.eval()
#     likelihood.eval()
#     test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

#     total_se = 0.0
#     total_count = 0
#     total_variance = 0.0

#     with torch.no_grad():
#         for x_batch, y_batch in test_loader:
#             x_batch = x_batch.to(device=device)
#             y_batch = y_batch.to(device=device)
            
#             preds = likelihood(model(x_batch))
#             predictions = preds.mean
#             variances = preds.variance

#             # Compute squared errors
#             se = (predictions - y_batch.view_as(predictions))**2
#             total_se += se.sum().item()
#             total_count += x_batch.size(0)
#             total_variance += variances.sum().item()

#     # Calculate RMSE and mean variance
#     rmse = sqrt(total_se / total_count)
#     mean_variance = total_variance / total_count
#     print("RMSE:", rmse, "NOISE", model.likelihood.noise_covar.noise.cpu().item(), "LENGTHSCALE", model.covar_module.lengthscale.cpu())

#     return rmse, mean_variance


# def eval_exact_gp(test_features, test_labels, model, likelihood,device):
#     model.eval()
#     likelihood.eval()
#     test_dataset = TensorDataset(test_features, test_labels)
#     test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

#     total_se = 0.0
#     total_count = 0
#     total_variance = 0.0

#     with torch.no_grad():
#         for x_batch, y_batch in test_loader:
#             if torch.cuda.is_available():
#                 x_batch = x_batch.to(device)
#                 y_batch = y_batch.to(device)
            
#             preds = likelihood(model(x_batch))
#             predictions = preds.mean
#             variances = preds.variance

#             se = (predictions - y_batch.view_as(predictions))**2
#             total_se += se.sum().item()
#             total_count += x_batch.size(0)
#             total_variance += variances.sum().item()
    
#     rmse = sqrt(total_se / total_count)
#     mean_variance = total_variance / total_count
#     return rmse, mean_variance