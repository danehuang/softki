# System/Library imports
import time
from typing import *

# Common data science imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader, Dataset

import wandb

# Gpytorch imports
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from linear_operator.settings import max_cholesky_size

# Our imports
from gp.soft_gp import SoftGP


# =============================================================================
# Train and Evaluate
# =============================================================================

def flatten_dataset(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    train_x = []
    train_y = []
    for batch_x, batch_y in train_loader:
        train_x += [batch_x]
        train_y += [batch_y]
    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0).squeeze(-1)
    return train_x, train_y


def split_dataset(dataset: Dataset, train_frac=4/9, val_frac=3/9) -> Tuple[Dataset, Dataset, Dataset]:
    train_size = int(len(dataset) * train_frac)
    val_size = int(len(dataset) * val_frac)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(elevators_dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


def train_gp(
    dataset_name: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    D: int,
    seed=42,
    train_frac=4/9,
    val_frac=3/9,
    kernel="matern",
    interp_type="softmax",
    noise=1e-3,
    learn_noise=False,
    num_inducing=1024,
    solver="solve",
    cg_tolerance=1e-5,
    epochs=50,
    batch_size=1024,
    lr=0.01,
    device="cuda:0",
    dtype=torch.float32,
    group="test",
    project="isgp",
    watch=False,
    trace=False,
):
    if watch:
        config = {
            "model": "softgp",
            "dataset_name": dataset_name,
            "dim": D,
            "dtype": dtype,
            "device": device,
            "num_inducing": num_inducing,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "learn_noise": learn_noise,
            "kernel": kernel,
            "interp_type": interp_type,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "seed": seed,
            "solver": solver,
            "cg_tolerance": cg_tolerance,
        }
        rname = f"softgp{dataset_name}_{interp_type}_{solver}_{dtype}_{num_inducing}_{batch_size}_{noise}"
        wandb.init(project=project, entity="bogp", group=group, name=rname, config=config)

    # Set seed
    np.random.seed(seed)

    # Initialize inducing points with kmeans
    train_features, train_labels = flatten_dataset(train_dataset)
    kmeans = KMeans(n_clusters=num_inducing)
    kmeans.fit(train_features)
    centers = kmeans.cluster_centers_
    inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    
    # Setup kernel
    if kernel == "rbf":
        k = RBFKernel()
    elif kernel == "rbf-ard":
        k = RBFKernel(ard_num_dims=D)
    elif kernel == "matern":
        k = MaternKernel(nu=1.5)
    elif kernel == "matern_ard":
        k = MaternKernel(nu=1.5, ard_num_dims=train_features.shape[-1], lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0))
    elif kernel == "matern0.5":
        k = MaternKernel(nu=0.5)
    else:
        raise ValueError(f"Kernel {kernel} not supported ...")
    
    # Setup model
    model = SoftGP(k, inducing_points, dtype=dtype, device=device, noise=noise, learn_noise=learn_noise, solve_method=solver, cg_tolerance=cg_tolerance)

    # Setup optimizer for hyperparameters
    def filter_param(named_params, name):
        return [param for n, param in named_params if n != name]
    if learn_noise:
        params = model.parameters()
    else:
        params = filter_param(model.named_parameters(), "likelihood.noise_covar.raw_noise")
    optimizer = torch.optim.Adam(params, lr=lr)

    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    cov_mats = []
    for epoch in pbar:
        t1 = time.perf_counter()
        neg_mlls = []
        for x_batch, y_batch in train_loader:
            # Load batch
            x_batch = x_batch.clone().detach().to(dtype=dtype, device=device)
            y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)
            
            # Perform optimization
            optimizer.zero_grad()
            with gpytorch.settings.max_root_decomposition_size(100), max_cholesky_size(int(1.e7)):
                neg_mll = -model.mll(x_batch, y_batch)
            neg_mlls += [-neg_mll.item()]
            neg_mll.backward()
            optimizer.step()

            # Log
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix(MLL=f"{-neg_mll.item()}")
        t2 = time.perf_counter()

        model.fit(train_features, train_labels)
        t3 = time.perf_counter()

        if trace:
            K_zz = model._mk_cov(model.inducing_points)
            W_xz = model._interp(train_features[:64].to(dtype=dtype, device=device))
            cov_mat = W_xz @ K_zz @ W_xz.T 
            if epoch % 10 == 0 or epoch == epochs - 1:
                cov_mats += [cov_mat.detach().cpu().numpy()]
                fig = plt.figure()
                plt.imshow(np.log(cov_mats[-1]), cmap='viridis', vmin=-10, vmax=0)
                plt.colorbar()
                plt.axis('off')
                plt.savefig(f"./figures/trace/{dataset_name}_secgp_{epoch}.png", bbox_inches='tight')
                plt.close(fig)
                
                fig = plt.figure()
                plt.imshow(np.log(model.inducing_points.detach().cpu().numpy().transpose()), cmap='viridis')
                plt.colorbar()
                plt.savefig(f"./figures/trace/{dataset_name}_secgp_induce_{epoch}.png", bbox_inches='tight')
                plt.close(fig)

        results = eval_gp(model, test_dataset, device=device)
        if watch:
            wandb.log({
                "loss": torch.tensor(neg_mlls).mean(),
                "noise": model.noise.cpu(),
                "lengthscale": model.kernel.lengthscale.cpu(),
                "test_rmse": results["rmse"],
                "neg_mll": neg_mll,
                "nll": results["nll"],
                "epoch_time": t2 - t1,
                "fit_time": t3 - t2,
            })

    if trace:
        np.savez(f"./figures/trace/{dataset_name}_softgp_covmats.npz", *cov_mats)

    return model


def eval_gp(model: SoftGP, test_dataset: Dataset, device="cuda:0") -> float:
    preds = []
    neg_mlls = []
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    for x_batch, y_batch in tqdm(test_loader):
        preds += [(model.pred(x_batch.to(device)) - y_batch.to(device)).detach().cpu()**2]
        neg_mlls += [-model.mll(x_batch, y_batch)]
    rmse = torch.sqrt(torch.sum(torch.cat(preds)) / len(test_dataset)).item()
    neg_mll = torch.sum(torch.tensor(neg_mlls))
    
    if isinstance(model.kernel, ScaleKernel):
        base_kernel_lengthscale = model.kernel.base_kernel.lengthscale.cpu()
    else:
        base_kernel_lengthscale = model.kernel.lengthscale.cpu()
        
    print("RMSE:", rmse, "NEG_MLL", neg_mll.item(), "NOISE", model.noise.cpu().item(), "LENGTHSCALE", base_kernel_lengthscale)
    
    return {
        "rmse": rmse,
        "nll": neg_mll,
    }


if __name__ == "__main__":
    from data.get_uci import ElevatorsDataset
    
    # Create dataset
    elevators_dataset = ElevatorsDataset("../data/uci_datasets/uci_datasets/elevators/data.csv")
    train_dataset, val_dataset, test_dataset = split_dataset(elevators_dataset)

    # Train
    device = "cpu"  # device = "cuda:0"
    model = train_gp(
        "elevators",
        train_dataset,
        test_dataset,
        elevators_dataset.dim,
        kernel="matern",
        num_inducing=1024,
        lr=0.01,
        epochs=50,
        device=device
    )
    
    # Eval
    eval_gp(model, test_dataset, device=device)
