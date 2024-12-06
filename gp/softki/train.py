# System/Library imports
import argparse
import time
from typing import *

# Common data science imports
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader, Dataset

# For logging
try:
    import wandb
except:
    pass

# Gpytorch imports
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from linear_operator.settings import max_cholesky_size

# Our imports
from gp.softki.model import SoftGP
from gp.util import dynamic_instantiation, flatten_dict, unflatten_dict, flatten_dataset, split_dataset, filter_param, heatmap


# =============================================================================
# Configuration
# =============================================================================

CONFIG = OmegaConf.create({
    'model': {
        'name': 'soft-gp',
        'kernel': {
            '_target_': 'RBFKernel',
            'ard_num_dims': None,
        },
        'use_ard': False,
        'use_scale': True,
        'num_inducing': 512,
        'induce_init': 'kmeans',
        'noise': 1e-3,
        'learn_noise': False,
        'use_T': False,
        'T': 5e-5,
        'learn_T': False,
        'threshold': 1e-2,
        'use_threshold': False,
        'learn_threshold': False,
        'solver': 'solve',
        'cg_tolerance': 1e-5,
        'mll_approx': 'hutchinson',
        'fit_chunk_size': 1024,
        'use_qr': True,
        'hutch_solver': 'solve',
        'dtype': 'float32',
        'device': 'cpu',
    },
    'dataset': {
        'name': 'elevators',
        'num_workers': 0,
        'train_frac': 0.9,
        'val_frac': 0.0,
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
# Train and Evaluate
# =============================================================================

def train_gp(config: DictConfig, train_dataset: Dataset, test_dataset: Dataset) -> SoftGP:
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_ard, use_scale, num_inducing, induce_init, dtype, device, noise, learn_noise, solver, cg_tolerance, mll_approx, fit_chunk_size, use_qr = (
        dynamic_instantiation(config.model.kernel),
        config.model.use_ard,
        config.model.use_scale,
        config.model.num_inducing,
        config.model.induce_init,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.learn_noise,
        config.model.solver,
        config.model.cg_tolerance,
        config.model.mll_approx,
        config.model.fit_chunk_size,
        config.model.use_qr,
    )
    if use_ard:
        config.model.kernel.ard_num_dims = train_dataset.dim
        kernel = dynamic_instantiation(config.model.kernel)

    use_T, T, learn_T, use_threshold, threshold, learn_threshold = (
        config.model.use_T,
        config.model.T,
        config.model.learn_T,
        config.model.use_threshold,
        config.model.threshold,
        config.model.learn_threshold,
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
        rname = f"softgp_{dataset_name}_{num_inducing}_{batch_size}_{noise}_{seed}"
        
        # Initialize wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=rname,
            config=config_dict
        )

    # Initialize inducing points with kmeans
    train_features, train_labels = flatten_dataset(train_dataset)
    if induce_init == "kmeans":
        print("Using kmeans ...")
        kmeans = KMeans(n_clusters=num_inducing)
        kmeans.fit(train_features)
        centers = kmeans.cluster_centers_
        inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    else:
        print("Using random ...")
        inducing_points = torch.rand(num_inducing, train_dataset.dim).to(device=device)
    
    # Setup model
    model = SoftGP(
        kernel,
        inducing_points,
        dtype=dtype,
        device=device,
        noise=noise,
        learn_noise=learn_noise,
        use_T=use_T,
        T=T,
        learn_T=learn_T,
        use_threshold=use_threshold,
        threshold=threshold,
        learn_threshold=learn_threshold,
        use_scale=use_scale,
        solver=solver,
        cg_tolerance=cg_tolerance,
        mll_approx=mll_approx,
        fit_chunk_size=fit_chunk_size,
        use_qr=use_qr,
    )

    # Setup optimizer for hyperparameters
    if learn_noise:
        params = model.parameters()
    else:
        params = filter_param(model.named_parameters(), "likelihood.noise_covar.raw_noise")
    optimizer = torch.optim.Adam(params, lr=lr)

    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.dataset.num_workers)
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    for epoch in pbar:
        t1 = time.perf_counter()
        
        # Perform an epoch of fitting hyperparameters (including inducing points)
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

        # Solve for weights given fixed inducing points
        use_pinv = model.fit(train_features, train_labels)
        t3 = time.perf_counter()

        # Evaluate gp
        results = eval_gp(model, test_dataset, device=device, num_workers=config.dataset.num_workers)

        # Record
        if config.wandb.watch:
            K_zz = model._mk_cov(model.inducing_points).detach().cpu().numpy()
            custom_bins = [0, 1e-20, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 20]
            hist = np.histogram(K_zz.flatten(), bins=custom_bins)
            results = {
                "loss": torch.tensor(neg_mlls).mean(),
                "use_pinv": 1 if use_pinv else 0,
                "test_rmse": results["rmse"],
                "test_nll": results["nll"],
                "epoch_time": t2 - t1,
                "fit_time": t3 - t2,
                "noise": model.noise.cpu(),
                "lengthscale": model.get_lengthscale(),
                "outputscale": model.get_outputscale(),
                "threshold": model.threshold.cpu().item(),
                "T": model.T.cpu().item(),
                # "K_zz_bins": wandb.Histogram(np_histogram=hist),
                "K_zz_norm_2": np.linalg.norm(K_zz, ord='fro'),
                "K_zz_norm_1": np.linalg.norm(K_zz, ord=1),
                "K_zz_norm_inf": np.linalg.norm(K_zz, ord=np.inf),
            }
            for cnt, edge in zip(hist[0], hist[1]):
                results[f"K_zz_bin_{edge}"] = cnt

            def save_parameters():
                artifact = wandb.Artifact(f"inducing_points_{rname}_{epoch}", type="parameters")
                np.save("array.npy", model.inducing_points.detach().cpu().numpy()) 
                artifact.add_file("array.npy")
                wandb.log_artifact(artifact)

                artifact = wandb.Artifact(f"K_zz_{rname}_{epoch}", type="parameters")
                np.save("K_zz.npy", K_zz) 
                artifact.add_file("K_zz.npy")
                wandb.log_artifact(artifact)

            if epoch % 10 == 0 or epoch == epochs - 1:
                img = heatmap(K_zz)
                results.update({
                    "inducing_points": wandb.Histogram(model.inducing_points.detach().cpu().numpy()),
                    "K_zz": wandb.Image(img)
                })
                save_parameters()
            
            wandb.log(results)

    return model


def eval_gp(model: SoftGP, test_dataset: Dataset, device="cuda:0", num_workers=0) -> float:
    preds = []
    neg_mlls = []
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=num_workers)
    
    for x_batch, y_batch in tqdm(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred_mean = model.pred(x_batch)
        preds += [(pred_mean - y_batch).detach().cpu()**2]
        covar = model.pred_cov(x_batch)
        nll = -torch.distributions.Normal(pred_mean, torch.sqrt(covar.diag())).log_prob(y_batch).detach().cpu()
        # nll = torch.tensor([0])
        neg_mlls += [nll]
    rmse = torch.sqrt(torch.sum(torch.cat(preds)) / len(test_dataset)).item()
    neg_mll = torch.cat(neg_mlls).mean()
            
    print("RMSE:", rmse, "NEG_MLL", neg_mll.item(), "NOISE", model.noise.cpu().item(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale(), "THRESHOLD", model.threshold.cpu().item(), "T", model.T.cpu().item())
    
    return {
        "rmse": rmse,
        "nll": neg_mll,
    }


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    from data.get_uci import (
        PoleteleDataset,
        ElevatorsDataset,
        BikeDataset,
        Kin40KDataset,
        ProteinDataset,
        KeggDirectedDataset,
        CTSlicesDataset,
        KeggUndirectedDataset,
        RoadDataset,
        SongDataset,
        BuzzDataset,
        HouseElectricDataset,
    )

    # Omega config to argparse
    parser = argparse.ArgumentParser(description="Example of converting OmegaConf to argparse")
    parser.add_argument("--data_dir", type=str, default="../../data/uci_datasets/uci_datasets")
    for key, value in flatten_dict(OmegaConf.to_container(CONFIG, resolve=True)).items():
        arg_type = type(value)  # Infer the type from the configuration
        parser.add_argument(f'--{key}', type=arg_type, default=value, help=f'Default: {value}')
    args = parser.parse_args()
    cli_config = OmegaConf.create(unflatten_dict(vars(args)))
    config = OmegaConf.merge(CONFIG, cli_config)

    # Create dataset
    if config.dataset.name == "pol":
        dataset = PoleteleDataset(f"{args.data_dir}/pol/data.csv")
    elif config.dataset.name == "elevators":
        dataset = ElevatorsDataset(f"{args.data_dir}/elevators/data.csv")
    elif config.dataset.name == "bike":
        dataset = BikeDataset(f"{args.data_dir}/bike/data.csv")
    elif config.dataset.name == "kin40k":
        dataset = Kin40KDataset(f"{args.data_dir}/kin40k/data.csv")
    elif config.dataset.name == "protein":
        dataset = ProteinDataset(f"{args.data_dir}/protein/data.csv")
    elif config.dataset.name == "keggdirected":
        dataset = KeggDirectedDataset(f"{args.data_dir}/keggdirected/data.csv")
    elif config.dataset.name == "slice":
        dataset = CTSlicesDataset(f"{args.data_dir}/slice/data.csv")
    elif config.dataset.name == "keggundirected":
        dataset = KeggUndirectedDataset(f"{args.data_dir}/keggundirected/data.csv")
    elif config.dataset.name == "3droad":
        dataset = RoadDataset(f"{args.data_dir}/3droad/data.csv")
    elif config.dataset.name == "song":
        dataset = SongDataset(f"{args.data_dir}/song/data.csv")
    elif config.dataset.name == "buzz":
        dataset = BuzzDataset(f"{args.data_dir}/buzz/data.csv")
    elif config.dataset.name == "houseelectric":
        dataset = HouseElectricDataset(f"{args.data_dir}/houseelectric/data.csv")
    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported ...")
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_frac=config.dataset.train_frac,
        val_frac=config.dataset.val_frac
    )

    # Train
    model = train_gp(config, train_dataset, test_dataset)
    
    # Eval
    eval_gp(model, test_dataset, device=config.model.device)
