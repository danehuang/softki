# System/Library imports
import argparse
import time
from typing import *

# Common data science imports
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
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
from gp.soft_gp import SoftGP


# =============================================================================
# Train and Evaluate
# =============================================================================

# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

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
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


def dynamic_instantiation(config: DictConfig) -> Any:
    # Instantiate the class using OmegaConf
    target_class = globals()[config['_target_']]  # Get the class from the globals() dictionary
    return target_class(**{k: v for k, v in config.items() if k != '_target_'})


def filter_param(named_params: list[Tuple[str, torch.nn.Parameter]], name: str) -> list[Tuple[str, torch.nn.Parameter]]:
    return [param for n, param in named_params if n != name]


def flatten_omegaconf(cfg, parent_key='', separator='.'):
    items = {}
    for key, value in cfg.items():
        new_key = f'{parent_key}{separator}{key}' if parent_key else key
        if isinstance(value, DictConfig):
            items.update(flatten_omegaconf(OmegaConf.create(value), new_key, separator=separator))
        elif isinstance(value, ListConfig):
            for i, item in enumerate(value):
                items.update(flatten_omegaconf(OmegaConf.create({i: item}), new_key, separator=separator))
        else:
            items[new_key] = value
    return items


def unflatten_dict(flat_dict):
    hierarchical_dict = {}
    for key, value in flat_dict.items():
        keys = key.split('.')
        d = hierarchical_dict
        for sub_key in keys[:-1]:
            if sub_key not in d:
                d[sub_key] = {}
            d = d[sub_key]
        d[keys[-1]] = value
    return hierarchical_dict


# ---------------------------------------------------------
# Train/Eval
# ---------------------------------------------------------

def train_gp(config: DictConfig, train_dataset: Dataset, test_dataset: Dataset) -> SoftGP:
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, num_inducing, dtype, device, noise, learn_noise, solver, cg_tolerance, mll_approx, fit_chunk_size = (
        dynamic_instantiation(config.model.kernel),
        config.model.num_inducing,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.learn_noise,
        config.model.solver,
        config.model.cg_tolerance,
        config.model.mll_approx,
        config.model.fit_chunk_size,
    )

    # Unpack training configuration
    seed, batch_size, epochs, lr = (
        config.training.seed,
        config.training.batch_size,
        config.training.epochs,
        config.training.learning_rate,
    )

    # Set name
    if config.wandb.watch:
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb_config = {
            "model": "softgp",    
        }.update(config_dict)
        rname = f"softgp{dataset_name}_{config.model.solver}_{dtype}_{num_inducing}_{batch_size}_{noise}"
        wandb.init(project=config.wandb.project, entity=config.wandb.entity, group=config.wandb.group, name=rname, config=wandb_config)

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize inducing points with kmeans
    train_features, train_labels = flatten_dataset(train_dataset)
    kmeans = KMeans(n_clusters=num_inducing)
    kmeans.fit(train_features)
    centers = kmeans.cluster_centers_
    inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    
    # Setup model
    model = SoftGP(
        kernel,
        inducing_points,
        dtype=dtype,
        device=device,
        noise=noise,
        learn_noise=learn_noise,
        solver=solver,
        cg_tolerance=cg_tolerance,
        mll_approx=mll_approx,
        fit_chunk_size=fit_chunk_size
    )

    # Setup optimizer for hyperparameters
    if learn_noise:
        params = model.parameters()
    else:
        params = filter_param(model.named_parameters(), "likelihood.noise_covar.raw_noise")
    optimizer = torch.optim.Adam(params, lr=lr)

    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        model.fit(train_features, train_labels)
        t3 = time.perf_counter()

        # Evaluated gp
        results = eval_gp(model, test_dataset, device=device)

        # Record
        if config.wandb.watch:
            wandb.log({
                "loss": torch.tensor(neg_mlls).mean(),
                "neg_mll": neg_mll,
                "test_rmse": results["rmse"],
                "test_nll": results["nll"],
                "epoch_time": t2 - t1,
                "fit_time": t3 - t2,
                "noise": model.noise.cpu(),
                "lengthscale": model.kernel.lengthscale.cpu(),
            })

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
    
    # Create config
    config = OmegaConf.create({
        'model': {
            'name': 'soft-gp',
            'kernel': {
                '_target_': 'RBFKernel'
            },
            'num_inducing': 1024,
            'noise': 1e-3,
            'learn_noise': False,
            'solver': 'solve',
            'cg_tolerance': 1e-5,
            'mll_approx': 'hutchinson',
            'fit_chunk_size': 1024,
            'dtype': 'float32',
            'device': 'cpu',
        },
        'dataset': {
            'name': 'elevators',
            'train_frac': 4/9,
            'val_frac': 3/9,
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
            'project': 'soft-gp-2',
        }
    })

    # Omega config to argparse
    parser = argparse.ArgumentParser(description="Example of converting OmegaConf to argparse")
    parser.add_argument("--data_dir", type=str, default="../data/uci_datasets/uci_datasets")
    for key, value in flatten_omegaconf(config).items():
        arg_type = type(value)  # Infer the type from the configuration
        parser.add_argument(f'--{key}', type=arg_type, default=value, help=f'Default: {value}')
    args = parser.parse_args()
    cli_config = OmegaConf.create(unflatten_dict(vars(args)))
    config = OmegaConf.merge(config, cli_config)

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
