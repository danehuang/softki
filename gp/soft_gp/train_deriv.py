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
from torch.utils.data import random_split, DataLoader, Dataset, Subset

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
from gp.soft_gp.soft_gp_deriv import SoftGPDeriv
from gp.util import dynamic_instantiation, flatten_dict, unflatten_dict, flatten_dataset, split_dataset, filter_param, heatmap


def my_collate_fn(batch):
    # Unpack the batch into separate lists of data
    # Assume each element in the batch is a tuple (features, label)
    data = [item[0] for item in batch]
    energies = [item[1]["energy"] for item in batch]
    forces = [item[1]["neg_force"] for item in batch]
    
    # Convert to PyTorch tensors
    data_tensor = torch.stack(data, dim=0)  # Stack along batch dimension
    # labels_tensor = torch.stack([torch.tensor(energies).unsqueeze(-1), torch.tensor(forces)], dim=-1)
    # energies = torch.cat(energies, dim=0)
    energies = torch.tensor(energies)
    forces = torch.cat(forces, dim=0)

    # print("E", energies.shape)
    # print("F", forces.shape)

    return data_tensor, torch.cat([energies, forces.reshape(-1)])

    # labels_tensor = torch.cat([energies, forces], dim=-1)
    # print(labels_tensor.shape)
    # print(torch.cat([labels_tensor[:, 0], labels_tensor[:, 1:].reshape(-1)]).shape)
    # return data_tensor, torch.cat([labels_tensor[:, 0], labels_tensor[:, 1:].reshape(-1)])
    # return data_tensor, labels_tensor

# =============================================================================
# Train and Evaluate
# =============================================================================

def train_gp(config: DictConfig, train_dataset: Dataset, test_dataset: Dataset) -> SoftGPDeriv:
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, use_scale, num_inducing, induce_init, dtype, device, noise, learn_noise, solver, cg_tolerance, mll_approx, fit_chunk_size, use_qr = (
        dynamic_instantiation(config.model.kernel),
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
    # D = train_dataset.dim
    # train_dataset = Subset(train_dataset, torch.arange(512))
    # train_dataset.dim = D
    train_features, train_labels = flatten_dataset(train_dataset, collate_fn=my_collate_fn)
    if induce_init == "kmeans":
        print("Using kmeans ...")
        kmeans = KMeans(n_clusters=num_inducing)
        kmeans.fit(train_features)
        centers = kmeans.cluster_centers_
        inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    else:
        print("Using random ...")
        inducing_points = torch.rand(num_inducing, train_dataset.dim).to(device=device)
    # inducing_points = torch.rand(num_inducing, train_dataset.dim).to(device=device)
    
    # Setup model
    model = SoftGPDeriv(
        kernel,
        inducing_points,
        dtype=dtype,
        device=device,
        noise=noise,
        learn_noise=learn_noise,
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.dataset.num_workers, collate_fn=my_collate_fn)
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    for epoch in pbar:
        t1 = time.perf_counter()
        
        # Perform an epoch of fitting hyperparameters (including inducing points)
        neg_mlls = []
        for x_batch, y_batch in train_loader:
            # Load batch
            # print(x_batch.shape, y_batch.shape)
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
        results = eval_gp(model, test_dataset, device=device, num_workers=config.dataset.num_workers, collate_fn=my_collate_fn)

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
                # "K_zz_bins": wandb.Histogram(np_histogram=hist),
                "K_zz_norm_2": np.linalg.norm(K_zz, ord='fro'),
                "K_zz_norm_1": np.linalg.norm(K_zz, ord=1),
                "K_zz_norm_inf": np.linalg.norm(K_zz, ord=np.inf),
                "W_xz_norm_2": np.linalg.norm(model.W_xz_cpu, ord='fro'),
                "W_xz_norm_1": np.linalg.norm(model.W_xz_cpu, ord=1),
                "W_xz_norm_inf": np.linalg.norm(model.W_xz_cpu, ord=np.inf),
            }
            for cnt, edge in zip(hist[0], hist[1]):
                results[f"K_zz_bin_{edge}"] = cnt

            def save_parameters():
                artifact = wandb.Artifact(f"inducing_points_{rname}_{epoch}", type="parameters")
                np.save("array.npy", model.inducing_points.detach().cpu().numpy()) 
                artifact.add_file("array.npy")
                wandb.log_artifact(artifact)

                artifact = wandb.Artifact(f"W_xz_{rname}_{epoch}", type="parameters")
                np.save("W_xz.npy", model.W_xz_cpu.numpy()) 
                artifact.add_file("W_xz.npy")
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


def eval_gp(model: SoftGPDeriv, test_dataset: Dataset, device="cuda:0", num_workers=8, collate_fn=None) -> float:
    preds = []
    neg_mlls = []
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    for x_batch, y_batch in tqdm(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        B = len(x_batch)
        # print("pred", model.pred(x_batch), "y", y_batch)
        preds += [(model.pred(x_batch) - y_batch.reshape(-1))[0:B].detach().cpu()**2]
        neg_mlls += [-model.mll(x_batch, y_batch.reshape(-1)).detach().cpu()]
    rmse = torch.sqrt(torch.sum(torch.cat(preds)) / len(test_dataset)).item()
    neg_mll = torch.sum(torch.tensor(neg_mlls))
            
    print("RMSE:", rmse, "NEG_MLL", neg_mll.item(), "NOISE", model.noise.cpu().item(), "LENGTHSCALE", model.get_lengthscale(), "OUTPUTSCALE", model.get_outputscale())
    
    return {
        "rmse": rmse,
        "nll": neg_mll,
    }


CONFIG = OmegaConf.create({
    'model': {
        'name': 'soft-gp',
        'kernel': {
            '_target_': 'RBFKernel'
        },
        'use_scale': False,
        'num_inducing': 256,
        'induce_init': 'kmeans',
        'noise': 1e-3,
        'learn_noise': False,
        'solver': 'solve',
        'cg_tolerance': 1e-5,
        'mll_approx': 'hutchinson',
        'fit_chunk_size': 32,
        'use_qr': False,
        'dtype': 'float32',
        'device': 'cpu',
    },
    'dataset': {
        'name': 'Sine',
        'num_workers': 1,
        'train_frac': 4/9,
        'val_frac': 3/9,
    },
    'training': {
        'seed': 42,
        'batch_size': 32,
        'learning_rate': 0.01,
        'epochs': 50,
    },
    'wandb': {
        'watch': False,
        'group': 'test',
        'entity': 'bogp',
        'project': 'soft-gp-2',
    }
})


class SineDataset(Dataset):
    def __init__(self):
        self.xs = torch.linspace(-1, 1, 100).unsqueeze(-1)
        self.ys = torch.sin(self.xs).unsqueeze(-1)
        self.dys = torch.cos(self.ys).unsqueeze(-1)
        self.dim = 1

    def __getitem__(self, idx) -> Any:
        return self.xs[idx], {
            "energy": self.ys[idx],
            "neg_force": self.dys[idx]
        }

    def __len__(self):
        return len(self.xs)


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

    from data.get_md22 import (
        MD22_AcAla3NHME_Dataset,
        MD22_DHA_Dataset,
        MD22_DNA_AT_AT_CG_CG_Dataset,
        MD22_DNA_AT_AT_Dataset,
        MD22_Stachyose_Dataset,
    )

    # Omega config to argparse
    parser = argparse.ArgumentParser(description="Example of converting OmegaConf to argparse")
    parser.add_argument("--data_dir", type=str, default="../../data")
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
    elif config.dataset.name == "Ac-Ala3-NHMe":
        dataset = MD22_AcAla3NHME_Dataset(f"{args.data_dir}/md22_Ac-Ala3-NHMe.npz", get_forces=True)
    elif config.dataset.name == "AT-AT":
        dataset = MD22_DNA_AT_AT_Dataset(f"{args.data_dir}/md22_AT-AT.npz", get_forces=True)
    elif config.dataset.name == "AT-AT-CG-CG":
        dataset = MD22_DNA_AT_AT_CG_CG_Dataset(f"{args.data_dir}/md22_AT-AT-CG-CG.npz", get_forces=True)
    elif config.dataset.name == "stachyose":
        dataset = MD22_Stachyose_Dataset(f"{args.data_dir}/md22_stachyose.npz", get_forces=True)
    elif config.dataset.name == "DHA":
        dataset = MD22_DHA_Dataset(f"{args.data_dir}/md22_DHA.npz", get_forces=True)
    elif config.dataset.name == "Sine":
        dataset = SineDataset()
    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported ...")
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_frac=config.dataset.train_frac,
        val_frac=config.dataset.val_frac
    )

    # Train
    model = train_gp(config, train_dataset, test_dataset)
    