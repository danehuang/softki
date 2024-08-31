# System/Library imports
from typing import *
import time

# Common data science imports
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

try:
    import wandb
except:
    pass

# GPytorch
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

# Our imports
from gp.util import dynamic_instantiation, flatten_dict, unflatten_dict, flatten_dataset, split_dataset, filter_param, heatmap


# =============================================================================
# Variational GP
# =============================================================================

class SGPRModel(gpytorch.models.ExactGP):
    """
    Adapated from:
    https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/SGPR_Regression_CUDA.html

    Args:
        gpytorch (_type_): _description_
    """    
    def __init__(self, kernel: Callable, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, inducing_points=None):
        super(SGPRModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # self.base_covar_module = ScaleKernel(kernel)
        # self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)
        self.covar_module = InducingPointKernel(kernel, inducing_points=inducing_points, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    

# =============================================================================
# Train / Eval
# =============================================================================

def train_gp(config, train_dataset, test_dataset):
    # Unpack dataset
    dataset_name = config.dataset.name

    # Unpack model configuration
    kernel, num_inducing, dtype, device, noise, learn_noise = (
        dynamic_instantiation(config.model.kernel),
        config.model.num_inducing,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.learn_noise,
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
        rname = f"svgp_{dataset_name}_{dtype}_{num_inducing}_{noise}"
        
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

    # Initialize inducing points with kmeans
    kmeans = KMeans(n_clusters=num_inducing)
    kmeans.fit(train_x)
    centers = kmeans.cluster_centers_
    inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    # inducing_points = torch.rand(num_inducing, train_dataset.dim).to(device=device)

    train_x = train_x.to(dtype=dtype, device=device)
    train_y = train_y.to(dtype=dtype, device=device)

    # Model
    # inducing_points = train_x[:num_inducing, :].clone() # torch.rand(num_inducing, D).cuda()
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-1)).to(device=device)
    # print("INITIAL NOISE", likelihood.noise_covar.noise.cpu())
    likelihood.noise = torch.tensor([noise]).to(device=device)
    model = SGPRModel(kernel, train_x, train_y, likelihood, inducing_points=inducing_points).to(device=device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training parameters
    model.train()
    likelihood.train()

    # Set optimizer
    if learn_noise:
        params = model.parameters()
    else:
        params = filter_param(model.named_parameters(), "likelihood.noise_covar.raw_noise")
    optimizer = torch.optim.Adam([{'params': params}], lr=lr)
    lr_sched = lambda epoch: 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)
    
    # Training loop
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    for epoch in pbar:
        t1 = time.perf_counter()

        # Load batch
        optimizer.zero_grad()
        output = model(train_x)
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
                "noise": model.likelihood.noise_covar.noise.cpu(),
                "lengthscale": model.covar_module.base_kernel.lengthscale.cpu(),
                # "lengthscale": model.base_covar_module.base_kernel.lengthscale.cpu(),
            }

            if epoch % 10 == 0:
                z = model.covar_module.inducing_points
                K_zz = model.covar_module(z).evaluate()
                K_zz = K_zz.detach().cpu().numpy()
                img = heatmap(K_zz)

                results.update({
                    "inducing_points": wandb.Histogram(z.detach().cpu().numpy()),
                    "K_zz": wandb.Image(img)
                })

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

    print("RMSE", rmse, rmse.dtype, "NLL", nll, "NOISE", model.likelihood.noise_covar.noise.cpu().item(), "LENGTHSCALE", model.covar_module.base_kernel.lengthscale.cpu())
    # print("RMSE", rmse, rmse.dtype, "NLL", nll, "NOISE", model.likelihood.noise_covar.noise.cpu().item(), "LENGTHSCALE", model.base_covar_module.base_kernel.lengthscale.cpu())
    return rmse, nll


CONFIG = OmegaConf.create({
    'model': {
        'name': 'sv-gp',
        'kernel': {
            '_target_': 'RBFKernel'
        },
        'num_inducing': 512,
        'noise': 0.5,
        'learn_noise': True,
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
        'learning_rate': 0.1,
        'epochs': 50,
    },
    'wandb': {
        'watch': True,
        'group': 'test',
        'entity': 'bogp',
        'project': 'soft-gp-2',
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
    
    # Evaluate
    eval_gp(model, likelihood, test_dataset, device=CONFIG.model.device)
