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

# Gpytorch and linear_operator
import gpytorch 
import gpytorch.constraints
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
import linear_operator
from linear_operator.operators.dense_linear_operator import DenseLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.settings import max_cholesky_size

# Our imports
from gp.mll import HutchinsonPseudoLoss
from linear_solver.cg import linear_cg


# =============================================================================
# Soft GP
# =============================================================================

class SoftGP(torch.nn.Module):
    def __init__(
        self,
        kernel: Callable,
        inducing_points: torch.Tensor,
        noise=1e-3,
        learn_noise=False,
        device="cpu",
        dtype=torch.float32,
        solve_method="solve",
        mll_approx="hutchinson",
        fit_chunk_size=1024,
        max_cg_iter=50,
        cg_tolerance=0.5,
    ) -> None:
        # Argument checking 
        methods = ["solve", "cholesky", "cg"]
        if not solve_method in methods:
            raise ValueError(f"Method {solve_method} should be in {methods} ...")
        
        # Check devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda"]
            for i in range(torch.cuda.device_count()):
                devices += [f"cuda:{i}"]
        if not device in devices:
            raise ValueError(f"Device {device} should be in {devices} ...")

        # Create torch module
        super(SoftGP, self).__init__()

        # Misc
        self.device = device
        self.dtype = dtype
        
        # Mll approximation settings
        self.solve_method = solve_method
        self.mll_approx = mll_approx
        self.fit_chunk_size = fit_chunk_size

        # Noise
        self.noise_constraint = gpytorch.constraints.Positive()
        noise = torch.tensor([noise], dtype=self.dtype, device=self.device)
        noise = self.noise_constraint.inverse_transform(noise)
        if learn_noise:
            self.register_parameter("raw_noise", torch.nn.Parameter(noise))
        else:
            self.raw_noise = noise

        # Kernel
        if isinstance(kernel, ScaleKernel):
            self.kernel = kernel.to(self.device)
        else:
            self.kernel = kernel.initialize(lengthscale=1).to(self.device)

        # Inducing points
        self.register_parameter("inducing_points", torch.nn.Parameter(inducing_points))

        # Interpolation
        def softmax_interp(X: torch.Tensor, sigma_values: torch.Tensor) -> torch.Tensor:
            distances = torch.linalg.vector_norm(X - sigma_values, ord=2, dim=-1)
            softmax_distances = torch.softmax(-distances, dim=-1)
            return softmax_distances
        self.interp = softmax_interp
        
        # Fit artifacts
        self.alpha = None
        self.K_zz_alpha = None

        # CG solver params
        self.max_cg_iter = max_cg_iter
        self.cg_tol = cg_tolerance
        self.x0 = None
        
    # -----------------------------------------------------
    # Soft GP Helpers
    # -----------------------------------------------------
    
    @property
    def noise(self):
        return self.noise_constraint.transform(self.raw_noise)

    def _mk_cov(self, z: torch.Tensor) -> torch.Tensor:
        return self.kernel(z, z).evaluate()
    
    def _interp(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(1).expand(-1, self.inducing_points.shape[0], -1)
        W_xz = self.interp(x_expanded, self.inducing_points)
        return W_xz

    # -----------------------------------------------------
    # Linear solver
    # -----------------------------------------------------

    def _solve_system(
        self,
        kxx: linear_operator.operators.LinearOperator,
        full_rhs: torch.Tensor,
        x0: torch.Tensor = None,
        forwards_matmul: Callable = None,
        precond: torch.Tensor = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            try:
                if self.solve_method == "solve":
                    solve = torch.linalg.solve(kxx, full_rhs)
                elif self.solve_method == "cholesky":
                    L = torch.linalg.cholesky(kxx)
                    solve = torch.cholesky_solve(full_rhs, L)
                elif self.solve_method == "cg":
                    # Source: https://github.com/AndPotap/halfpres_gps/blob/main/mlls/mixedpresmll.py
                    solve = linear_cg(
                        forwards_matmul,
                        full_rhs,
                        max_iter=self.max_cg_iter,
                        tolerance=self.cg_tol,
                        initial_guess=x0,
                        preconditioner=precond,
                    )
                else:
                    raise ValueError(f"Unknown method: {self.solve_method}")
            except RuntimeError as e:
                print("Fallback to pseudoinverse: ", str(e))
                solve = torch.linalg.pinv(kxx.evaluate()) @ full_rhs

        # Apply torch.nan_to_num to handle NaNs from percision limits 
        return torch.nan_to_num(solve)

    # -----------------------------------------------------
    # Marginal Log Likelihood
    # -----------------------------------------------------

    def mll(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the marginal log likelihood of a soft GP:
            
            log p(y) = log N(y | mu_x, Q_xx)

            where
                mu_X: mean of soft GP
                Q_XX = W_xz K_zz W_zx

        Args:
            X (torch.Tensor): B x D tensor of inputs where each row is a point.
            y (torch.Tensor): B tensor of targets.

        Returns:
            torch.Tensor:  log p(y)
        """        
        # Construct covariance matrix components
        K_zz = self._mk_cov(self.inducing_points)
        W_xz = self._interp(X)
        
        if self.mll_approx == "exact":
            # [Note]: Compute MLL with a multivariate normal. Unstable for float.
            # 1. mean: 0
            mean = torch.zeros(len(X), dtype=self.dtype, device=self.device)
            
            # 2. covariance: Q_xx = (W_xz L) (L^T W_xz) + noise I  where K_zz = L L^T
            L = psd_safe_cholesky(K_zz)
            LK = (W_xz @ L).to(device=self.device)
            cov_diag = self.noise * torch.ones(len(X), dtype=self.dtype, device=self.device)

            # 3. N(mu, Q_xx)
            normal_dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(mean, LK, cov_diag, validate_args=None)
            
            # 4. log N(y | mu, Q_xx)
            return normal_dist.log_prob(y)
        elif self.mll_approx == "hutchinson":
            # [Note]: Compute MLL with Hutchinson's trace estimator
            # 1. mean: 0
            mean = torch.zeros(len(X), dtype=self.dtype, device=self.device)
            
            # 2. covariance: Q_xx = W_xz K_zz K_zx + noise I
            cov_mat = W_xz @ K_zz @ W_xz.T 
            cov_mat += torch.eye(cov_mat.shape[1], dtype=self.dtype, device=self.device) * self.noise

            # 3. log N(y | mu, Q_xx) \appox 
            hutchinson_mll = HutchinsonPseudoLoss(self, num_trace_samples=10)
            return hutchinson_mll(mean, cov_mat, y)
        else:
            raise ValueError(f"Unknown MLL approximation method: {self.mll_approx}")
        
    # -----------------------------------------------------
    # Fit
    # -----------------------------------------------------

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fits a SoftGP to dataset (X, y). That is, solve:

                (hat{K}_zx @ noise^{-1}) y = (K_zz + hat{K}_zx @ noise^{-1} @ hat{K}_xz) \alpha
        
            for \alpha where
            1. inducing points z are fixed,
            2. hat{K}_zx = K_zz W_zx, and
            3. hat{K}_xz = hat{K}_zx^T.

        Args:
            X (torch.Tensor): N x D tensor of inputs
            y (torch.Tensor): N tensor of outputs
        """        
        # Prepare inputs
        N = len(X)
        M = len(self.inducing_points)
        X = X.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)

        # Form K_zz
        K_zz = self._mk_cov(self.inducing_points)

        # Construct A and b for linear solve
        #   A = (K_zz + hat{K}_zx @ noise^{-1} @ hat{K}_xz)
        #   b = (hat{K}_zx @ noise^{-1}) y
        if X.shape[0] * X.shape[1] <= 32768:
            # Case: "small" X
            # Form estimate \hat{K}_xz ~= W_xz K_zz
            W_xz = self._interp(X)
            hat_K_xz = W_xz @ K_zz
            hat_K_zx = hat_K_xz.T
            
            # Form A and b
            Lambda_inv_diag = (1 / self.noise) * torch.ones(N, dtype=self.dtype).to(self.device)
            A = K_zz + hat_K_zx @ (Lambda_inv_diag.unsqueeze(1) * hat_K_xz)
            b = hat_K_zx @ (Lambda_inv_diag * y)
        else:
            # Case: "large" X
            with torch.no_grad():
                # Initialize outputs
                A = torch.zeros(M, M, dtype=self.dtype, device=self.device)
                b = torch.zeros(M, dtype=self.dtype, device=self.device)
                
                # Initialize temporary values
                fit_chunk_size = self.fit_chunk_size
                batches = int(np.floor(N / fit_chunk_size))
                Lambda_inv = (1 / self.noise) * torch.eye(fit_chunk_size, dtype=self.dtype, device=self.device)
                tmp1 = torch.zeros(fit_chunk_size, M, dtype=self.dtype, device=self.device)
                tmp2 = torch.zeros(M, M, dtype=self.dtype, device=self.device)
                tmp3 = torch.zeros(fit_chunk_size, dtype=self.dtype, device=self.device)
                tmp4 = torch.zeros(M, dtype=self.dtype, device=self.device)
                tmp5 = torch.zeros(M, dtype=self.dtype, device=self.device)
                
                # Compute batches
                for i in range(batches):
                    # Update A: A += W_zx @ Lambda_inv @ W_xz
                    X_batch = X[i*fit_chunk_size:(i+1)*fit_chunk_size]
                    W_xz = self._interp(X_batch)
                    W_zx = W_xz.T
                    torch.matmul(Lambda_inv, W_xz, out=tmp1)
                    torch.matmul(W_zx, tmp1, out=tmp2)
                    A.add_(tmp2)
                    
                    # Update b: b += K_zz @ W_zx @ (Lambda_inv @ Y[i*batch_size:(i+1)*batch_size])
                    torch.matmul(Lambda_inv, y[i*fit_chunk_size:(i+1)*fit_chunk_size], out=tmp3)
                    torch.matmul(W_zx, tmp3, out=tmp4)
                    torch.matmul(K_zz, tmp4, out=tmp5)
                    b.add_(tmp5)
                
                # Compute last batch
                if N - (i+1)*fit_chunk_size > 0:
                    Lambda_inv = (1 / self.noise) * torch.eye(N - (i+1)*fit_chunk_size, dtype=self.dtype, device=self.device)
                    X_batch = X[(i+1)*fit_chunk_size:]
                    W_xz = self._interp(X_batch)
                    A += W_xz.T @ Lambda_inv @ W_xz
                    b += K_zz @ W_xz.T @ Lambda_inv @ y[(i+1)*fit_chunk_size:]

                # Aggregate result
                A = K_zz + K_zz @ A @ K_zz

        # Safe solve A \alpha = b
        A = DenseLinearOperator(A)
        self.alpha = self._solve_system(
            A,
            b.unsqueeze(1),
            x0=torch.zeros_like(b),
            forwards_matmul=A.matmul,
            precond=None
        )

        # Store for fast prediction
        self.K_zz_alpha = K_zz @ self.alpha

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    def pred(self, x_star: torch.Tensor) -> torch.Tensor:
        """Give the posterior predictive:
        
            p(y_star | x_star, X, y) 
                = W_star_z (K_zz \alpha)
                = W_star_z K_zz (K_zz + hat{K}_zx @ noise^{-1} @ hat{K}_xz)^{-1} (hat{K}_zx @ noise^{-1}) y

        Args:
            x_star (torch.Tensor): B x D tensor of points to evaluate at.

        Returns:
            torch.Tensor: B tensor of p(y_star | x_star, X, y).
        """        
        W_star_z = self._interp(x_star)
        return torch.matmul(W_star_z, self.K_zz_alpha).squeeze(-1)


# =============================================================================
# Train and Test Harness
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
    