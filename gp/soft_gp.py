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
from torch.distributions import Gamma

import wandb

# Gpytorch and linear_operator
import gpytorch 
import gpytorch.constraints
from gpytorch.distributions import MultivariateNormal
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
        interp_type: str,
        inducing_points: torch.Tensor,
        mll_noise=1e-3,
        learn_noise=False,
        device="cpu",
        dtype=torch.float32,
        method="solve",
        mll_approx = "exact",
        cg_tolerance=0.5
    ) -> None:
        # Argument checking 
        methods = ["solve", "cholesky", "cg"]
        if not method in methods:
            raise ValueError(f"Method {method} should be in {methods} ...")
        
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
        self.method = method
        
        #Mll approximation settings
        self.mll_approx = mll_approx
        self.noise_prior = Gamma(concentration=1.1, rate=0.05)
        
        # CG solver params
        self.max_cg_iter = 50
        self.cg_tol = cg_tolerance
        self.x0 = None

        # Noise
        self.noise_constraint = gpytorch.constraints.Positive()
        noise = torch.tensor([mll_noise], dtype=self.dtype, device=self.device)
        noise = self.noise_constraint.inverse_transform(noise)
        if learn_noise:
            self.register_parameter("raw_mll_noise", torch.nn.Parameter(noise))
        else:
            self.raw_mll_noise = noise

        # Kernel
        if isinstance(kernel, ScaleKernel):
            self.kernel = kernel.to(self.device)
        else:
            self.kernel = kernel.initialize(lengthscale=1).to(self.device)

        # Inducing points
        self.register_parameter("inducing_points", torch.nn.Parameter(inducing_points))

        # Interpolation
        if interp_type == "softmax":
            def softmax_interp(X: torch.Tensor, sigma_values: torch.Tensor) -> torch.Tensor:
                distances = torch.linalg.vector_norm(X - sigma_values, ord=2, dim=-1)
                softmax_distances = torch.softmax(-distances, dim=-1)
                return softmax_distances
            self.interp = softmax_interp         
        elif interp_type == "boltzmann":
            self.T = torch.nn.Parameter(torch.tensor(1.0))
            def boltzmann_interp(X: torch.Tensor, sigma_values: torch.Tensor) -> torch.Tensor:
                distances = torch.linalg.vector_norm(X - sigma_values, ord=2, dim=-1)
                exp_distances = torch.exp(distances / self.T)
                Z_theta = torch.sum(exp_distances, dim=-1, keepdim=True)
                normalized_distances = exp_distances / Z_theta
                
                return normalized_distances
            self.interp = boltzmann_interp
        else:
            raise ValueError(f"Interpolation {interp_type} not supported ...")
        
        # Fit artifacts
        self.alpha = None
        self.K_zz = None
        self.K_zz_alpha = None
        
    # -----------------------------------------------------
    # GP Helpers
    # -----------------------------------------------------
    
    @property
    def mll_noise(self):
        return self.noise_constraint.transform(self.raw_mll_noise)

    def _mk_cov(self, z: torch.Tensor) -> torch.Tensor:
        return self.kernel(z, z).evaluate()
    
    def _interp(self, x):
        x_expanded = x.unsqueeze(1).expand(-1, self.inducing_points.shape[0], -1)
        W_xz = self.interp(x_expanded, self.inducing_points)
        return W_xz

    # -----------------------------------------------------
    # Linear solve
    # -----------------------------------------------------

    def _solve_system(
        self,
        kxx: linear_operator.operators.LinearOperator,
        x0: torch.Tensor,
        forwards_matmul: Callable,
        full_rhs: torch.Tensor,
        precond=None
    ) -> torch.Tensor:
        with torch.no_grad():
            try:
                if self.method == "solve":
                    solve = torch.linalg.solve(kxx, full_rhs)
                elif self.method == "cholesky":
                    L = torch.linalg.cholesky(kxx)
                    solve = torch.cholesky_solve(full_rhs, L)
                elif self.method == "cg":
                    # Source: https://github.com/AndPotap/halfpres_gps/blob/main/mlls/mixedpresmll.py
                    solve = linear_cg(
                        forwards_matmul,
                        full_rhs,
                        initial_guess=x0,
                        max_iter=self.max_cg_iter,
                        tolerance=self.cg_tol,
                        preconditioner=precond
                    )
                else:
                    raise ValueError(f"Unknown method: {self.method}")
            except RuntimeError as e:
                print("Fallback to pseudoinverse: ", str(e))
                solve = torch.linalg.pinv(kxx.evaluate()) @ full_rhs

        # Apply torch.nan_to_num to handle NaNs from percision limits 
        return torch.nan_to_num(solve)

    # -----------------------------------------------------
    # Marginal Log Likelihood
    # -----------------------------------------------------

    def mll(self, Z: torch.Tensor, X: torch.Tensor, y: torch.Tensor, mll_approx="hutchinson") -> torch.Tensor:        
        # Construct covariance matrix components
        K_zz = self._mk_cov(Z)
        W_xz = self._interp(X)
        
        if mll_approx == "exact":
            # Note: Unstable for float.
            L = psd_safe_cholesky(K_zz)
            LK = (W_xz @ L).to(device=self.device)
            mean = torch.zeros(len(X), dtype=self.dtype, device=self.device)
            cov_diag = self.mll_noise * torch.ones(len(X), dtype=self.dtype, device=self.device)
            normal_dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(mean, LK, cov_diag, validate_args=None)
            return normal_dist.log_prob(y)

            # mean = torch.zeros(len(X), dtype=self.dtype, device=self.device)
            # function_dist = MultivariateNormal(mean, cov_mat)
            # return function_dist.log_prob(y)
        elif mll_approx == "hutchinson":
            # Construct covariance matrix
            cov_mat = W_xz @ K_zz @ W_xz.T 
            cov_mat += torch.eye(cov_mat.shape[1], dtype=self.dtype, device=self.device) * self.mll_noise         
            self.cov_mat = cov_mat

            # Compute estimate of MLL
            mean = torch.zeros(len(X), dtype=self.dtype, device=self.device)
            mll = HutchinsonPseudoLoss(self, cg_iters=self.max_cg_iter, cg_tolerance=self.cg_tol, num_trace_samples=10)      
            return mll(mean, cov_mat, y)
        else:
            raise ValueError(f"Unknown MLL approximation method: {mll_approx}")
        
    # -----------------------------------------------------
    # Fit
    # -----------------------------------------------------

    def fit(self, X: torch.Tensor, Y: torch.Tensor, batch=False) -> None:
        N = len(X)
        M = len(self.inducing_points)
        X, Y = X.to(self.device, dtype=self.dtype), Y.to(self.device, dtype=self.dtype)

        # Form K_zz
        self.K_zz = self._mk_cov(self.inducing_points)

        # Form estimate \hat{K}_xz ~= W_xz K_zz
        if batch or X.shape[0] * X.shape[1] > 32768:
            with torch.no_grad():
                batch_size = 1024
                batches = int(np.floor(N / batch_size))
                A = torch.zeros(M, M, dtype=self.dtype, device=self.device)
                b = torch.zeros(M, dtype=self.dtype, device=self.device)
                Lambda_inv = (1 / self.mll_noise) * torch.eye(batch_size, dtype=self.dtype, device=self.device)
                tmp1 = torch.zeros(batch_size, M, dtype=self.dtype, device=self.device)
                tmp2 = torch.zeros(M, M, dtype=self.dtype, device=self.device)
                tmp3 = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
                tmp4 = torch.zeros(M, dtype=self.dtype, device=self.device)
                tmp5 = torch.zeros(M, dtype=self.dtype, device=self.device)
                for i in range(batches):
                    x = X[i*batch_size:(i+1)*batch_size]
                    
                    # [Note]:
                    #   A += W_zx @ Lambda_inv @ W_xz
                    W_xz = self._interp(x)
                    W_zx = W_xz.T
                    torch.matmul(Lambda_inv, W_xz, out=tmp1)
                    torch.matmul(W_zx, tmp1, out=tmp2)
                    A.add_(tmp2)
                    
                    # [Note]:
                    #   b += self.K_zz @ W_zx @ (Lambda_inv @ Y[i*batch_size:(i+1)*batch_size])
                    torch.matmul(Lambda_inv, Y[i*batch_size:(i+1)*batch_size], out=tmp3)
                    torch.matmul(W_zx, tmp3, out=tmp4)
                    torch.matmul(self.K_zz, tmp4, out=tmp5)
                    b.add_(tmp5)
                
                if N - (i+1)*batch_size > 0:
                    Lambda_inv = (1 / self.mll_noise) * torch.eye(N - (i+1)*batch_size, dtype=self.dtype, device=self.device)
                    x = X[(i+1)*batch_size:]
                    W_xz = self._interp(x)
                    A += W_xz.T @ Lambda_inv @ W_xz
                    b += self.K_zz @ W_xz.T @ Lambda_inv @ Y[(i+1)*batch_size:]
                A = self.K_zz + self.K_zz @ A @ self.K_zz
        else:
            W_xz = self._interp(X)
            hat_K_xz = W_xz @ self.K_zz
            hat_K_zx = hat_K_xz.T
            
            # Note:
            #   A = K_zz + \hat{K}_zx @ noise^{-1} @ \hat{K}_xz
            #   B = \hat{K}_zx @ noise^{-1} @ y
            Lambda_inv_diag = (1 / self.mll_noise) * torch.ones(N, dtype=self.dtype).to(self.device)
            A = self.K_zz + hat_K_zx @ (Lambda_inv_diag.unsqueeze(1) * hat_K_xz)
            b = hat_K_zx @ (Lambda_inv_diag * Y)

        # Safe solve A \alpha = B
        kxx = DenseLinearOperator(A)
        self.alpha = self._solve_system(
            kxx=kxx,
            full_rhs=b.unsqueeze(1),
            x0=torch.zeros_like(b),
            forwards_matmul=kxx.matmul,
            precond=None
        )

        # Store for fast prediction
        self.K_zz_alpha = self.K_zz @ self.alpha

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    def pred(self, x_star: torch.Tensor) -> torch.Tensor:
        W_star_z = self._interp(x_star)
        return torch.matmul(W_star_z, self.K_zz_alpha).squeeze(-1)


# =============================================================================
# Train and Test Harness
# =============================================================================

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
        mll_noise=1e-3,
        learn_noise=False,
        num_inducing=1024,
        solver="solve",
        cg_tolerance=1e-5,
        epochs=50,
        batch_size=1024,
        lr=0.01,
        device="cuda:0",
        dtype=torch.float64,
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
        rname = f"softgp{dataset_name}_{interp_type}_{solver}_{dtype}_{num_inducing}_{batch_size}_{mll_noise}"
        wandb.init(project=project, entity="bogp", group=group, name=rname, config=config)

    # Prepare dataset
    train_features, train_labels = zip(*[(torch.tensor(features), torch.tensor(labels)) for features, labels in train_dataset])
    train_features = torch.stack(train_features).squeeze(-1)
    train_labels = torch.stack(train_labels).squeeze(-1)

    test_features, test_labels = zip(*[(torch.tensor(features), torch.tensor(labels)) for features, labels in test_dataset])
    test_features = torch.stack(test_features).squeeze(-1)
    test_labels = torch.stack(test_labels).squeeze(-1)

    # Model setup
    np.random.seed(seed)
    kmeans = KMeans(n_clusters=num_inducing)
    kmeans.fit(train_features)
    centers = kmeans.cluster_centers_
    # indices = np.random.choice(len(train_features), num_inducing, replace=False)
    # inducing_points = train_features[indices].to(dtype=dtype, device=device)
    inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    
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
    
    model = SoftGP(k, interp_type, inducing_points, dtype=dtype, device=device, mll_noise=mll_noise, learn_noise=learn_noise, method=solver, cg_tolerance=cg_tolerance)

    def filter_param(named_params, name):
        params = []
        for n, param in named_params:
            if n == name:
                continue
            params += [param]
        return params
    
    if learn_noise:
        params = model.parameters()
    else:
        params = filter_param(model.named_parameters(), "likelihood.noise_covar.raw_noise")

    # params = [
    #     {"params": model.inducing_points, "lr": 0.02},
    #     {"params": model.kernel.raw_lengthscale, "lr": 0.005},
    # ]
    # optimizer = torch.optim.Adam(params)
    optimizer = torch.optim.Adam(params, lr=lr)

    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    cov_mats = []
    for epoch in pbar:
        t1 = time.perf_counter()
        neg_mlls = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.clone().detach().to(dtype=dtype, device=device)
            y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)
            
            optimizer.zero_grad()
            with gpytorch.settings.max_root_decomposition_size(100), max_cholesky_size(int(1.e7)):
                neg_mll = -model.mll(inducing_points, x_batch, y_batch)
            neg_mlls += [-neg_mll.item()]
            neg_mll.backward()
            optimizer.step()

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

        test_rmse, nll = eval_gp(model, test_dataset, device=device)
        if watch:
            wandb.log({
                "loss": torch.tensor(neg_mlls).mean(),
                "noise": model.mll_noise.cpu(),
                "lengthscale": model.kernel.lengthscale.cpu(),
                # "shear_strength": model.shear_strength.cpu(),
                "test_rmse": test_rmse,
                "neg_mll": neg_mll,
                "nll": nll,
                "epoch_time": t2 - t1,
                "fit_time": t3 - t2,
            })

    if trace:
        np.savez(f"./figures/trace/{dataset_name}_secgp_covmats.npz", *cov_mats)

    return model


def eval_gp(model, test_dataset, device="cuda:0"):
    preds = []
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    for x_batch, y_batch in tqdm(test_loader):
        preds += [(model.pred(x_batch.to(device)) - y_batch.to(device)).detach().cpu()**2]
    rmse = torch.sqrt(torch.sum(torch.cat(preds)) / len(test_dataset)).item()
    
    if isinstance(model.kernel, ScaleKernel):
        base_kernel_lengthscale = model.kernel.base_kernel.lengthscale.cpu()
    else:
        base_kernel_lengthscale = model.kernel.lengthscale.cpu()
        
    print("RMSE:", rmse, "NOISE", model.mll_noise.cpu().item(), "LENGTHSCALE", base_kernel_lengthscale)
    return rmse, 0


if __name__ == "__main__":
    from data.get_uci import ElevatorsDataset
    
    elevators_dataset = ElevatorsDataset("../data/uci_datasets/uci_datasets/elevators/data.csv")
    train_size = int(len(elevators_dataset) * 4/9)
    val_size = int(len(elevators_dataset) * 3/9)
    test_size = len(elevators_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(elevators_dataset, [train_size, val_size, test_size])

    # device = "cuda:0"
    device = "cpu"
    model = train_gp("elevators", train_dataset, test_dataset, elevators_dataset.dim, kernel="matern", num_inducing=1024, lr=0.01, epochs=50, device=device)
    eval_gp(model, test_dataset, device=device)