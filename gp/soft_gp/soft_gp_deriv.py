# System/Library imports
from typing import *

# Common data science imports
import numpy as np
import torch
from torchviz import make_dot


# Gpytorch and linear_operator
import gpytorch 
import gpytorch.constraints
from gpytorch.kernels import ScaleKernel
import linear_operator
from linear_operator.operators.dense_linear_operator import DenseLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky

# Our imports
from gp.soft_gp.mll import HutchinsonPseudoLoss
from linear_solver.cg import linear_cg


# =============================================================================
# Soft GP
# =============================================================================

class SoftGPDeriv(torch.nn.Module):
    def __init__(
        self,
        kernel: Callable,
        inducing_points: torch.Tensor,
        noise=1e-3,
        learn_noise=False,
        use_scale=False,
        device="cpu",
        dtype=torch.float32,
        solver="solve",
        max_cg_iter=50,
        cg_tolerance=0.5,
        mll_approx="hutchinson",
        fit_chunk_size=1024,
        use_qr=False,
    ) -> None:
        # Argument checking 
        methods = ["solve", "cholesky", "cg"]
        if not solver in methods:
            raise ValueError(f"Method {solver} should be in {methods} ...")
        
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
        self.solve_method = solver
        self.mll_approx = mll_approx

        # Fit settings
        self.use_qr = use_qr
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
        self.use_scale = use_scale
        if use_scale:
            self.kernel = ScaleKernel(kernel).to(self.device)
        else:
            self.kernel = kernel.to(self.device)

        # Inducing points
        self.register_parameter("inducing_points", torch.nn.Parameter(inducing_points))
        
        # Fit artifacts
        M = len(self.inducing_points)
        self.U_zz = torch.zeros((M, M), dtype=self.dtype, device=self.device)
        self.K_zz_alpha = torch.zeros(M, dtype=self.dtype, device=self.device)
        self.alpha = torch.zeros((M, 1), dtype=self.dtype, device=self.device)

        # QR artifacts
        self.fit_buffer = None
        self.fit_b = None
        self.Q = None
        self.R = None

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

    def get_lengthscale(self) -> float:
        if self.use_scale:
            return self.kernel.base_kernel.lengthscale.cpu()
        else:
            return self.kernel.lengthscale.cpu()
        
    def get_outputscale(self) -> float:
        if self.use_scale:
            return self.kernel.outputscale.cpu()
        else:
            return 1.

    def _mk_cov(self, z: torch.Tensor) -> torch.Tensor:
        return self.kernel(z, z).evaluate()
    
    def _interp(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inducing_points
        x_expanded = x.unsqueeze(1).expand(-1, z.shape[0], -1)        
        diff = x_expanded - z
        distances = torch.linalg.vector_norm(diff, ord=2, dim=-1)
        softmax_distances = torch.softmax(-distances, dim=-1)
        W_xz = softmax_distances
        softmax_deriv = softmax_distances * (1 - softmax_distances)
        W_xz_deriv = -(softmax_deriv.unsqueeze(-1) * diff / distances.unsqueeze(-1)).transpose(2, 1)
        W_xz_full = torch.cat([W_xz, W_xz_deriv.reshape(-1, z.shape[0])], dim=0)
        return W_xz_full

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
        return_pinv: bool = False,
    ) -> torch.Tensor:
        use_pinv = False
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
                use_pinv = True

        # Apply torch.nan_to_num to handle NaNs from percision limits 
        solve = torch.nan_to_num(solve)
        return (solve, use_pinv) if return_pinv else solve

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

    def _qr_solve_fit(self, M, N, X, y, K_zz):
        if X.shape[0] * X.shape[1] <= 32768:
            # Compute: W_xz K_zz
            W_xz = self._interp(X)
            self.fit_buffer[:N,:] = W_xz @ K_zz
        else:
            if self.fit_buffer is None:
                self.fit_buffer = torch.zeros((N + M, M), dtype=self.dtype, device=self.device)
                self.fit_b = torch.zeros(N, dtype=self.dtype, device=self.device)

            # Compute: W_xz K_zz in a batched fashion
            with torch.no_grad():
                # Compute batches
                fit_chunk_size = self.fit_chunk_size
                batches = int(np.floor(N / fit_chunk_size))
                for i in range(batches):
                    start = i*fit_chunk_size
                    end = (i+1)*fit_chunk_size
                    X_batch = X[start:end,:]
                    W_xz = self._interp(X_batch)
                    torch.matmul(W_xz, K_zz, out=self.fit_buffer[start:end,:])
                
                start = (i+1)*fit_chunk_size
                if N - start > 0:
                    X_batch = X[start:]
                    W_xz = self._interp(X_batch)
                    torch.matmul(W_xz, K_zz, out=self.fit_buffer[start:N,:])
        
                self.W_xz_cpu = self.fit_buffer[:N, :].detach().cpu()

        with torch.no_grad():
            # B^T = [(Lambda^{-1/2} \hat{K}_xz) U_zz ]
            psd_safe_cholesky(K_zz, out=self.U_zz, upper=True, max_tries=10)
            # Lambda_half_inv_diag = (1 / torch.sqrt(self.noise)) * torch.ones(N, dtype=self.dtype).to(self.device)
            # self.fit_buffer[:N,:] = Lambda_half_inv_diag.unsqueeze(1) * hat_K_xz
            self.fit_buffer[:N,:] *= 1 / torch.sqrt(self.noise)
            self.fit_buffer[N:,:] = self.U_zz

            if self.Q is None:
                self.Q = torch.zeros((N + M, M), dtype=self.dtype, device=self.device)
                self.R = torch.zeros((M, M), dtype=self.dtype, device=self.device)
        
            # B = QR
            torch.linalg.qr(self.fit_buffer, out=(self.Q, self.R))

            # \alpha = R^{-1} @ Q^T @ Lambda^{-1/2}b
            self.fit_b[:] = 1 / torch.sqrt(self.noise) * y
            torch.linalg.solve_triangular(self.R, (self.Q.T[:, 0:N] @ self.fit_b).unsqueeze(1), upper=True, out=self.alpha).squeeze(1)

            # Store for fast inference
            # self.K_zz_alpha = K_zz @ alpha
            torch.matmul(K_zz, self.alpha.squeeze(-1), out=self.K_zz_alpha)

        return False
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> bool:
        """Fits a SoftGP to dataset (X, y). That is, solve:

                (hat{K}_zx @ noise^{-1}) y = (K_zz + hat{K}_zx @ noise^{-1} @ hat{K}_xz) \alpha
        
            for \alpha where
            1. inducing points z are fixed,
            2. hat{K}_zx = K_zz W_zx, and
            3. hat{K}_xz = hat{K}_zx^T.

        Args:
            X (torch.Tensor): N x D tensor of inputs
            y (torch.Tensor): N tensor of outputs

        Returns:
            bool: Returns true if the pseudoinverse was used, false otherwise.
        """        
        # Prepare inputs
        N = len(X)
        M = len(self.inducing_points)
        X = X.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)

        # Form K_zz
        K_zz = self._mk_cov(self.inducing_points)

        if self.use_qr:
            # return self._qr_solve_fit(M, N, X, y, K_zz)
            return self._qr_solve_fit(M, N, X, y, K_zz)
        else:
            return self._direct_solve_fit(M, N, X, y, K_zz)

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
