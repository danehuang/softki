from gpytorch.distributions import MultivariateNormal
import torch
from gpytorch.functions import pivoted_cholesky
from gpytorch.settings import max_preconditioner_size
from linear_solver.preconditioner import _default_preconditioner


"""
Copyright (c) 2022, Wesley Maddox, Andres Potapczynski, Andrew Gordon Wilson
All rights reserved.
"""

class HutchinsonPseudoLoss:
    def __init__(self, model, num_trace_samples=10, vector_format="randn"):
        self.model = model
        self.x0 = None
        self.vf = vector_format
        self.num_trace_samples = num_trace_samples

    def update_x0(self, full_rhs):
        x0 = torch.zeros_like(full_rhs)
        return x0

    def forward(self, mean, cov_mat, target, *params):
        function_dist = MultivariateNormal(mean, cov_mat)
        
        full_rhs, probe_vectors = self.get_rhs_and_probes(
            rhs=target - function_dist.mean,
            num_random_probes=self.num_trace_samples
        )
        kxx = function_dist.lazy_covariance_matrix.evaluate_kernel()
        forwards_matmul = kxx.matmul

        if self.model.hutch_solver=="solve":
            with torch.no_grad():
                result = torch.linalg.solve(kxx, full_rhs)
                result = torch.nan_to_num(result)
        else:
            # Cholesky Woodbury matrix preconditioner
            k = 10  # Number of steps for the decomposition
            # input: Anysor, rank: int, error_tol: Optional[float] = None, return_pivots: bool = False        # Greedy nystrom ! 
            L_k = pivoted_cholesky(kxx, rank=k)
            def preconditioner(v):
                # sigma_sq = 1e-2  # Regularization term, can be adjusted based on problem
                # Woodbury-based preconditioner P^{-1}v
                P_inv_v = (v / 1e-3) - torch.matmul(
                    L_k,
                    torch.linalg.solve(
                        torch.eye(L_k.size(1)) + (1 / 1e-3) * torch.matmul(L_k.T, L_k),
                        torch.matmul(L_k.T, v)
                    )
                )
                return P_inv_v
            
            precond=preconditioner
            if precond is None:
                print("precon was none")
                precond = _default_preconditioner
                
            x0 = self.update_x0(full_rhs)
            result = self.model._solve_system(
                kxx,
                full_rhs,
                x0=x0,
                forwards_matmul=forwards_matmul,
                precond=precond
            )
        
        self.x0 = result.clone()
        return self.compute_pseudo_loss(forwards_matmul, result, probe_vectors, mean.shape[0])

    def compute_pseudo_loss(self, forwards_matmul, solve, probe_vectors, num_data):
        data_solve = solve[..., 0].unsqueeze(-1).contiguous()
        data_term = (-data_solve * forwards_matmul(data_solve).float()).sum(-2) / 2
        logdet_term = (
            (solve[..., 1:] * forwards_matmul(probe_vectors).float()).sum(-2)
            / (2 * probe_vectors.shape[-1])
        )
        res = -data_term - logdet_term.sum(-1)
        #num_data = function_dist.event_shape.numel()
        return res.div_(num_data)

    def get_rhs_and_probes(self, rhs, num_random_probes):
        dim = rhs.shape[-1]
        
        probe_vectors = torch.randn(dim, num_random_probes, device=rhs.device, dtype=rhs.dtype).contiguous()
        # if self.vf == "sphere":
        #     probe_vectors = probe_vectors / probe_vectors.norm(dim=0) 
        full_rhs = torch.cat((rhs.unsqueeze(-1), probe_vectors), -1)
        return full_rhs, probe_vectors

    def __call__(self, mean, cov_mat, target, *params):
        return self.forward(mean, cov_mat, target, *params)
    