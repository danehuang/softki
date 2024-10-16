from gpytorch.functions import pivoted_cholesky
import torch


def _default_preconditioner(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def woodbury_preconditioner(A: torch.Tensor, k=10, device="cpu", noise=1e-3):
    # Greedy nystrom!
    L_k = pivoted_cholesky(A, rank=k)
    
    def preconditioner(v: torch.Tensor) -> torch.Tensor:
        # sigma_sq = 1e-2  # Regularization term, can be adjusted based on problem
        # Woodbury-based preconditioner P^{-1}v
        P_inv_v = (v / noise) - torch.matmul(
            L_k,
            torch.linalg.solve(
                torch.eye(L_k.size(1), device=device) + (1. / noise) * torch.matmul(L_k.T, L_k),
                torch.matmul(L_k.T, v)
            )
        )
        return P_inv_v
    
    return preconditioner
