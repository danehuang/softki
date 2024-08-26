import torch


def _default_preconditioner(x: torch.Tensor) -> torch.Tensor:
    return x.clone()
