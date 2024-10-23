from typing import *

import gpytorch
import gpytorch.constraints
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.means import ZeroMean
import torch


# =============================================================================
# Exact GP
# =============================================================================

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel=MaternKernel(nu=1.5), use_scale=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = ZeroMean()
        self.use_scale = use_scale
        if self.use_scale:
            self.covar_module = ScaleKernel(kernel)
        else:
            self.covar_module = kernel
       
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def get_noise(self) -> float:
        return self.likelihood.noise_covar.noise.cpu()

    def get_lengthscale(self) -> float:
        if self.use_scale:
            return self.covar_module.base_kernel.lengthscale.cpu()
        else:
            return self.covar_module.lengthscale.cpu()

    def get_outputscale(self) -> float:
        if self.use_scale:
            return self.covar_module.outputscale.cpu()
        else:
            return 1.
