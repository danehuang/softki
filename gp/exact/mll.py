from typing import *

import gpytorch
import gpytorch.constraints
import torch


# =============================================================================
# Conjugate Gradient MLL
# =============================================================================

class CGDMLL(gpytorch.mlls.ExactMarginalLogLikelihood):
    """
    Adapated from: https://github.com/AndPotap/halfpres_gps

    Copyright (c) 2022, Wesley Maddox, Andres Potapczynski, Andrew Gordon Wilson
    All rights reserved.
    """
    def __init__(self, likelihood, model, max_cg_iters=50, cg_tolerance=1e-5):
        super().__init__(likelihood=likelihood, model=model)
        self.max_cg_iters = max_cg_iters
        self.cg_tolerance = cg_tolerance

    def forward(self, function_dist, target):
        function_dist = self.likelihood(function_dist)
        mean = function_dist.mean
        cov_matrix = function_dist.lazy_covariance_matrix.evaluate()

        residual = target - mean

        # Select the solver method
        solve = self.conjugate_gradient(cov_matrix, residual)
        mll = -0.5 * (residual.squeeze() @ solve).sum() - torch.logdet(cov_matrix)
        return mll
    
    def conjugate_gradient(self, A, b, preconditioner=None):
        """
        Adapated from: https://github.com/AndPotap/halfpres_gps
        
        Copyright (c) 2022, Wesley Maddox, Andres Potapczynski, Andrew Gordon Wilson
        All rights reserved.
        """
        if preconditioner is None:
            preconditioner = torch.eye(b.size(0), device=b.device) 
        
        x = torch.zeros_like(b)
        r = b - A.matmul(x)
        z = preconditioner.matmul(r) 
        p = z.clone()
        rz_old = torch.dot(r.view(-1), z.view(-1))

        for i in range(self.max_cg_iters):
            Ap = A.matmul(p)
            alpha = rz_old / torch.dot(p.view(-1), Ap.view(-1))
            x = x + alpha * p
            r = r - alpha * Ap
            z = preconditioner.matmul(r)  
            rz_new = torch.dot(r.view(-1), z.view(-1))
            if torch.sqrt(rz_new) < self.cg_tolerance:
                break
            p = z + (rz_new / rz_old) * p
            rz_old = rz_new

        return x
    