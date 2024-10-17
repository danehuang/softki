
import torch 
import numpy as np 
from linear_operator.utils.cholesky import psd_safe_cholesky
global_dtype = torch.float64

class GaussianProcessRegression:
    def __init__(self, lengthscale=1.0, sigma=0.1, dtype=global_dtype, kernel=None):
        self.sigma = sigma
        self.dtype = dtype
        self.kernel = kernel(nu=1.5, lengthscale=lengthscale)

    def fit(self, X, y):
        self.X_train = X.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        K = self.kernel(X, X).to_dense() + self.sigma ** 2 * torch.eye(len(X), dtype=self.dtype)
        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(y, self.L)

    def predict(self, X):
        X = X.to(dtype=self.dtype)
        K_trans = self.kernel(X, self.X_train).to_dense()
        K = self.kernel(X, X).to_dense()
        mean = K_trans @ self.alpha
        v = torch.cholesky_solve(K_trans.t(), self.L)
        covariance = K - K_trans @ v
        
        # Detach from the computation graph 
        mean_np = mean.detach().cpu().numpy()
        covariance_np = covariance.detach().cpu().numpy()
        
        return mean_np.squeeze(), covariance_np
    
class KernelRidgeRegression:
    def __init__(self, _lambda=1.0, lengthscale=0.1, kernel=None, dtype=torch.float32):
        self._lambda = _lambda
        self.kernel = kernel(nu=1.5, lengthscale=lengthscale)
        self.dtype = dtype

    def fit(self, X, y):
        self.X_train = X.to(self.dtype)
        y = y.to(self.dtype)
        K = self.kernel(X, X).to(self.dtype)
        n = K.shape[0]
        self.alpha_vector = torch.linalg.solve(K + self._lambda * n * torch.eye(n, dtype=self.dtype), y)

    def predict(self, X):
        K = self.kernel(X.to(self.dtype), self.X_train).to(self.dtype)
        return torch.matmul(K, self.alpha_vector)    
    
class NystromKRR:
    def __init__(self, _lambda=1.0, lengthscale=0.1, kernel=None, m=10, dtype=global_dtype):
        self._lambda = _lambda
        self.kernel = kernel(nu=1.5,lengthscale=lengthscale)
        self.m = m
        self.dtype = dtype

    def fit(self, X, y):
        self.X_train = X.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        n = X.shape[0]
        
        indices = np.random.choice(n, self.m, replace=False)
        self.Z = X[indices]

        K_ZZ = self.kernel(self.Z, self.Z).to(dtype=self.dtype)
        K_XZ = self.kernel(X, self.Z).to(dtype=self.dtype)
        K_ZX = K_XZ.T
        
        A = K_ZX @ K_XZ + n * self._lambda * K_ZZ
        b = K_ZX @ y
        self.beta = torch.linalg.solve(A, b)

    def predict(self, X):
        X = X.to(dtype=self.dtype)
        K_XZ = self.kernel(X, self.Z).to(dtype=self.dtype)
        return K_XZ @ self.beta
    
class SparseVariationalGaussianProcess:
    def __init__(self, inducing_points, lengthscale=1.0, sigma=0.1, kernel=None, dtype=global_dtype):
        self.inducing_points = inducing_points.clone().detach().to(dtype=dtype)
        self.sigma = sigma
        self.kernel = kernel(nu=1.5,lengthscale=lengthscale)
        self.dtype = dtype

        if self.dtype==torch.float32:
            self.jitter = 1e-2
        else: 
            self.jitter = 1e-16

    def fit(self, X, y):
        self.y_train = y.to(dtype=self.dtype)
        self.K_zz = self.kernel(self.inducing_points, self.inducing_points).to(dtype=self.dtype).to_dense() + self.jitter * torch.eye(len(self.inducing_points), dtype=self.dtype)
        self.K_zx = self.kernel(self.inducing_points, X).to(dtype=self.dtype).to_dense()
        
        self.L_zz = psd_safe_cholesky(self.K_zz)
        self.K_zz_inv = torch.cholesky_inverse(self.L_zz)
        

    def predict(self, test_x):
        test_x = test_x.to(dtype=self.dtype)
        K_Z = self.kernel(self.inducing_points, test_x).to(dtype=self.dtype).to_dense()
        
        L = psd_safe_cholesky(self.sigma**2 * self.K_zz + self.K_zx @ self.K_zx.T)
        L_inv = torch.cholesky_inverse(L)
        
        mean_star = K_Z.T @ L_inv @ self.K_zx @ self.y_train
        
        term1 = self.kernel(test_x, test_x).to(dtype=self.dtype)
        term2 = -K_Z.T @ self.K_zz_inv @ K_Z
        
        L_full = psd_safe_cholesky(self.K_zz + (1 / self.sigma**2) * self.K_zx @ self.K_zx.T)
        L_full_inv = torch.cholesky_inverse(L_full)
        
        term3 = K_Z.T @ L_full_inv @ K_Z
        
        cov_star = term1 + term2 + term3
        
        return mean_star.squeeze(), cov_star
