from abc import abstractmethod
from typing import Optional

import torch
from torch.distributions import MultivariateNormal, Distribution
from torch.distributions.utils import tril_matrix_to_vec
import torch.nn as nn

from .covariance_layers import make_cholesky


class DensityModel(nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Distribution:
        pass


class GaussianModel(DensityModel):
    def __init__(self, param_dim: int) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.randn(param_dim))
        self.raw_covar_factor = nn.Parameter(
            torch.randn(param_dim * (param_dim + 1) // 2)
        )
        self.param_dim = param_dim

    def reset(
        self,
        mean: Optional[torch.Tensor] = None,
        covariance_matrix: Optional[torch.Tensor] = None,
    ):
        if mean is None:
            init_mean = torch.randn(self.param_dim)
        else:
            init_mean = mean
        if covariance_matrix is None:
            init_covar_factor = torch.randn(self.param_dim * (self.param_dim + 1) // 2)
        else:
            covar_facor = torch.linalg.cholesky(covariance_matrix)
            init_covar_factor = tril_matrix_to_vec(covar_facor)
        with torch.no_grad():
            self.mean.copy_(init_mean)
            self.raw_covar_factor.copy_(init_covar_factor)

    def forward(self) -> MultivariateNormal:
        chol = make_cholesky(self.raw_covar_factor)

        mvn = MultivariateNormal(self.mean, scale_tril=chol)
        return mvn
