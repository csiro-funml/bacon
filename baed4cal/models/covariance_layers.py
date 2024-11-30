from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.utils import vec_to_tril_matrix

from abc import abstractmethod


def make_cholesky(x: torch.Tensor):
    """Transforms a (..., d*(d+1)/2) array into a (..., d, d) lower triangular matrix array"""
    tril = vec_to_tril_matrix(x)  # makes lower triangular matrix
    tril.diagonal(
        dim1=-2, dim2=-1
    ).exp_()  # exponentiates diagonal elements in place to ensure they are positive,
    # so that Cholesky factor is unique
    return tril


class CovarianceLayer(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given an unconstrained parameter vector, return a positive-definite covariance matrix"""
        raise NotImplementedError()


class DiagonalCovarianceLayer(CovarianceLayer):
    def __init__(self):
        super().__init__()

    def forward(self, raw_diagonal: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(raw_diagonal.exp())


class IsotropicCovarianceLayer(DiagonalCovarianceLayer):
    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, raw_variance: torch.Tensor) -> torch.Tensor:
        expanded = raw_variance.expand(*raw_variance.shape, self.d)
        return super().forward(expanded)


class CholeskyLayer(nn.Module):
    def __init__(self, d: Optional[int] = None):
        super().__init__()
        self.d = d

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        if self.d is not None:
            assert (
                parameters.shape[-1] == self.d * (self.d + 1) // 2
            ), "Parameters must represent lower-triangular entries"
        chol = make_cholesky(parameters)
        return chol


class CholeskyCovarianceLayer(CovarianceLayer):
    def __init__(self, d: Optional[int] = None):
        super().__init__()
        self.d = d
        self.chol_layer = CholeskyLayer(d)

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        chol = self.chol_layer(parameters)
        cov_mat = chol @ chol.transpose(-1, -2)
        return cov_mat
