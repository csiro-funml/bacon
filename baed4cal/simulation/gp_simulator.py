from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.means import Mean, ZeroMean
from scipy.stats.qmc import MultivariateNormalQMC


class GPSimulator(nn.Module):
    def __init__(
        self,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        likelihood: Optional[Likelihood] = None,
        default_noise_variance: float = 1e-2,
    ) -> None:
        super().__init__()
        self.gp_model: SingleTaskGP | None = None
        if covar_module is None:
            covar_module = RBFKernel()
            if covar_module.has_lengthscale:
                covar_module.lengthscale = 0.25
        self.covar_module = covar_module
        if mean_module is None:
            mean_module = ZeroMean()
        self.mean_module = mean_module
        if likelihood is None:
            likelihood = GaussianLikelihood()
            likelihood.noise = default_noise_variance
        self.likelihood = likelihood
        self.eval()
        self.requires_grad_(False)

    def reset(self):
        del self.gp_model
        self.gp_model = None

    def _update(self, x, y):
        # Update internal GP
        with torch.no_grad():
            if y.dim() < 2:
                y = y.view(*x.shape[:-1], 1)
            if self.gp_model is None:
                self.gp_model = SingleTaskGP(
                    x,
                    y,
                    covar_module=self.covar_module,
                    mean_module=self.mean_module,
                    likelihood=self.likelihood,
                )
                self.gp_model.eval()
                self.gp_model.requires_grad_(False)
            else:
                self.gp_model = self.gp_model.condition_on_observations(x, y)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.atleast_2d(x)
        if self.gp_model is None:
            f_dist = MultivariateNormal(self.mean_module(x), self.covar_module(x))
        else:
            f_dist = self.gp_model(x)
        # Sample observation
        y_dist = self.likelihood(f_dist)
        y = y_dist.rsample()
        self._update(x, y)
        return y


class GPSampleSimulator(GPSimulator):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
    ):
        super().__init__(covar_module, mean_module)
        # inducing_dist = MultivariateNormal(
        #     self.mean_module(inducing_points), self.covar_module(inducing_points)
        # )
        n_inducing = inducing_points.shape[-2]
        inducing_dist = MultivariateNormalQMC(mean=np.zeros(n_inducing))
        raw_samples = torch.from_numpy(inducing_dist.random()).to(inducing_points)
        inducing_weights = (
            self.covar_module(inducing_points)
            .cholesky()
            .transpose(-1, -2)
            .solve_triangular(raw_samples.T, upper=True)
        ).squeeze(-1)
        # diffs = (inducing_values - self.mean_module(inducing_points)).unsqueeze(-1)
        # inducing_weights = torch.linalg.solve_triangular(
        #     inducing_dist.scale_tril, diffs, upper=False
        # ).squeeze(-1)
        self.inducing_points = inducing_points
        self.inducing_weights = inducing_weights

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.atleast_2d(x)
        cov_x_u = self.covar_module(x, self.inducing_points)
        f_x = self.mean_module(x) + cov_x_u @ self.inducing_weights
        return f_x
