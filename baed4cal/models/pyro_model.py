from typing import Optional
import gpytorch
import gpytorch.distributions as gd
import pyro
import pyro.distributions as pd
import torch
from gpytorch.kernels import Kernel
from gpytorch.means import Mean, ZeroMean
from gpytorch.likelihoods import (
    Likelihood,
    GaussianLikelihood,
    FixedNoiseGaussianLikelihood,
)


from baed4cal.strategies.observations import ObservationsDataset
from baed4cal.models.joint_model import JointFullGP


class PyroJointGP:
    def __init__(self, theta_prior: pd.Distribution, model: JointFullGP) -> None:
        self.theta_prior = theta_prior
        self.joint_model = model

    def model(self):
        theta = pyro.sample("theta", self.theta_prior)
        with gpytorch.settings.trace_mode():
            y_dist = self.joint_model.marginal(theta)
        obs = pyro.sample("y", y_dist, obs=self.joint_model.combined_outcomes)
        return obs


class PyroLatentInputExactGP(gpytorch.models.GP):
    def __init__(
        self,
        theta_prior: pd.Distribution,
        covar_module: Kernel,
        mean_module: Optional[Mean] = None,
        likelihood: Optional[Likelihood] = None,
        observation_noise_sd: Optional[float] = None,
        simulation_noise_sd: float = 1e-2,
    ):
        super().__init__()
        if mean_module is None:
            mean_module = ZeroMean()
        if likelihood is None:
            likelihood = GaussianLikelihood()
        if isinstance(likelihood, FixedNoiseGaussianLikelihood):
            if observation_noise_sd is None:
                raise ValueError(
                    "Observation noise level required for 'FixedNoiseGaussianLikelihood'"
                )
            if not torch.is_tensor(observation_noise_sd):
                observation_noise_sd = torch.as_tensor(observation_noise_sd)
            if not torch.is_tensor(simulation_noise_sd):
                simulation_noise_sd = torch.as_tensor(simulation_noise_sd)

        self.covar_module = covar_module
        self.mean_module = mean_module
        self.theta_prior = theta_prior
        self.likelihood = likelihood  # TODO: Split real and simulations likelihood
        self.observation_noise_sd = observation_noise_sd
        self.simulation_noise_sd = simulation_noise_sd

    def forward(self, x: torch.Tensor):
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gd.MultivariateNormal(mean, covariance)

    def model(
        self,
        real_inputs: torch.Tensor,
        sim_inputs: torch.Tensor,
        real_observations: Optional[torch.Tensor] = None,
        sim_observations: Optional[torch.Tensor] = None,
    ):
        theta = pyro.sample("theta", self.theta_prior)

        completed_inputs = torch.cat(
            [real_inputs, theta.expand(*real_inputs.shape[:-1], *theta.shape)], -1
        )
        all_inputs = torch.cat([completed_inputs, sim_inputs])

        if real_observations is not None and sim_observations is not None:
            all_observations = torch.cat([real_observations, sim_observations])
        else:
            all_observations = None

        f_dist = self.forward(all_inputs)

        lik_kwargs = dict()
        if isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
            if real_observations is not None and sim_observations is not None:
                noise_sds = torch.cat(
                    [
                        self.observation_noise_sd.expand_as(real_observations),
                        self.simulation_noise_sd.expand_as(sim_observations),
                    ]
                )
                lik_kwargs["noise"] = noise_sds.pow(2)
        y_marginal = self.likelihood.marginal(f_dist, **lik_kwargs)

        y = pyro.sample("y", y_marginal, obs=all_observations)

        return y


class PyroKOHGP(PyroLatentInputExactGP):
    def __init__(
        self,
        theta_prior: pd.Distribution,
        covar_module: Kernel,
        err_covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        likelihood: Likelihood | None = None,
        scale_prior: pd.Distribution | None = None,
        observation_noise_sd: float | None = None,
        simulation_noise_sd: float = 0.01,
        observation_noise_prior: pd.Distribution | None = None,
    ):
        super().__init__(
            theta_prior,
            covar_module,
            mean_module,
            likelihood,
            observation_noise_sd,
            simulation_noise_sd,
        )
        if scale_prior is None:
            scale_prior = pd.LogNormal(1.0, 1.0)
        self.scale_prior = scale_prior
        self.err_covar_module = err_covar_module
        if observation_noise_prior is None:
            observation_noise_prior = pd.Gamma(
                concentration=1.1, rate=0.05
            )  # BoTorch default noise prior
        self.observation_noise_prior = observation_noise_prior

    def model(self, sim_data: ObservationsDataset, real_data: ObservationsDataset):
        sampled_model = self.pyro_sample_from_prior()

        theta = pyro.sample("theta", self.theta_prior)
        real_inputs = real_data.points

        completed_inputs = torch.cat(
            [real_inputs, theta.expand(*real_inputs.shape[:-1], *theta.shape)], -1
        )

        sim_scale = pyro.sample("sim_scale", self.scale_prior)
        cov_sim_sim = sampled_model.covar_module(sim_data.points).to_dense()
        if self.err_covar_module:
            cov_err = sampled_model.err_covar_module(real_data.points).to_dense()
        else:
            cov_err = torch.zeros([])

        cov_sim_real = sampled_model.covar_module(
            sim_data.points, completed_inputs
        ).to_dense()
        cov_real_real = sampled_model.covar_module(completed_inputs).to_dense()

        if self.observation_noise_sd is not None:
            obs_noise_var = self.observation_noise_sd**2
        else:
            obs_noise_var = pyro.sample("obs_noise_var", self.observation_noise_prior)

        y_cov = torch.cat(
            [
                torch.cat(
                    [
                        cov_sim_sim
                        + torch.eye(*cov_sim_sim.shape) * self.simulation_noise_sd**2,
                        cov_sim_real * sim_scale,
                    ],
                    -1,
                ),
                torch.cat(
                    [
                        cov_sim_real.T * sim_scale,
                        cov_err
                        + cov_real_real * sim_scale**2
                        + torch.eye(*cov_real_real.shape) * obs_noise_var,
                    ],
                    -1,
                ),
            ],
            -2,
        )
        all_inputs = torch.cat([sim_data.points, completed_inputs], 0)
        all_observations = torch.cat([sim_data.outcomes, real_data.outcomes])

        y_mean = sampled_model.mean_module(all_inputs)

        y_dist = gd.MultivariateNormal(y_mean, y_cov)

        y = pyro.sample("y", y_dist, obs=all_observations)

        return y
