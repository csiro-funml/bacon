from contextlib import nullcontext
from typing import Optional, Callable

import gpytorch.settings
import pyro
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import Kernel
from pyro.distributions import Distribution, MultivariateNormal, Gamma

from calibration.utils import combine_inputs


class PyroModel:
    def __init__(
        self,
        model: SingleTaskGP,
        prior: Distribution | Callable[..., torch.Tensor],
        sample_hp: bool = False,
        noise: float | torch.Tensor = 0.1,
    ):
        self.model = model
        self.prior = prior
        self.sample_hp = sample_hp
        if not torch.is_tensor(noise):
            noise = torch.as_tensor(noise)
        self.noise = noise

    def __call__(
        self,
        real_designs: torch.Tensor,
        real_outcomes: Optional[torch.Tensor] = None,
    ):
        model = self.model
        model = model.to(real_designs)
        model._strict(False)

        hp_batch_shape = model.covar_module.batch_shape
        if model.num_outputs > 1 and len(hp_batch_shape) > 0:
            plate_context = pyro.plate("outputs_plate", size=model.num_outputs)
        else:
            plate_context = nullcontext()

        if callable(self.prior):
            theta = pyro.deterministic("theta", self.prior()).to(real_designs)
        else:
            theta = pyro.sample("theta", self.prior).to(real_designs)
        with gpytorch.settings.trace_mode(), gpytorch.settings.debug(False):
            if self.sample_hp:
                for (
                    name,
                    parent_module,
                    prior,
                    closure,
                    inv_closure,
                ) in model.named_priors():
                    if isinstance(parent_module, Kernel):
                        prior = prior.to_event(len(prior.batch_shape))
                        with plate_context:
                            value = pyro.sample(name, prior)
                    inv_closure(parent_module, value)

            real_inputs = combine_inputs(real_designs, theta)
            real_inputs = model.transform_inputs(real_inputs)
            real_noise = self.noise  # TODO: Add option to infer noise level
            if real_outcomes is not None:
                if hasattr(model, "outcome_transform"):
                    real_outcomes, real_noise = model.outcome_transform(
                        real_outcomes, real_noise.expand_as(real_outcomes)
                    )
                if model.num_outputs > 1:
                    real_outcomes = real_outcomes.T  # for batched multi-output case
                    real_noise = real_noise.T

            pred = model(real_inputs)

            lik = model.likelihood(pred, real_inputs, noise=real_noise)
            if not isinstance(plate_context, nullcontext):
                lik = MultivariateNormal(lik.mean, scale_tril=lik.scale_tril)

            with plate_context:
                real_y = pyro.sample("real_y", lik, obs=real_outcomes)

        return real_y


class PyroErrModel(PyroModel):
    def __init__(
        self,
        model: SingleTaskGP,
        prior: Distribution | Callable[..., torch.Tensor],
        design_d: int,
        sim_scale: float | torch.Tensor | Distribution = 1.0,
        noise: float | torch.Tensor = 0.1,
        noise_prior: Distribution | None = None,
        err_kernel_class: str = "MaternKernel",
        lengthscale_prior: Distribution | None = None,
        outputscale_prior: Distribution | None = None,
        lengthscale_min: float = 1e-4,
        **err_kernel_kwargs,
    ):
        super().__init__(model, prior, noise=noise)
        err_kernel_class = getattr(gpytorch.kernels, err_kernel_class)
        err_kernel = err_kernel_class(
            active_dims=list(range(design_d)),
            ard_num_dims=design_d,
            **err_kernel_kwargs,
        )
        err_kernel.eval().requires_grad_(False)
        self.err_kernel = err_kernel
        self.lengthscale_min = lengthscale_min
        if lengthscale_prior is None:
            lengthscale_prior = Gamma(2, 0.2)
        self.lengthscale_prior = lengthscale_prior
        if outputscale_prior is None:
            outputscale_prior = Gamma(1.0, 0.01)
        self.outputscale_prior = outputscale_prior
        self.noise_prior = noise_prior
        if isinstance(sim_scale, Distribution) or torch.is_tensor(sim_scale):
            self.sim_scale = sim_scale
        else:
            self.sim_scale = torch.as_tensor(sim_scale)

    def _sample_err_hp(self, design_d: int) -> tuple[torch.Tensor]:
        model = self.model
        hp_batch_shape = model.covar_module.batch_shape
        if model.num_outputs > 1 and len(hp_batch_shape) > 0:
            plate_context = pyro.plate("hp_plate", size=model.num_outputs, dim=-1)
        else:
            plate_context = nullcontext()
        with plate_context:
            lengthscale = pyro.sample(
                "err_lengthscale",
                self.lengthscale_prior.expand([1, design_d]).to_event(2),
            )
            lengthscale = lengthscale + self.lengthscale_min
            outputscale = pyro.sample("err_outputscale", self.outputscale_prior)
        return lengthscale, outputscale

    def _error_model(
        self, real_designs: torch.Tensor, theta: torch.Tensor
    ) -> tuple[torch.Tensor]:
        design_d = real_designs.shape[-1]
        real_inputs = combine_inputs(real_designs, theta)
        real_inputs = self.model.transform_inputs(real_inputs)
        tf_designs = real_inputs[..., :design_d]
        lengthscale, outputscale = self._sample_err_hp(design_d)
        err_kernel = self.err_kernel
        err_cov = (
            err_kernel(tf_designs.div(lengthscale))
            .mul(outputscale.view(*outputscale.shape, 1, 1))
            .to_dense()
        )
        return torch.zeros(err_cov.shape[:-1]), err_cov

    def __call__(
        self,
        real_designs: torch.Tensor,
        real_outcomes: Optional[torch.Tensor] = None,
    ):
        model = self.model
        model = model.to(real_designs)

        if isinstance(self.prior, Distribution):
            theta = pyro.sample("theta", self.prior).to(real_designs)
        else:
            theta = pyro.deterministic("theta", self.prior()).to(real_designs)
        with gpytorch.settings.trace_mode(), gpytorch.settings.debug(
            False
        ), torch.device(real_designs.device):
            real_inputs = combine_inputs(real_designs, theta)
            real_inputs = model.transform_inputs(real_inputs)
            with gpytorch.settings.lazily_evaluate_kernels(False):
                sim_pred = model(real_inputs)
            err_mean, err_cov = self._error_model(real_designs, theta)

            if self.noise_prior is not None:
                real_noise = pyro.sample("real_noise", self.noise_prior)
            else:
                real_noise = self.noise

            real_noise = real_noise.view(*real_noise.shape, 1).expand(
                *real_noise.shape[:-1], *real_designs.shape[:-1], model.num_outputs
            )
            if real_outcomes is not None:
                if hasattr(model, "outcome_transform"):
                    real_outcomes, real_noise = model.outcome_transform(
                        real_outcomes, real_noise
                    )
                real_outcomes = real_outcomes.transpose(
                    -1, -2
                )  # for batched multi-output case
            else:
                if hasattr(model, "outcome_transform"):
                    _, real_noise = model.outcome_transform(
                        torch.zeros_like(real_noise), real_noise
                    )
            real_noise = real_noise.transpose(-1, -2)

            if isinstance(self.sim_scale, Distribution):
                with pyro.plate("sim_scale_plate", model.num_outputs, dim=-1):
                    sim_scale = pyro.sample("sim_scale", self.sim_scale)
            else:
                sim_scale = self.sim_scale

            noise_cov = torch.eye(real_noise.shape[-1]).mul(real_noise.unsqueeze(-1))
            err_pred_cov = (
                sim_pred.covariance_matrix.mul(
                    sim_scale.pow(2).view(*sim_scale.shape, 1, 1)
                )
                + err_cov
            )
            lik_cov = err_pred_cov + noise_cov
            lik_mean = (sim_pred.mean + err_mean).mul(sim_scale.unsqueeze(-1))
            lik = MultivariateNormal(lik_mean + err_mean, lik_cov)
            with pyro.plate("obs_plate", size=model.num_outputs):
                real_y = pyro.sample("real_y", lik, obs=real_outcomes)

        return real_y
