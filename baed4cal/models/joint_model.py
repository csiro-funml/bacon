from abc import abstractmethod
from typing import Callable, Optional, Union
import warnings

import gpytorch
import linear_operator
import torch
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, GPyTorchPosterior
from gpytorch.distributions import Distribution, MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.means import Mean, ZeroMean
from gpytorch.models import GP, ApproximateGP
from gpytorch.variational import UnwhitenedVariationalStrategy
from linear_operator import LinearOperator

from kohgp.models.kernels import BiFiKernel
from baed4cal.models.conditional_density_estimation import (
    BaseConditionalVariationalDistribution,
    ConditionalVariationalDistribution,
)


class ConditionalSparseGPModel(ApproximateGP):
    def __init__(
        self,
        conditional_variational_distribution: ConditionalVariationalDistribution,
        inducing_points: torch.Tensor,
        covar_module: Kernel,
        mean_module: Optional[Mean] = None,
        learn_inducing_locations: bool = False,
    ):
        variational_strategy = UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            conditional_variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        self.conditional_variational_distribution = conditional_variational_distribution
        self.covar_module = covar_module
        if mean_module is None:
            mean_module = ZeroMean()
        self.mean_module = mean_module

    def condition(self, conditioner: torch.Tensor):
        self.conditional_variational_distribution.condition(conditioner)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class JointModel(GP):
    @abstractmethod
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Distribution:
        raise NotImplementedError()


class JointSparseGPModel(JointModel, ConditionalSparseGPModel):
    def __init__(
        self,
        conditional_variational_distribution: ConditionalVariationalDistribution,
        inducing_points: torch.Tensor,
        covar_module: Kernel,
        mean_module: Mean | None = None,
        learn_inducing_locations: bool = False,
    ):
        super(ConditionalSparseGPModel, self).__init__(
            conditional_variational_distribution,
            inducing_points,
            covar_module,
            mean_module,
            learn_inducing_locations=learn_inducing_locations,
        )
        self.conditional_variational_distribution.initialize_variational_distribution(
            super().forward(inducing_points)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> MultivariateNormal:
        self.condition(condition)  # conditions variational inducing points distribution
        self.variational_strategy._clear_cache()  # clears cache of the variational distribution to use the conditioned one
        return super(ConditionalSparseGPModel, self).__call__(x)


class ConditionalOptimalInducingVariableDistribution(
    BaseConditionalVariationalDistribution
):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        covar_module: Kernel,
        training_targets: Optional[torch.Tensor] = None,
        conditional_training_points: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        noise_var: Union[float, torch.Tensor, None] = None,
        init_conditioner: Optional[torch.Tensor] = None,
        batch_shape: torch.Size = torch.Size([]),
        jitter: float = 1e-6,
    ):
        num_inducing_points = inducing_points.shape[-2]
        super().__init__(num_inducing_points, batch_shape)
        self.covar_module = covar_module
        self.inducing_points = inducing_points
        self.conditional_training_points = conditional_training_points
        self.training_targets = training_targets
        self.conditioner = init_conditioner
        if noise_var is not None:
            self.noise_var = (
                torch.as_tensor(noise_var).to(inducing_points)
                if not torch.is_tensor(noise_var)
                else noise_var
            )
        else:
            self.noise_var = None
        self.conditioned_distribution: MultivariateNormal | None = None
        self._inducing_covar: LinearOperator | None = None
        self.jitter = jitter

    def set_training_targets(self, targets: torch.Tensor):
        self.training_targets = targets

    def set_training_points_mapper(
        self, conditional_training_points: Callable[[torch.Tensor], torch.Tensor]
    ):
        self.conditional_training_points = conditional_training_points

    @property
    def inducing_covar(self) -> LinearOperator:
        return self.covar_module(self.inducing_points)

    def initialize_variational_distribution(
        self, prior_dist: MultivariateNormal
    ) -> None:
        self._inducing_covar = prior_dist.covariance_matrix

    def condition(self, conditioner: torch.Tensor):
        assert self.training_targets is not None, "Training target must be set"
        assert (
            self.conditional_training_points is not None
        ), "Training points mapper must be set"
        assert self.inducing_covar is not None, "Must be initialized"
        assert self.noise_var is not None, "Noise variance must be set"
        self.conditioner = conditioner

        kernel = self.covar_module
        inducing_covar = self.inducing_covar
        training_points = self.conditional_training_points(conditioner)

        psi_1 = kernel(training_points, self.inducing_points)
        psi_2 = psi_1.transpose(-1, -2).matmul(psi_1)
        noise_var = self.noise_var

        data_cov = inducing_covar.mul(noise_var) + psi_2
        data_chol = data_cov.cholesky()
        factor = data_chol.solve(inducing_covar.to_dense())

        mean = factor.transpose(-1, -2) @ data_chol.solve(
            psi_1.transpose(-1, -2).matmul(self.training_targets)
        )
        # covariance =  noise_var * factor.transpose(-1, -2) @ factor
        covariance = linear_operator.operators.RootLinearOperator(
            factor.transpose(-1, -2).mul(noise_var**0.5)
        )

        self.conditioned_distribution = MultivariateNormal(
            mean.view(*conditioner.shape[:-1], -1), covariance.add_jitter(self.jitter)
        )

    def forward(self) -> MultivariateNormal:
        assert self.conditioned_distribution is not None, "Condition must be set"
        return self.conditioned_distribution


class JointFullGP(JointModel):
    def __init__(
        self,
        sim_points: torch.Tensor,
        sim_targets: torch.Tensor,
        real_points: torch.Tensor,
        real_targets: torch.Tensor,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        observation_noise_sd: Optional[float] = None,
        simulation_noise_sd: float = 1e-2,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        super().__init__()
        if mean_module is None:
            mean_module = ZeroMean()
        self.real_points = real_points
        self.real_targets = real_targets
        if observation_noise_sd is None:
            warnings.warn(
                "Observation noise level has not been informed and will be assumed "
                "to be the same as the simulation noise level"
            )
            observation_noise_sd = simulation_noise_sd
        self.observation_noise_sd = torch.as_tensor(observation_noise_sd).to(
            real_targets
        )
        self.simulation_noise_sd = torch.as_tensor(simulation_noise_sd).to(sim_targets)
        sim_points = torch.cat(
            [sim_points, torch.zeros(sim_points.shape[:-1] + (1,)).to(sim_points)], -1
        )
        if covar_module is None:
            design_d = real_points.shape[
                -1
            ]  # design parameters dimension, i.e., the number of design parameters
            calibration_d = (
                sim_points.shape[-1] - design_d
            )  # calibration parameters dimension
            covar_module = BiFiKernel(
                design_d, calibration_d, fidelity_coord=-1, use_ard=True
            )
        self.gp_model = SingleTaskGP(
            sim_points,
            sim_targets.unsqueeze(-1),
            train_Yvar=self.simulation_noise_sd.pow(2).expand(*sim_targets.shape, 1),
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    def forward(
        self,
        design: torch.Tensor,
        conditioning_theta: torch.Tensor,
        fidelity: torch.Tensor | int = 0,
    ) -> MultivariateNormal:
        """Evaluates posterior predictive distribution at given simulation inputs

        Args:
            design (torch.Tensor): simulation designs (design + calibration parameters)
            conditioning_theta (torch.Tensor): assumed true calibration parameter
            fidelity (torch.Tensor | int): fidelity level, defaults to 0

        Returns:
            MultivariateNormal: Posterior predictive distribution for latent function
        """
        conditioned_gp = self.condition(conditioning_theta)
        point = self.combine_fidelity(design, fidelity)
        post = conditioned_gp.posterior(point)
        return post.distribution

    @staticmethod
    def combine_fidelity(
        points: torch.Tensor, fidelity: torch.Tensor | int
    ) -> torch.Tensor:
        if not torch.is_tensor(fidelity):
            fidelity = torch.as_tensor(fidelity).to(points)
            fidelity = torch.atleast_1d(fidelity)
        combined = torch.cat(
            [points, fidelity.expand(points.shape[:-1] + fidelity.shape)], -1
        )
        return combined

    @staticmethod
    def combine_inputs(
        design: torch.Tensor, theta: torch.Tensor, fidelity: torch.Tensor | int = 1
    ) -> torch.Tensor:
        """Combine designs with calibration parameters into inputs for the GP model. Batch shapes are broadcasted.

        Args:
            design (torch.Tensor): (..., N, D) tensor with design points
            theta (torch.Tensor): (..., d) tensor with calibration parameters
            fidelity (torch.Tensor | int, optional): Fidelity (1 = real, 0 = simulation). Defaults to 1.

        Returns:
            torch.Tensor: Combined (..., N, D + d + 1) tensor of inputs
        """
        if not torch.is_tensor(fidelity):
            fidelity = torch.as_tensor(fidelity).to(design)

        batch_shape = torch.broadcast_shapes(theta.shape[:-1], design.shape[:-2])
        n_designs = design.shape[-2]

        full_input = torch.cat(
            [
                design.expand(*batch_shape, *design.shape[-2:]),
                theta.unsqueeze(-2).expand(*batch_shape, n_designs, *theta.shape[-1:]),
                fidelity.expand(*batch_shape, n_designs, 1),
            ],
            -1,
        )
        return full_input

    def condition(self, theta: torch.Tensor) -> BatchedMultiOutputGPyTorchModel:
        """Condition GP model on calibration parameters sample

        Args:
            theta (torch.Tensor): (..., d) tensor with calibration parameters

        Returns:
            BatchedMultiOutputGPyTorchModel: GP model conditioned on real data with given parameters and simulations
        """
        # We first complete the inputs and expand the data to the sample batch shape
        full_real_points = self.combine_inputs(self.real_points, theta)
        expanded_targets = self.real_targets.unsqueeze(-1).expand(
            *theta.shape[:-1], *self.real_targets.shape, -1
        )
        noise = self.observation_noise_sd.pow(2).expand_as(expanded_targets)
        # Transform inputs
        full_real_points = self.gp_model.transform_inputs(full_real_points)
        if hasattr(self.gp_model, "outcome_transform"):
            # We only transform the noise here, since `condition_on_observations`
            # already applies the outcome transform to the targets, while assuming
            # that the noise has already been transformed.
            _, noise = self.gp_model.outcome_transform(self.real_targets, noise)
        # Check if cached prediction strategy is available for computations
        if self.gp_model.prediction_strategy is None:
            with torch.no_grad():
                self.gp_model.posterior(full_real_points)  # creates caches
        # Compute a new GP conditioned on the batched data
        conditioned_gp = self.gp_model.condition_on_observations(
            full_real_points,
            expanded_targets,
            noise=noise,
        )
        return conditioned_gp

    def likelihood(
        self,
        design: torch.Tensor,
        theta: torch.Tensor,
        fidelity: torch.Tensor | int = 1,
    ) -> MultivariateNormal:
        """Predictive distribution of query data at combined design inputs
        (i.e., design + calibration parameters) when conditioned on real and
        simulation data

        Args:
            design (torch.Tensor): real inputs (design + assumed calibration parameter)
            theta (torch.Tensor): calibration parameters to condition GP on
            fidelity (torch.Tensor | int): fidelity level, defaults to 1

        Returns:
            MultivariateNormal: Predictive observations distribution (including noise)
        """
        conditioned_gp = self.condition(theta)
        full_input = self.combine_fidelity(design, fidelity)
        tf_input = conditioned_gp.transform_inputs(full_input)
        pred = conditioned_gp(tf_input)
        noises = torch.stack([self.simulation_noise_sd, self.observation_noise_sd]).pow(
            2
        )
        noises = self.transform_noise(noises)
        mvn = self.gp_model.likelihood(
            pred, noise=noises[fidelity].expand_as(pred.mean)
        )
        mvn = self._transform_distribution(mvn)
        return mvn

    def marginal(self, theta: torch.Tensor) -> MultivariateNormal:
        """Compute marginal GP prior probability distribution of entire data (sim + real) for a given sample

        Args:
            theta (torch.Tensor): (..., d) calibration parameters sample

        Returns:
            MultivariateNormal: (possibly batched) data distribution
        """
        full_real_points = self.combine_inputs(self.real_points, theta, fidelity=1)
        full_real_points = self.gp_model.transform_inputs(full_real_points)
        sim_points = self.gp_model.train_inputs[0]
        expanded_sim_points = sim_points.expand(*theta.shape[:-1], *sim_points.shape)
        expanded_inputs = torch.cat([full_real_points, expanded_sim_points], -2)
        noise = self.observation_noise_sd.pow(2).expand_as(self.real_targets)
        noise = self.transform_noise(noise)
        noise = torch.cat(
            [
                noise,
                self.gp_model.likelihood.noise,
            ]
        )
        with gpytorch.settings.prior_mode():
            prior = self.gp_model(expanded_inputs)
            marginal_likelihood = self.gp_model.likelihood(prior, noise=noise)
        marginal_likelihood = self._transform_distribution(marginal_likelihood)
        return marginal_likelihood

    def _transform_distribution(self, mvn: MultivariateNormal) -> MultivariateNormal:
        if hasattr(self.gp_model, "outcome_transform"):
            assert isinstance(self.gp_model.outcome_transform, OutcomeTransform)
            gpt_post = GPyTorchPosterior(mvn)
            utf_post: GPyTorchPosterior = (
                self.gp_model.outcome_transform.untransform_posterior(gpt_post)
            )
            mvn = utf_post.distribution
        return mvn

    def transform_noise(self, noise: torch.Tensor) -> torch.Tensor:
        if hasattr(self.gp_model, "outcome_transform"):
            noise.unsqueeze_(-1)
            assert isinstance(self.gp_model.outcome_transform, OutcomeTransform)
            _, noise = self.gp_model.outcome_transform(torch.zeros_like(noise), noise)
            noise.squeeze_(-1)
        return noise

    def mll(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate log-marginal likelihood of entire data given calibration parameters sample

        Args:
            theta (torch.Tensor): (..., d) calibration parameters sample

        Returns:
            torch.Tensor: log-marginal likelihood value(s)
        """
        marginal_likelihood = self.marginal(theta)
        return marginal_likelihood.log_prob(self.combined_outcomes)

    @property
    def combined_outcomes(self):
        sim_outcomes = self.gp_model.train_targets.unsqueeze(-1)
        if hasattr(self.gp_model, "outcome_transform"):
            sim_outcomes, _ = self.gp_model.outcome_transform.untransform(sim_outcomes)
        return torch.cat([self.real_targets, sim_outcomes.squeeze(-1)])
