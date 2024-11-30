import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Iterable, Optional, Callable, Tuple, Union

from tqdm import trange
from baed4cal.models.mcmc_posterior import MCMCPosterior
from baed4cal.strategies.observations import ObservationsDataset

from botorch import fit_gpytorch_mll
import gpytorch
import pyro
import pyro.distributions as pyd
import pyro.optim
import torch
import torch.distributions as td
import torch.multiprocessing as mp
import torch.nn as nn
from botorch.models.gpytorch import GPyTorchPosterior
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import get_matern_kernel_with_gamma_prior
from botorch.optim.fit import torch_minimize, ExpMAStoppingCriterion
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.utils.warnings import NumericalWarning
from linear_operator.utils.errors import NotPSDError
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide
from pyro.infer.autoguide.initialization import init_to_sample
from torch import Tensor

from ..models.factories import create_conditional_set_nf
from ..models.conditional_density_estimation import ConditionalDistributionModel
from ..models.joint_model import JointFullGP
from ..models.pyro_model import PyroJointGP
from ..problems import BAEDProblem
from ..utils.gpytorch import prior_log_prob


class Strategy(nn.Module, metaclass=ABCMeta):
    def __init__(self, problem: BAEDProblem) -> None:
        super().__init__()
        self.problem = problem

    @abstractmethod
    def generate(self, n: int = 1) -> torch.Tensor:
        raise NotImplementedError()


class RandomStrategy(Strategy):
    def __init__(self, problem: BAEDProblem):
        super().__init__(problem)

    def generate(self, n: int = 1) -> torch.Tensor:
        design_dist = td.Uniform(
            low=self.problem.lower_bounds, high=self.problem.upper_bounds
        )
        x = torch.cat(
            [design_dist.sample((n,)), self.problem.prior.sample((n,)).view(n, -1)], -1
        )
        return x


class AdaptiveStrategy(Strategy):
    """Base class for strategies which make adaptive design choices"""

    def __init__(
        self,
        problem: BAEDProblem,
        real_data: ObservationsDataset,
        sim_data: ObservationsDataset,
    ) -> None:
        super().__init__(problem)
        self.real_data = real_data
        self.sim_data = sim_data
        self.problem = problem

    @abstractmethod
    def generate(self, n: int = 1):
        raise NotImplementedError()

    @property
    @abstractmethod
    def posterior(self) -> pyd.Distribution:
        raise NotImplementedError()

    def update(self, sim_data: ObservationsDataset):
        new_data = ObservationsDataset(
            torch.cat([self.sim_data.points, sim_data.points]),
            torch.cat([self.sim_data.outcomes, sim_data.outcomes]),
        )
        self.sim_data = new_data


class FullGPStrategy(AdaptiveStrategy):
    def __init__(
        self,
        problem: BAEDProblem,
        real_data: ObservationsDataset,
        sim_data: ObservationsDataset,
        observation_noise_sd: Optional[float] = None,
        simulation_noise_sd: float = 1e-2,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        n_retrain_samples: int = 1,
        retrain_gp: bool = True,
    ):
        super().__init__(problem, real_data, sim_data)
        if covar_module is None:
            covar_module = get_matern_kernel_with_gamma_prior(
                ard_num_dims=problem.dimension
            )
        self.covar_module = covar_module
        self.mean_module = mean_module
        if observation_noise_sd is None:
            if hasattr(problem, "noise_sd"):
                observation_noise_sd = problem.noise_sd
            else:
                raise ValueError(
                    "Either observation noise must be specified or the problem needs to inform it."
                )
        if not torch.is_tensor(observation_noise_sd):
            observation_noise_sd = torch.as_tensor(
                observation_noise_sd, dtype=real_data.outcomes.dtype
            )
        self.observation_noise_sd = observation_noise_sd
        if not torch.is_tensor(simulation_noise_sd):
            simulation_noise_sd = torch.as_tensor(
                simulation_noise_sd, dtype=real_data.outcomes.dtype
            )
        self.simulation_noise_sd = simulation_noise_sd
        self.n_retrain_samples = n_retrain_samples
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform
        self.retrain_gp = retrain_gp
        self.model = self._create_model()
        if self.retrain_gp:
            self.retrain()

    def mcmc_posterior(
        self,
        n_samples: int = 500,
        n_chains: int = 1,
        n_warmup: Optional[int] = None,
        show_progress: bool = False,
        jit_compile: bool = False,
    ) -> MCMCPosterior:
        model = self.model
        model.eval().requires_grad_(False)
        gpm = PyroJointGP(self.problem.prior, model)
        model.gp_model.prediction_strategy = (
            None  # clears cache to remove tensors with grad
        )
        with gpytorch.settings.trace_mode(), gpytorch.settings.detach_test_caches():
            mcmc_kernel = NUTS(
                gpm.model,
                init_strategy=init_to_sample,
                jit_compile=jit_compile,
                ignore_jit_warnings=True,
            )
            mcmc = MCMC(
                mcmc_kernel,
                n_samples,
                num_chains=n_chains,
                warmup_steps=n_warmup,
                disable_progbar=not show_progress,
                mp_context="spawn",
            )
            mcmc.run()
            theta_samples = mcmc.get_samples()["theta"]
        # The following clears internal cache that might contain tensors with grad,
        # which therefore could not be later pickled in multiprocessing operations
        model.gp_model.prediction_strategy = None

        posterior = MCMCPosterior(
            samples=theta_samples,
            log_joint=model.mll(theta_samples)
            + self.problem.prior.log_prob(theta_samples),
        )
        return posterior

    def update(self, sim_data: ObservationsDataset | None = None):
        if sim_data is not None:
            super().update(sim_data)
        self.model = self._create_model()
        if self.retrain_gp:
            self.retrain()
        else:
            self.model.eval()

    def _create_model(self):
        model = JointFullGP(
            sim_points=self.sim_data.points,
            sim_targets=self.sim_data.outcomes,
            real_points=self.real_data.points,
            real_targets=self.real_data.outcomes,
            covar_module=self.covar_module.train(),
            mean_module=(
                self.mean_module.train()
                if isinstance(self.mean_module, Mean)
                else self.mean_module
            ),
            observation_noise_sd=self.observation_noise_sd,
            simulation_noise_sd=self.simulation_noise_sd,
            outcome_transform=(
                self.outcome_transform.train()
                if isinstance(self.outcome_transform, OutcomeTransform)
                else self.outcome_transform
            ),
            input_transform=(
                self.input_transform.train()
                if isinstance(self.input_transform, InputTransform)
                else self.input_transform
            ),
        )
        return model

    def retrain(self):
        """Fit GP hyper-parameters to data via MAP estimation"""

        def _evaluate_fit():
            # Zero grad
            for p in parameters.values():
                p.grad = None
            # Sample calibration parameters
            try:
                theta = self.posterior.sample(torch.Size([self.n_retrain_samples]))
            except AttributeError:
                theta = self.problem.prior.sample(torch.Size([self.n_retrain_samples]))
            # Combine with real data design points
            real_inputs = self.model.combine_inputs(self.real_data.points, theta)
            # Transform inputs (defaults to identity if model does not have a transform)
            tf_inputs = model.transform_inputs(real_inputs)
            # Make prediction at the real design points conditioned on simulation data
            pred = model(tf_inputs)
            # Compute log-likelihood
            outcomes = self.real_data.outcomes
            noise = real_noise
            if hasattr(model, "outcome_transform"):  # transforms outcomes if needed
                outcomes, noise = model.outcome_transform(outcomes, noise)
            p_data = model.likelihood(pred, noise=noise)
            log_likes = p_data.log_prob(outcomes)
            # Approximate log marginal likelihood under calibration posterior
            log_like = log_likes.logsumexp(0) - torch.log(
                torch.as_tensor(self.n_retrain_samples)
            )
            # Evaluate log prior probabilites of GP hyper-parameters
            log_prior = prior_log_prob(model, shape=log_like.shape)
            # Return loss and grads
            loss = -(log_like + log_prior)
            loss.backward()
            grads = tuple(p.grad for p in parameters.values())
            return loss, grads

        # Prepare for hyper-parameters optimisation
        model = self.model.gp_model
        model.train()
        model.requires_grad_(True)
        real_noise = self.observation_noise_sd.pow(2).expand_as(self.real_data.outcomes)

        # Fit GP simulation hyper-parameters
        sim_mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(sim_mll)

        # Fix simulation kernel and I/O transform hyper-parameters
        model.covar_module.sim_kernel.requires_grad_(False)
        model.mean_module.requires_grad_(False)
        if hasattr(model, "input_transform"):
            model.input_transform.requires_grad_(False)
        if hasattr(model, "outcome_transform"):
            model.outcome_transform.requires_grad_(False)
        # Set the model to evaluation mode to compute posterior predictions over real data given simulations
        model.eval()

        # Fit GP discrepancy hyper-parameters
        parameters = {
            k: p for k, p in model.named_parameters() if p.requires_grad
        }  # includes only the remaining trainable parameters
        if parameters:  # checks if there's any parameter left to optimise
            result = torch_minimize(
                _evaluate_fit,
                parameters=parameters,
                stopping_criterion=ExpMAStoppingCriterion(),
            )
            if result.message:
                warnings.warn(
                    f"GP retraining result message: {result.message}",
                    category=RuntimeWarning,
                )
        model.requires_grad_(False)


class FullGPRandomStrategy(FullGPStrategy):
    def __init__(
        self,
        problem: BAEDProblem,
        real_data: ObservationsDataset,
        sim_data: ObservationsDataset,
        observation_noise_sd: Optional[float] = None,
        simulation_noise_sd: float = 1e-2,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        retrain_gp: bool = True,
    ):
        super().__init__(
            problem,
            real_data,
            sim_data,
            observation_noise_sd,
            simulation_noise_sd,
            covar_module,
            mean_module,
            outcome_transform,
            input_transform,
            retrain_gp=retrain_gp,
        )

    @property
    def posterior(self) -> pyd.Distribution:
        return self.problem.prior

    def generate(self, n: int = 1) -> Tensor:
        design_dist = td.Uniform(
            low=self.problem.lower_bounds, high=self.problem.upper_bounds
        )
        x = torch.cat(
            [design_dist.sample((n,)), self.problem.prior.sample((n,)).view(n, -1)], -1
        )
        return x

    def update(self, sim_data: ObservationsDataset | None = None):
        super().update(sim_data)


class OptimisationStrategy(FullGPStrategy):
    def __init__(
        self,
        problem: BAEDProblem,
        real_data: ObservationsDataset,
        sim_data: ObservationsDataset,
        observation_noise_sd: float | None = None,
        simulation_noise_sd: float = 0.01,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        n_retrain_samples: int = 1,
        prior_epsilon: float = 0.4,  # TODO: Reduce
        fix_designs: bool = False,
        retrain_gp: bool = True,
        **opt_kwargs,
    ):
        super().__init__(
            problem,
            real_data,
            sim_data,
            observation_noise_sd,
            simulation_noise_sd,
            covar_module=covar_module,
            mean_module=mean_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            n_retrain_samples=n_retrain_samples,
            retrain_gp=retrain_gp,
        )
        self.optimisation_settings = self.default_optimisation_settings
        self.optimisation_settings.update(opt_kwargs)
        self.prior_epsilon = prior_epsilon
        self.fix_designs = fix_designs

    @classmethod
    @property
    def default_optimisation_settings(cls):
        """The default_optimisation_settings property."""
        default_opt_kwargs = {
            "n_restarts": 1,  # Number of optimisation runs, each with an independent initial solution
            "n_steps": 400,  # Number of optimisation steps
            "scheduler": "ExponentialLR",  # PyTorch learning rate scheduler class
            "scheduler_args": {"gamma": 0.99},  # Scheduler (keyword) arguments
            "optim": "Adam",  # PyTorch optimisation algorithm class
            "optim_args": {"lr": 0.1},  # Optimiser (keyword) arguments
            "lr_factor": 1.0,  # Learning-rate factor, multiplying the learning rate of `x`, the first tensor variable
            "stopping_tol": 1.0e-5,  # Relative tolerance for exponential moving average stopping criterion
            "stopping_window": 40,  # Window size for stopping criterion
        }
        return default_opt_kwargs

    @property
    def optimisation_settings(self):
        """Optimisation settings with keys: 'n_restarts', 'n_steps', 'scheduler', 'scheduler_args', 'optim', 'optim_args', 'lr_factor'"""
        return self._optimisation_settings

    @optimisation_settings.setter
    def optimisation_settings(
        self, value
    ):  # TODO: Try to ensure default values are present for non-existing keys in `value`
        self._optimisation_settings = value

    def _load_optimiser(self, optim_params):
        if "optim" in self.optimisation_settings:
            optim_class = getattr(torch.optim, self.optimisation_settings["optim"])
            if "optim_args" in self.optimisation_settings:
                optim_kwargs = self.optimisation_settings["optim_args"]
            else:
                optim_kwargs = dict()
        else:
            optim_class = self.default_optimisation_settings["optim"]
            optim_kwargs = self.default_optimisation_settings["optim_args"]
        opt = optim_class(optim_params, **optim_kwargs)
        return opt

    def _load_scheduler(self, opt):
        # Setup learning rate scheduler
        if "scheduler" in self.optimisation_settings:
            scheduler_class = getattr(
                torch.optim.lr_scheduler, self.optimisation_settings["scheduler"]
            )
            if "scheduler_args" in self.optimisation_settings:
                scheduler_kwargs = self.optimisation_settings["scheduler_args"]
            else:
                scheduler_kwargs = dict()
        else:
            scheduler_class = self.default_optimisation_settings["scheduler"]
            scheduler_kwargs = self.default_optimisation_settings["scheduler_args"]
        sched = scheduler_class(opt, **scheduler_kwargs)
        return sched

    def _run_optimisation(
        self,
        loss_fn: Callable[..., torch.Tensor],
        x: torch.Tensor,
        *other_variables: Union[torch.Tensor, nn.Module],
        return_losses: bool = False,
        **loss_kwargs,
    ):
        """Run optimisation of a given loss function using optimisation pre-specified settings

        Args:
            loss_fn (Callable[..., torch.Tensor]): Loss function.
            x (torch.Tensor): Main optimisation variable.
            other_variables (Union[torch.Tensor, nn.Module]): Other optimisation variables. Unrecognised types will be ignored for optimisation, but still passed on to `loss_fn`.
            return_losses (bool, optional): If enabled, returns losses as a third element on the returned tuple. Defaults to False.
            loss_kwargs (Dict[Any]): Extra kwargs passed on to loss function.

        Returns:
            Tuple[torch.Tensor]: Optimised `x`, final loss, and optionally the array of recorded loss values, if `return_losses=True`.
        """
        warnings.simplefilter("ignore", category=NumericalWarning)

        # Setup optimisation variables
        x.requires_grad_(True)
        optim_params = [
            {
                "params": (x,),
                "lr": self.optimisation_settings["lr_factor"]
                * self.optimisation_settings["optim_args"]["lr"],
            }
        ]
        for i, v in enumerate(other_variables):
            if torch.is_tensor(v) and v.requires_grad:
                optim_params.append({"params": v})
            elif isinstance(v, nn.Module) and v.training:
                optim_params.append({"params": v.parameters()})

        # Setup optimiser
        opt = self._load_optimiser(optim_params)

        # Setup learning rate scheduler
        sched = self._load_scheduler(opt)

        n_steps = self.optimisation_settings["n_steps"]

        # Setup stopping criterion
        stopping_tol = self.optimisation_settings.get("stopping_tol", 1.0e-5)
        stopping_window = self.optimisation_settings.get("stopping_window", 40)
        stopping_criterion = ExpMAStoppingCriterion(
            maxiter=n_steps, rel_tol=stopping_tol, n_window=stopping_window
        )

        loss = None
        losses = []
        xs = []

        # Run optimisation of the design points
        s_iter = trange(n_steps, desc="Optimising designs...")
        for s in s_iter:
            try:
                opt.zero_grad()

                # Compute loss
                loss = loss_fn(x, *other_variables, **loss_kwargs)

                # Tracking for debugging
                losses.append(loss.detach().clone())
                xs.append(x.detach().clone())

                # Update parameters
                loss.backward()
                opt.step()
                sched.step()

                s_iter.set_postfix(loss=loss.cpu().detach().item())
            except NotPSDError:
                continue  # Skip this optimisation step
            finally:
                if not self.fix_designs:
                    # Always ensure design is within bounds
                    with torch.no_grad():
                        x[..., : self.problem.design_dimension].clamp_(
                            min=self.problem.lower_bounds, max=self.problem.upper_bounds
                        )
            if stopping_criterion.evaluate(loss.detach()):
                break

        if torch.is_tensor(loss):
            loss = loss.detach()

        x = x.detach()

        self.losses = torch.stack(losses)
        self.xs = torch.stack(xs)

        # x = xs[losses.argmin()]

        if return_losses:
            return x, loss, self.losses
        else:
            return x, loss

    @abstractmethod
    def _optimisation_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def _sample_mixed_thetas(self, sample_shape: tuple[int]) -> torch.Tensor:
        if isinstance(self.posterior, pyd.Delta):
            selected_samples = self.problem.prior.sample(sample_shape)
        else:
            mask = td.Bernoulli(self.prior_epsilon).sample((sample_shape[-1:])).bool()
            prior_samples = self.problem.prior.sample(sample_shape)
            posterior_samples = self.posterior.sample(sample_shape)
            # all_samples = torch.stack([prior_samples, posterior_samples], -1)
            indices = torch.arange(sample_shape[-1], dtype=torch.long)
            selected_samples = torch.cat(
                [
                    torch.index_select(prior_samples, dim=-2, index=indices[mask]),
                    torch.index_select(posterior_samples, dim=-2, index=indices[~mask]),
                ],
                -2,
            )
        return selected_samples

    def _calibration_only_loss(
        self, parameters: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        x = torch.cat([self.real_data.points[: parameters.shape[-2]], parameters], -1)
        return self._optimisation_loss(x, *args, **kwargs)

    def _generate(
        self,
        n_batch: int = 1,
        extra_vars: Optional[Iterable[Tuple[Union[nn.Module, torch.Tensor]]]] = None,
        **loss_kwargs,
    ):
        n_restarts = self.optimisation_settings.get("n_restarts", 1)

        # Sample initial candidates for design points
        design_dist = td.Uniform(
            low=self.problem.lower_bounds, high=self.problem.upper_bounds
        )
        initial_thetas = self._sample_mixed_thetas((n_restarts, n_batch))

        if self.fix_designs:
            n_real = self.real_data.points.shape[-2]
            assert n_batch == n_real
            initial_x = initial_thetas
            loss_fn = self._calibration_only_loss
        else:
            initial_designs = design_dist.sample((n_restarts, n_batch))
            initial_x = torch.cat([initial_designs, initial_thetas], -1)
            loss_fn = self._optimisation_loss

        process_fn = partial(self._run_optimisation, loss_fn, **loss_kwargs)

        # Prepare list of optimisation variables
        if extra_vars is None:
            extra_vars = [tuple()] * n_restarts
        candidates = []
        for x, extra in zip(initial_x, extra_vars, strict=True):
            entry = [x]
            entry.extend(extra)
            candidates.append(entry)

        # Run optimisation loops
        self.model.eval()
        self.model.requires_grad_(False)
        if n_restarts > 1:
            with mp.Pool() as pool:
                res = pool.starmap(process_fn, candidates)
        else:
            res = [process_fn(*next(iter(candidates)))]

        # Select best result
        best_loss = torch.tensor(torch.inf)
        best_x = None
        best_idx = None
        i = 0
        for point, loss in res:
            if None in (point, loss):
                warnings.warn(
                    f"Discarding unsuccessful optimisation attempt ({i+1}/{n_restarts})...",
                    category=RuntimeWarning,
                )
            elif loss < best_loss:
                best_x = point
                best_loss = loss
                best_idx = i
            i += 1
        if best_x is None:
            raise RuntimeError("Unable to optimise design point")

        if self.fix_designs:
            best_x = torch.cat([self.real_data.points, best_x], -1)

        if len(extra_vars[best_idx]) > 0:
            return best_x.detach(), best_idx
        else:
            return best_x.detach()


class FullGPVariationalStrategy(OptimisationStrategy):
    def __init__(
        self,
        problem: BAEDProblem,
        real_data: ObservationsDataset,
        sim_data: ObservationsDataset,
        observation_noise_sd: Optional[float] = None,
        simulation_noise_sd: float = 1e-2,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        conditional_model_factory: Callable[
            [int, int], ConditionalDistributionModel
        ] = None,
        condition_on_all_data: bool = True,
        condition_on_inputs: bool = True,
        use_mcmc_samples: bool = False,
        use_reg: bool = True,
        split_training: bool = True,
        n_samples: int = 32,
        n_init_samples: int = 2000,
        n_init_chains: int = 8,
        n_retrain_samples: int = 1,
        retrain_gp: bool = True,
        **opt_kwargs,
    ):
        super().__init__(
            problem,
            real_data,
            sim_data,
            observation_noise_sd,
            simulation_noise_sd,
            covar_module=covar_module,
            mean_module=mean_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            n_retrain_samples=n_retrain_samples,
            retrain_gp=retrain_gp,
            **opt_kwargs,
        )

        self.n_init_samples = n_init_samples
        self.n_init_chains = n_init_chains
        self.n_samples = n_samples
        self.last_batch: ObservationsDataset | None = None
        self.q_theta: td.Distribution = self.mcmc_posterior(
            n_samples=self.n_init_samples, n_chains=self.n_init_chains
        )
        if conditional_model_factory is None:
            conditional_model_factory = create_conditional_set_nf
        self.conditional_model_factory = conditional_model_factory
        self.conditional_model: ConditionalDistributionModel | None = None
        self._condition_on_all_data = condition_on_all_data
        self._condition_on_inputs = condition_on_inputs
        self.use_mcmc_samples = use_mcmc_samples
        self.use_reg = use_reg
        self.split_training = split_training

    @property
    def posterior(self) -> pyd.Distribution:
        return self.q_theta

    def update(self, sim_data: ObservationsDataset | None = None):
        super().update(sim_data)

        if sim_data is not None:
            self.last_batch = ObservationsDataset(sim_data.points, sim_data.outcomes)
            if self.use_mcmc_samples or (self.conditional_model is None):
                self.q_theta = self.mcmc_posterior(
                    n_samples=self.n_init_samples, n_chains=self.n_init_chains
                )
            else:
                if self._condition_on_all_data:
                    self.q_theta = self.conditional_model(
                        self.sim_data.outcomes,
                        points=(
                            self.sim_data.points if self._condition_on_inputs else None
                        ),
                    )
                else:
                    self.q_theta = self.conditional_model(
                        sim_data.outcomes,
                        points=sim_data.points if self._condition_on_inputs else None,
                    )

    def _pretrain_conditional(
        self, theta_variational: ConditionalDistributionModel, n_batch: int
    ):
        if not self.split_training:
            return theta_variational

        opt = self._load_optimiser(theta_variational.parameters())
        sched = self._load_scheduler(opt)
        n_steps = self.optimisation_settings["n_steps"]

        # Setup stopping criterion
        stopping_tol = self.optimisation_settings.get("stopping_tol", 1.0e-5)
        stopping_window = self.optimisation_settings.get("stopping_window", 40)
        stopping_criterion = ExpMAStoppingCriterion(
            maxiter=n_steps, rel_tol=stopping_tol, n_window=stopping_window
        )

        design_dist = td.Uniform(
            low=self.problem.lower_bounds, high=self.problem.upper_bounds
        )

        n_pretrain = self.n_samples

        losses = []

        t_iter = trange(n_steps, desc="Pre-training flow...")
        for t in t_iter:
            opt.zero_grad()
            thetas = self._sample_mixed_thetas((n_batch,))
            pretrain_x = torch.cat(
                [
                    design_dist.sample((n_batch,)),
                    thetas,
                ],
                -1,
            )
            pretrain_thetas = self.posterior.sample((n_pretrain,))
            q_y_theta = self.model.likelihood(pretrain_x, pretrain_thetas, fidelity=0)
            y_sim = q_y_theta.sample()
            if self._condition_on_all_data:
                # Combine predictions with past data
                conditioning_points = torch.cat([self.sim_data.points, pretrain_x], -2)
                conditioning_outcomes = torch.cat(
                    [
                        self.sim_data.outcomes.unsqueeze(-2).expand(
                            y_sim.shape[:-1] + self.sim_data.outcomes.shape
                        ),
                        y_sim,
                    ],
                    -1,
                )
            else:
                conditioning_points = pretrain_x
                conditioning_outcomes = y_sim

            if self.use_reg:
                # Regularisation term: ELBO on GP marginal likelihood
                q_prior = theta_variational(
                    y_sim, points=pretrain_x if self._condition_on_inputs else None
                )  # We compute q(theta) by marginalising y out of q(theta|y)
                q_prior_samples = (
                    q_prior.rsample()
                )  # samples theta from the marginal q(theta)
                prior_lps = self.problem.prior.log_prob(q_prior_samples)
                kl_prior = (q_prior.log_prob(q_prior_samples) - prior_lps).mean(
                    -1
                )  # KL(q||p)
                mll = self.model.mll(q_prior_samples).mean(-1)
                reg = mll - kl_prior
            else:
                reg = 0

            # Approximate EIG
            q_dist = theta_variational(
                conditioning_outcomes,
                points=conditioning_points if self._condition_on_inputs else None,
            )
            eig_approx = q_dist.log_prob(pretrain_thetas).mean(-1)
            loss = -eig_approx - reg
            loss.backward()

            opt.step()
            sched.step()

            losses.append(loss.detach().clone())
            t_iter.set_postfix(loss=loss.detach().cpu().item())

            if stopping_criterion.evaluate(loss.detach()):
                break

        losses = torch.stack(losses)
        self.q_losses = losses

        return theta_variational.eval().requires_grad_(False)

    def _optimisation_loss(
        self, x: torch.Tensor, theta_variational: ConditionalDistributionModel
    ) -> torch.Tensor:
        sample_shape = torch.Size((self.n_samples,))
        theta_samples = self.posterior.sample(sample_shape)

        # Sample prediction
        q_y_theta = self.model.likelihood(
            torch.atleast_2d(x), theta_samples, fidelity=0
        )
        y_sim = q_y_theta.rsample()

        if self._condition_on_all_data:
            # Combine predictions with past data
            conditioning_points = torch.cat(
                [self.sim_data.points, torch.atleast_2d(x)], -2
            )

            conditioning_outcomes = torch.cat(
                [
                    self.sim_data.outcomes.unsqueeze(-2).expand(
                        y_sim.shape[:-1] + self.sim_data.outcomes.shape
                    ),
                    y_sim,
                ],
                -1,
            )
        else:
            conditioning_points = torch.atleast_2d(x)
            conditioning_outcomes = y_sim

        # Approximate EIG
        q_dist = theta_variational(
            conditioning_outcomes,
            points=conditioning_points if self._condition_on_inputs else None,
        )
        eig_approx = q_dist.log_prob(theta_samples).mean(-1)

        # Compute loss
        loss = -eig_approx
        return loss

    def generate(self, n_batch: int = 1):
        n_restarts = self.optimisation_settings.get("n_restarts")
        if self._condition_on_all_data:
            n_obs = n_batch + len(self.sim_data.outcomes)
        else:
            n_obs = n_batch
        if self._condition_on_inputs:
            obs_d = self.problem.dimension + 1
        else:
            obs_d = 1
        with torch.device(self.real_data.outcomes.device):
            candidates_theta_variational = [
                (
                    self._pretrain_conditional(
                        self.conditional_model_factory(
                            theta_d=self.problem.calibration_dimension,
                            obs_d=obs_d,
                            n_obs=n_obs,
                        ),
                        n_batch,
                    ),
                )
                for _ in range(n_restarts)
            ]

        best_x, best_idx = self._generate(
            n_batch=n_batch, extra_vars=candidates_theta_variational
        )
        self.conditional_model = (
            candidates_theta_variational[best_idx][0].eval().requires_grad_(False)
        )

        return best_x


class FullGPIMSPEStrategy(OptimisationStrategy):
    def __init__(
        self,
        problem: BAEDProblem,
        real_data: ObservationsDataset,
        sim_data: ObservationsDataset,
        n_prediction_designs: int = 256,
        observation_noise_sd: Optional[float] = None,
        simulation_noise_sd: float = 1e-2,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        n_retrain_samples: int = 1,
        resample_integration: bool = False,
        retrain_gp: bool = True,
        **opt_kwargs,
    ):
        super().__init__(
            problem,
            real_data,
            sim_data,
            observation_noise_sd,
            simulation_noise_sd,
            covar_module,
            mean_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            n_retrain_samples=n_retrain_samples,
            retrain_gp=retrain_gp,
            **opt_kwargs,
        )
        self._theta_estimate: torch.Tensor | None = None
        self._resample_integration = resample_integration
        self.n_prediction_designs = n_prediction_designs
        self._sample_integration_points()

    def _sample_integration_points(self):
        with torch.no_grad():
            sample_shape = torch.Size([self.n_prediction_designs])
            self.prediction_designs = td.Uniform(
                self.problem.lower_bounds, self.problem.upper_bounds
            ).sample(sample_shape)

    @property
    def posterior(self) -> pyd.Distribution:
        theta = self._theta_estimate
        if theta is None:
            warnings.warn("MAP estimate unavailable. Using prior mean, instead...")
            theta = self.problem.prior.mean
        delta = pyd.Delta(theta).to_event()
        return delta

    def retrain(self):
        def _evaluate_fit():
            theta = self._theta_estimate
            mll = self.model.mll(theta)
            theta_logp = self.problem.prior.log_prob(theta)
            loss = -mll - theta_logp
            loss.backward()
            grads = tuple(p.grad for p in self.model.parameters())
            return loss, grads

        self.model.eval()
        self.model.requires_grad_(True)
        self._theta_estimate = (
            self.problem.prior.sample().detach().clone().requires_grad_(True)
        )

        parameters = dict(self.model.named_parameters(), theta=self._theta_estimate)

        torch_minimize(
            _evaluate_fit,
            parameters=parameters,
            stopping_criterion=ExpMAStoppingCriterion(),
        )
        self.model.requires_grad_(False)
        self._theta_estimate.requires_grad_(False)

    def _optimise_theta(self):
        model = self.model
        model.eval().requires_grad_(False)
        gpm = PyroJointGP(self.problem.prior, model)

        opt_class = getattr(torch.optim, self.optimisation_settings["optim"])
        optim_args = self.optimisation_settings.get("optim_args")
        scheduler_class = getattr(pyro.optim, self.optimisation_settings["scheduler"])
        scheduler_args = self.optimisation_settings["scheduler_args"]
        scheduler = scheduler_class(
            {"optimizer": opt_class, "optim_args": optim_args} | scheduler_args
        )
        guide = autoguide.AutoDelta(gpm.model, init_loc_fn=autoguide.init_to_sample)
        svi = SVI(gpm.model, guide, optim=scheduler, loss=Trace_ELBO())

        n_steps = self.optimisation_settings["n_steps"]
        for s in range(n_steps):
            loss = svi.step()

        theta = guide.median(
            self.real_data.points,
            self.sim_data.points,
            self.real_data.outcomes,
            self.sim_data.outcomes,
        )["theta"]
        return theta.detach(), loss

    def _optimisation_loss(self, x: torch.Tensor, model: Model) -> torch.Tensor:
        map_theta = self._theta_estimate
        x = torch.cat(
            [x, torch.zeros(*x.shape[:-1], 1).to(x)], -1
        )  # concatenate fidelity parameter (0 = sim)
        y = model.posterior(x).sample()
        gp_y = y.reshape(*x.shape[:-1], model.num_outputs)
        noise = self.simulation_noise_sd.pow(2).expand_as(gp_y)
        noise = self.model.transform_noise(noise)
        conditioned_on_x = model.condition_on_observations(
            model.transform_inputs(x), gp_y, noise=noise
        )
        if self._resample_integration:
            self._sample_integration_points()
        query_points = torch.cat(
            [
                self.prediction_designs,
                map_theta.expand(*self.prediction_designs.shape[:-1], -1),
                torch.ones(*self.prediction_designs.shape[:-1], 1).to(
                    self.prediction_designs
                ),  # query predictions on real-data fidelity (1 = real)
            ],
            -1,
        )
        pred = conditioned_on_x.posterior(query_points)
        loss = pred.variance.mean(0)
        return loss

    def generate(self, n_batch: int = 1):
        n_restarts = self.optimisation_settings.get("n_restarts")

        # MAP estimate
        best_loss = torch.tensor(torch.inf)
        best_theta = None
        if n_restarts > 1:
            with mp.Pool() as pool:
                res = pool.starmap(self._optimise_theta, [()] * n_restarts)
        else:
            res = [self._optimise_theta()]

        # Select best estimate
        for theta, loss in res:
            if loss < best_loss:
                best_loss = loss
                best_theta = theta
        self._theta_estimate = best_theta

        # Optimise designs
        with torch.no_grad():
            conditioned_gp = self.model.condition(self._theta_estimate)
        conditioned_gp.eval()
        best_x = self._generate(n_batch=n_batch, model=conditioned_gp)

        return best_x


class FullGPMCMCStrategy(OptimisationStrategy):
    def __init__(
        self,
        problem: BAEDProblem,
        real_data: ObservationsDataset,
        sim_data: ObservationsDataset,
        observation_noise_sd: Optional[float] = None,
        simulation_noise_sd: float = 1e-2,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        n_samples: int = 32,
        n_chains: int = 8,
        n_warmup: Optional[int] = 200,
        n_retrain_samples: int = 1,
        **opt_kwargs,
    ):
        super().__init__(
            problem,
            real_data,
            sim_data,
            observation_noise_sd,
            simulation_noise_sd,
            covar_module=covar_module,
            mean_module=mean_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            n_retrain_samples=n_retrain_samples,
            **opt_kwargs,
        )

        self.n_samples = n_samples
        self.n_chains = n_chains
        self.n_warmup = n_warmup
        self.last_batch: ObservationsDataset | None = None
        self.q_theta: td.Distribution = self.mcmc_posterior(
            n_samples=self.n_samples, n_chains=self.n_chains, n_warmup=self.n_warmup
        )

    @property
    def posterior(self) -> pyd.Distribution:
        return self.q_theta

    def update(self, sim_data: ObservationsDataset):
        super().update(sim_data)

        if sim_data is not None:
            self.last_batch = ObservationsDataset(sim_data.points, sim_data.outcomes)
            self.q_theta = self.mcmc_posterior(
                n_samples=self.n_samples, n_chains=self.n_chains, n_warmup=self.n_warmup
            )

    def compute_eig(
        self, x: torch.Tensor, n_samples: int = 32, n_marg_samples: int = 32
    ) -> torch.Tensor:
        # Nested Monte Carlo estimator for EIG
        theta_samples = self.posterior.sample(torch.Size((n_samples,)))
        marg_samples = self.posterior.sample(torch.Size([n_samples, n_marg_samples]))

        ## Sample prediction
        q_f_theta = self.model(torch.atleast_2d(x), theta_samples)
        f_sim = q_f_theta.rsample()

        ## Compute NMC estimator for EIG
        cond_lp = q_f_theta.log_prob(f_sim)
        marg_p = self.model(torch.atleast_2d(x), marg_samples)
        marg_lp = marg_p.log_prob(f_sim.unsqueeze(-2)).logsumexp(-1) - torch.log(
            torch.tensor(n_marg_samples)
        )
        eig_terms = cond_lp - marg_lp
        eig = eig_terms.mean()

        return eig

    def _optimisation_loss(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = n_marg_samples = self.n_samples
        eig = self.compute_eig(x, n_samples, n_marg_samples)
        return -eig

    def generate(self, n_batch: int = 1):
        return self._generate(n_batch=n_batch)


class FullGPEntropyStrategy(FullGPMCMCStrategy):
    """Baseline strategy which selects points where simulation GP's predictive uncertainty is high"""

    def __init__(
        self,
        problem: BAEDProblem,
        real_data: ObservationsDataset,
        sim_data: ObservationsDataset,
        observation_noise_sd: float | None = None,
        simulation_noise_sd: float = 0.01,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        n_samples: int = 1000,
        n_chains: int = 8,
        n_warmup: Optional[int] = None,
        n_retrain_samples: int = 1,
        retrain_gp: bool = True,
        **opt_kwargs,
    ):
        super().__init__(
            problem,
            real_data,
            sim_data,
            observation_noise_sd,
            simulation_noise_sd,
            covar_module,
            mean_module,
            outcome_transform,
            input_transform,
            n_samples=n_samples,
            n_chains=n_chains,
            n_warmup=n_warmup,
            n_retrain_samples=n_retrain_samples,
            retrain_gp=retrain_gp,
            **opt_kwargs,
        )

    def _optimisation_loss(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.atleast_2d(x)
        pred_inputs = torch.cat([x, torch.zeros(*x.shape[:-1], 1).to(x)], -1)
        pred: GPyTorchPosterior = self.model.gp_model.posterior(pred_inputs)
        return -pred.distribution.entropy()
