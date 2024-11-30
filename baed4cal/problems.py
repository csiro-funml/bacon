import warnings
from abc import ABC
import botorch
from pyro.distributions.torch_distribution import TorchDistribution
from scipy.stats.qmc import LatinHypercube
from typing import Optional, Callable, Union

import torch
import torch.distributions as td
import pyro.distributions as pyd
from gpytorch.kernels import RBFKernel
from torch import Tensor

from kohgp.models.kernels import BiFiKernel

from .simulation.gp_simulator import GPSampleSimulator
from .simulation.location import LocationFindingSimulator


class BAEDProblem(ABC):
    """
    Abstract base class for Bayesian adaptive experimental design problems with simulators
    """

    def __init__(
        self,
        simulator: Callable[[torch.Tensor], torch.Tensor],
        prior: pyd.TorchDistribution,
        lower_bounds: Optional[torch.Tensor] = None,
        upper_bounds: Optional[torch.Tensor] = None,
        design_dimension: Optional[int] = None,
    ):
        """Constructor

        Args:
            simulator (Callable[[torch.Tensor],torch.Tensor]): Simulator which takes as inputs an array of points, each containing design and calibration parameters concatenated along the last dimension
            prior (pyd.Distribution): Prior probability distribution for the true/optimal calibration parameters
            lower_bounds (Optional[torch.Tensor], optional): Lower bounds for the design space. Defaults to None.
            upper_bounds (Optional[torch.Tensor], optional): Upper bounds for the design space. Defaults to None.
            dimension (Optional[int], optional): Dimensionality of the design space. Defaults to None.
        """
        self.simulator = simulator
        self.prior = prior
        self._theta_d = self.prior.event_shape[-1]
        assert design_dimension is not None or (
            lower_bounds is not None and upper_bounds is not None
        ), "Dimension or bounds must be provided"
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self._design_dimension = design_dimension
        if design_dimension is None:
            self._design_dimension = self.lower_bounds.numel()

    @property
    def dimension(self) -> int:
        """Dimensionality of the combined design and calibration parameters space

        Returns:
            int: dimension of the space
        """
        return self.design_dimension + self.calibration_dimension

    @property
    def design_dimension(self) -> int:
        """Dimensionality of the design space

        Returns:
            int: dimension of the space
        """
        return self._design_dimension

    @property
    def calibration_dimension(self) -> int:
        """Dimensionality of the calibration parameters space

        Returns:
            int: dimension of the space
        """
        return self.prior.event_shape[-1]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        warnings.warn(
            "Deprecated method. Use either simulate() or observe().",
            category=FutureWarning,
        )
        if x.shape[-1] != self.design_dimension + self.prior.event_shape[-1]:
            raise ValueError("Inputs must agree with the problem dimension")

        return self.simulator(x)

    def observations_likelihood(self, points: torch.Tensor) -> pyd.Distribution:
        """Likelihood model for real data observations

        Args:
            points (torch.Tensor): Simulator inputs (i.e., designs and calibration parameters concatenated along the last dimension)

        Raises:
            NotImplementedError: Not implemented in base class

        Returns:
            pyd.Distribution: Probability distribution for real observations given design and realisations of the calibration parameters
        """
        raise NotImplementedError()

    def simulate(self, points: torch.Tensor) -> torch.Tensor:
        """Run simulations at given input points.

        Args:
            points (torch.Tensor): Simulator inputs composed of design points and calibration parameters concatenated along the last dimension

        Returns:
            torch.Tensor: Simulation outputs
        """
        if points.shape[-1] != self.dimension:
            raise ValueError(
                "Points last dimension does not match concatenated design and calibration parameters dimension"
            )
        return self.simulator(points)

    def observe(self, designs: torch.Tensor) -> torch.Tensor:
        """Observe real data at the provided design points

        Args:
            designs (torch.Tensor): Design points (i.e., not containing simulator's calibration parameters), one per row

        Raises:
            NotImplementedError: Method not implemented in base class

        Returns:
            torch.Tensor: Real data observatins
        """
        raise NotImplementedError()


class BAEDSyntheticProblem(BAEDProblem):
    def __init__(
        self,
        simulator: Callable[[Tensor], Tensor],
        prior: TorchDistribution,
        lower_bounds: Tensor | None = None,
        upper_bounds: Tensor | None = None,
        design_dimension: int | None = None,
        noise_sd: float = 0.1,
    ):
        super().__init__(simulator, prior, lower_bounds, upper_bounds, design_dimension)
        self._noise_sd = noise_sd
        self.true_parameters = self.prior.sample()

    @property
    def true_parameters(self) -> torch.Tensor:
        """True simulator parameters which generated the real data"""
        return self._true_parameters

    @true_parameters.setter
    def true_parameters(self, value: torch.Tensor):
        if value.shape != self.prior.event_shape:
            raise ValueError(
                "Invalid shape for true parameters not matching the prior's event shape"
            )
        self._true_parameters = value

    @property
    def noise_sd(self) -> float:
        return self._noise_sd

    def observations_likelihood(self, points: torch.Tensor) -> pyd.TorchDistribution:
        simulation = self.simulate(points)
        likelihood = pyd.Normal(simulation, self._noise_sd)
        return likelihood

    def observe(self, designs: torch.Tensor) -> torch.Tensor:
        if designs.shape[-1] != self.design_dimension:
            raise ValueError(
                "Provided desgins do not match dimensionality of this problem's design space"
            )
        points = torch.cat(
            [
                designs,
                self.true_parameters.expand(
                    *designs.shape[:-1], *self.true_parameters.shape
                ),
            ],
            -1,
        )
        true_likelihood = self.observations_likelihood(points)
        observations = true_likelihood.sample()
        return observations


class SyntheticGPProblem(BAEDSyntheticProblem):
    def __init__(
        self,
        design_dimension: int = 1,
        theta_d: Optional[int] = None,
        n_inducing: int = 256,
        lengthscale: Union[torch.Tensor, float, None] = None,
        noise_sd: float = 0.5,
    ):
        assert design_dimension >= 0
        if theta_d is None:
            theta_d = design_dimension
        assert theta_d > 0
        assert noise_sd > 0
        design_d = design_dimension
        lower_bounds = torch.zeros(design_d)
        upper_bounds = torch.ones(design_d)
        prior = pyd.MultivariateNormal(torch.zeros(theta_d), torch.eye(theta_d))
        # xs = torch.linspace(0, 1, int(n_inducing ** (1.0 / design_d)))
        # inducing_designs = torch.stack(
        #     torch.meshgrid([xs] * design_d, indexing="xy"), -1
        # ).view(-1, design_d)
        lhs = LatinHypercube(design_dimension + theta_d)
        lhs_points = torch.as_tensor(
            lhs.random(n=n_inducing),
            device=upper_bounds.device,
            dtype=upper_bounds.dtype,
        )
        inducing_designs = lhs_points[:, :design_dimension]
        inducing_thetas = (
            lhs_points[:, design_dimension:] * 6 - 3
        )  # covering +/- 3 std. deviations
        # inducing_designs = (
        #     pyd.Uniform(lower_bounds, upper_bounds).sample((n_inducing,)) * 2
        # )
        # inducing_thetas = prior.sample(inducing_designs.shape[:-1]) * 3
        inducing_points = torch.cat(
            [inducing_designs.view(-1, design_d), inducing_thetas.view(-1, theta_d)], -1
        )
        covar_module = RBFKernel(design_dimension + theta_d)
        if lengthscale is None:
            lengthscale = torch.tensor([0.2] * design_dimension + [1.0] * theta_d)
        covar_module.lengthscale = lengthscale
        simulator = GPSampleSimulator(inducing_points, covar_module=covar_module)
        super().__init__(
            simulator,
            prior,
            lower_bounds,
            upper_bounds,
            design_dimension,
            noise_sd=noise_sd,
        )


class KOHSyntheticGPProblem(SyntheticGPProblem):
    simulator: GPSampleSimulator

    def __init__(
        self,
        design_dimension: int = 1,
        theta_d: int | None = None,
        n_inducing: int = 1024,
        lengthscale: Tensor | float | None = None,
        err_lengthscale: Tensor | float = 0.5,
        err_outputscale: Tensor | float = 0.01,
        noise_sd: float = 0.5,
        inducing_fidelities_p: float = 0.25,
    ):
        super().__init__(design_dimension, theta_d, n_inducing, lengthscale, noise_sd)
        with torch.no_grad():
            kernel = BiFiKernel(
                self.design_dimension,
                self.calibration_dimension,
                use_scale=False,
                use_ard=True,
                nu=2.5,
            )
            err_kernel = kernel.err_kernel
            if hasattr(err_kernel, "outputscale"):
                err_kernel.outputscale = err_outputscale
            if hasattr(err_kernel, "base_kernel"):
                if err_kernel.base_kernel.has_lengthscale:
                    err_kernel.base_kernel.lengthscale = err_lengthscale
            sim_kernel = kernel.sim_kernel
            sim_kernel.outputscale = 1
            sim_kernel.base_kernel.lengthscale = (
                self.simulator.covar_module.lengthscale.detach().clone()
            )
            inducing_fidelities = td.Bernoulli(
                torch.tensor(inducing_fidelities_p)
            ).sample((n_inducing, 1))
            inducing_points = torch.cat(
                [self.simulator.inducing_points.detach().clone(), inducing_fidelities],
                -1,
            )
            simulator = GPSampleSimulator(
                inducing_points=inducing_points,
                covar_module=kernel,
                mean_module=self.simulator.mean_module,
            )
            del self.simulator
            self.simulator = simulator

    def simulate(self, points: Tensor) -> Tensor:
        sim_inputs = torch.cat(
            [points, torch.zeros(*points.shape[:-1], 1).to(points)], -1
        )
        sim_out = self.simulator(sim_inputs)
        return sim_out

    def observe(self, designs: Tensor) -> Tensor:
        f_inputs = torch.cat(
            [
                designs,
                self.true_parameters.expand(
                    *designs.shape[:-1], *self.true_parameters.shape
                ),
                torch.ones(*designs.shape[:-1], 1).to(designs),
            ],
            -1,
        )
        f = self.simulator(f_inputs)
        noise = torch.randn_like(f) * self.noise_sd
        obs = f + noise
        return obs

    def observations_likelihood(self, points: Tensor) -> TorchDistribution:
        """Compute the distribution of the observations given a known simulator
        with the error function marginalised out

        Args:
            points (Tensor): Real design points and calibration parameters

        Returns:
            TorchDistribution: A Gaussian distribution centred at the simulation output
        """
        err_cov = self.simulator.covar_module.err_kernel(points).to_dense()
        sim = self.simulate(points)
        mvn = pyd.MultivariateNormal(
            sim, err_cov + torch.eye(*err_cov.shape[-2:]).to(points) * self.noise_sd**2
        )
        return mvn


class LocationFindingProblem(BAEDSyntheticProblem):
    def __init__(self, spatial_d: int = 2, n_sources: int = 2, noise_sd: float = 0.1):
        assert spatial_d > 0
        prior = pyd.MultivariateNormal(
            torch.zeros(spatial_d * n_sources), torch.eye(spatial_d * n_sources)
        )
        lower_bounds = torch.zeros(spatial_d)
        upper_bounds = torch.ones(spatial_d)
        simulator = LocationFindingSimulator(n_sources=n_sources, spatial_d=spatial_d)
        super().__init__(
            simulator,
            prior,
            lower_bounds,
            upper_bounds,
            design_dimension=spatial_d,
            noise_sd=noise_sd,
        )


class ClassicTestFunctionProblem(BAEDSyntheticProblem):
    def __init__(
        self,
        design_dimension: int = 2,
        test_function: str = "Levy",
        lower_bounds: Tensor | None = None,
        upper_bounds: Tensor | None = None,
        noise_sd: float = 1.0,
        negate: bool = True,
        **test_function_kwargs
    ):
        prior = pyd.MultivariateNormal(
            torch.zeros(design_dimension), torch.eye(design_dimension)
        )
        test_fn_cls = getattr(botorch.test_functions, test_function)
        problem_d = design_dimension * 2
        assert issubclass(test_fn_cls, botorch.test_functions.SyntheticTestFunction)
        test_fn_proto = test_fn_cls(
            dim=problem_d, negate=negate, **test_function_kwargs
        )
        bounds = torch.as_tensor(test_fn_proto._bounds)
        bounds[design_dimension:, 0] = -torch.inf
        bounds[design_dimension:, 1] = torch.inf
        test_fn = test_fn_cls(
            dim=problem_d, bounds=bounds, negate=negate, **test_function_kwargs
        )
        self.test_function = test_fn
        simulator = test_fn.evaluate_true
        lower_bounds = bounds[:design_dimension, 0]
        upper_bounds = bounds[:design_dimension, 1]
        super().__init__(
            simulator, prior, lower_bounds, upper_bounds, design_dimension, noise_sd
        )
