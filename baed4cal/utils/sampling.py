from typing import Any

import gpytorch
import pyro
import torch
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_sample

from baed4cal.problems import BAEDProblem


class SimulatorPosteriorModel:
    def __init__(self, problem) -> None:
        self.problem = problem

    def __call__(self, real_xs, real_ys) -> Any:
        theta = pyro.sample("theta", self.problem.prior)
        points = torch.cat(
            [
                real_xs.expand(*theta.shape[:-1], *real_xs.shape),
                theta.unsqueeze(-2).expand(*theta.shape[:-1], *real_xs.shape[:-1], -1),
            ],
            -1,
        )
        with pyro.plate("y_plate", size=len(points)):
            y = pyro.sample(
                "y", self.problem.observations_likelihood(points), obs=real_ys
            )
        return y


def sample_posterior(
    problem: BAEDProblem,
    real_xs: torch.Tensor,
    real_ys: torch.Tensor,
    n_samples: int,
    n_chains: int = 8,
    show_progress_bar: bool = False,
) -> torch.Tensor:
    model = SimulatorPosteriorModel(problem)
    with gpytorch.settings.trace_mode():
        mcmc_kernel = NUTS(model, init_strategy=init_to_sample, jit_compile=True)
        mcmc = MCMC(
            mcmc_kernel,
            num_samples=n_samples,
            num_chains=n_chains,
            disable_progbar=not show_progress_bar,
            mp_context="spawn",
        )
        mcmc.run(real_xs, real_ys)
        mcmc_samples = mcmc.get_samples()
    return mcmc_samples["theta"]
