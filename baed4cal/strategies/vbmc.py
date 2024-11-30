import math
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import pyro
import torch
import yaml
from pyro.distributions import TorchDistribution, Uniform
from pyvbmc import VBMC, VariationalPosterior

from ..problems import BAEDSyntheticProblem, BAEDProblem
from .observations import ObservationsDataset
from ..utils.data import combine_with_parameters
from ..utils.experiments import evaluate_posterior
from ..utils import stats
from ..logging_config import make_file_logger


class VBMCTarget:
    def __init__(self, problem: BAEDProblem, real_data: ObservationsDataset) -> None:
        self.problem = problem
        self.real_data = real_data

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        theta = torch.from_numpy(theta).to(self.real_data.points)

        points = combine_with_parameters(self.real_data.points, theta)
        sim_like = self.problem.observations_likelihood(points)
        log_like = sim_like.log_prob(self.real_data.outcomes).sum(-1)

        log_prior = self.problem.prior.log_prob(theta)
        log_joint = log_like + log_prior

        return log_joint.cpu().numpy()


class VBMCPosterior(TorchDistribution):
    has_rsample = False
    has_enumerate_support = False

    def __init__(self, vbmc_posterior: VariationalPosterior):
        super().__init__(
            batch_shape=torch.Size([]),
            event_shape=torch.Size([vbmc_posterior.D]),
            validate_args=False,
        )
        self.vbmc_posterior = vbmc_posterior

    @property
    def _prototype_tensor(self):
        # For data conversion purposes, capturing default device and dtype
        return torch.zeros([])

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        x = torch.atleast_2d(value).cpu().numpy()
        log_p = self.vbmc_posterior.log_pdf(x)
        log_p = torch.from_numpy(log_p).to(self._prototype_tensor)

        return log_p.reshape(value.shape[:-1])

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        n_samples = math.prod(sample_shape)
        samples, _ = self.vbmc_posterior.sample(n_samples)
        samples = torch.from_numpy(samples).to(self._prototype_tensor)
        return samples.reshape(*sample_shape, *self.event_shape)


class VBMCRunner:
    def __init__(
        self,
        problem: BAEDSyntheticProblem,
        real_data: ObservationsDataset,
        max_eval: int,
        out_dir: str,
        posterior_samples: Optional[torch.Tensor] = None,
        n_assessment_samples: int = 10000,
        seed: Optional[int] = None,
    ) -> None:
        self.problem = problem
        self.real_data = real_data
        if max_eval > 0:
            self.max_eval = max_eval
        else:
            raise ValueError(
                "Maximum number of evaluations must be a positive integer for VBMC."
            )
        self.out_dir = out_dir
        self.n_assessment_samples = n_assessment_samples
        self.seed = seed
        self.posterior_samples = posterior_samples

    def _final_assessment(self, final_q: VBMCPosterior, runtime: float):
        # Evaluate information gain and final RMSE (with final MCMC posterior)
        final_metrics = dict()
        final_metrics["Time"] = runtime

        # Get samples
        sample_shape = torch.Size([self.n_assessment_samples])
        q_samples = final_q.sample(sample_shape)
        prior_samples = self.problem.prior.sample(sample_shape)

        distribution_samples = {
            "q": q_samples,
            "true": self.posterior_samples,
            "prior": prior_samples,
        }
        # Estimate KL divergence between all valid pairs
        for p_label, p in distribution_samples.items():
            for q_label, q in distribution_samples.items():
                if p_label != q_label and None not in (p, q):
                    label = f"KL_{p_label}_to_{q_label}"
                    kl = stats.kl_estimate(p, q)
                    final_metrics[label] = kl

        return final_metrics

    def __call__(self) -> dict:
        if self.seed is not None:
            pyro.set_rng_seed(self.seed)
        logger = make_file_logger(
            os.path.join(self.out_dir, "VBMC.log"),
            logger_name="VBMC",
            include_warnings=True,
        )

        # Run VBMC
        try:
            posterior, runtime = run_vbmc(
                self.problem,
                self.real_data,
                self.max_eval,
                self.out_dir,
                return_runtime=True,
            )
        except Exception:
            logger.exception("Error occurred during VBMC run...")
            return None

        # Evaluate results
        logger.info("Evaluating final results...")
        final_metrics = self._final_assessment(posterior, runtime)

        metrics = evaluate_fit(
            posterior,
            posterior_samples=self.posterior_samples,
            problem=self.problem,
            true_theta=getattr(self.problem, "true_parameters", None),
            n_assessment_samples=self.n_assessment_samples,
        )
        metrics.update(final_metrics)
        for metric_name, metric_value in metrics.items():
            logger.info(f"Metric '{metric_name}': {metric_value}")

        # Save performance results
        df = pd.DataFrame(metrics, index=[0])
        df.to_csv(os.path.join(self.out_dir, "VBMC-metrics.csv"))
        return metrics


def format_results(data: dict):
    formatted_dict = dict()
    for k, v in data.items():
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        formatted_dict[k] = v
    return formatted_dict


def run_vbmc(
    problem: BAEDSyntheticProblem,
    real_data: ObservationsDataset,
    max_eval: Optional[int] = None,
    out_dir: Optional[str] = None,
    n_plausible_std: int = 3,
    stable_count_ratio: float = 0.75,
    return_runtime: bool = False,
) -> TorchDistribution:

    # Prepare target
    target = VBMCTarget(problem, real_data)

    # Prepare search space
    d = problem.calibration_dimension
    prior_mu = problem.prior.mean.view(1, d)
    prior_std = problem.prior.stddev.view(1, d)

    plb = prior_mu - n_plausible_std * prior_std  # Plausible lower bounds
    pub = prior_mu + n_plausible_std * prior_std  # Plausible upper bounds

    theta0 = Uniform(plb, pub).sample().cpu().numpy()

    plb = plb.cpu().numpy()
    pub = pub.cpu().numpy()

    options = {
        "max_fun_evals": max_eval,  # Max number of target function evaluations
        "tol_stable_count": int(
            stable_count_ratio * max_eval
        ),  # Required stable function evals for termination
        "log_file_name": (
            os.path.join(out_dir, "VBMC.log") if out_dir is not None else None
        ),
    }

    # Run VBMC
    vbmc = VBMC(
        target,
        theta0,
        plausible_lower_bounds=plb,
        plausible_upper_bounds=pub,
        options=options,
    )
    start_time = time.time()
    vbmc_posterior, results = vbmc.optimize()
    runtime = time.time() - start_time

    # Convert result
    posterior = VBMCPosterior(vbmc_posterior)

    # Save result
    if out_dir is not None:
        vbmc.save(os.path.join(out_dir, "VBMC"))
        vbmc_posterior.save(os.path.join(out_dir, "VBMC-posterior"))
        results = format_results(results)
        with open(os.path.join(out_dir, "VBMC-result.yaml"), "w") as f:
            yaml.safe_dump(results, f)

    if return_runtime:
        return posterior, runtime
    else:
        return posterior


def evaluate_fit(
    posterior: VBMCPosterior,
    problem: BAEDProblem,
    posterior_samples: Optional[torch.Tensor] = None,
    true_theta: Optional[torch.Tensor] = None,
    n_assessment_samples: int = 10000,
) -> dict:
    prior_samples = problem.prior.sample(torch.Size([n_assessment_samples]))
    if true_theta is None and isinstance(problem, BAEDSyntheticProblem):
        true_theta = problem.true_parameters
    metrics = evaluate_posterior(
        posterior,
        posterior_samples,
        prior_samples=prior_samples,
        true_theta=true_theta,
        n_assessment_samples=n_assessment_samples,
    )
    return metrics
