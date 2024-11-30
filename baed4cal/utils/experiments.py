import time
from typing import Optional

import gpytorch
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from pyro.distributions import Delta, Empirical, Distribution
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_sample

from baed4cal.models.joint_model import JointFullGP
from baed4cal.models.mcmc_posterior import MCMCPosterior
from baed4cal.models.pyro_model import PyroJointGP
from baed4cal.strategies.eig_strategy import (
    FullGPStrategy,
)
from baed4cal.strategies.observations import ObservationsDataset
from baed4cal.utils import stats
from baed4cal.utils.data import combine_with_parameters


def evaluate_rmse(
    strategy: FullGPStrategy,
    test_data: ObservationsDataset,
    n_pred_samples: int,
    posterior: Optional[Distribution] = None,
):
    if posterior is not None:
        pred_theta_samples = posterior.sample(torch.Size((n_pred_samples,)))
    else:
        pred_theta_samples = strategy.posterior.sample(torch.Size((n_pred_samples,)))
    pred_query = combine_with_parameters(test_data.points, pred_theta_samples)
    pred_like = strategy.model.likelihood(pred_query, pred_theta_samples)
    rmse = (pred_like.mean.mean(0) - test_data.outcomes).pow(2).mean().sqrt()
    return rmse.cpu().item()


def evaluate_posterior(
    candidate_posterior: Distribution,
    posterior_samples: Optional[torch.Tensor] = None,
    prior_samples: Optional[torch.Tensor] = None,
    true_theta: Optional[torch.Tensor] = None,
    n_assessment_samples: int = 10000,
) -> dict:
    metrics = dict()

    with torch.no_grad():
        q_theta_samples = candidate_posterior.sample(
            torch.Size((n_assessment_samples,))
        )

        entropy = stats.differential_entropy(
            q_theta_samples
        )  # TODO: Adjust it for Empirical distributions
        metrics["entropy"] = entropy

        if prior_samples is not None:
            metrics["KL_prior"] = stats.kl_estimate(q_theta_samples, prior_samples)

        if true_theta is not None:
            log_prob = candidate_posterior.log_prob(true_theta)
            metrics["log_prob"] = log_prob.cpu().item()

            metrics["cal_err"] = (
                (q_theta_samples.mean(0).to(true_theta) - true_theta)
                .norm()
                .cpu()
                .item()
            )

            if isinstance(candidate_posterior, Delta):
                map_estimate = candidate_posterior.mode
            elif isinstance(candidate_posterior, MCMCPosterior):
                map_estimate = candidate_posterior.enumerate_support()[
                    candidate_posterior.log_joint.argmax()
                ]
            else:
                map_estimate = q_theta_samples[
                    candidate_posterior.log_prob(q_theta_samples).argmax()
                ]
            if map_estimate is not None:
                metrics["map_err"] = (
                    (map_estimate.to(true_theta) - true_theta).norm().cpu().item()
                )

        if posterior_samples is not None:
            mmd = stats.mmd(posterior_samples, q_theta_samples)
            metrics["MMD"] = mmd

            kl = stats.kl_estimate(posterior_samples, q_theta_samples)
            metrics["KL"] = kl

            kl_rev = stats.kl_estimate(q_theta_samples, posterior_samples)
            metrics["KL_rev"] = kl_rev

    return metrics


def evaluate_strategy(
    strategy: FullGPStrategy,
    test_data: Optional[ObservationsDataset] = None,
    posterior_samples: Optional[torch.Tensor] = None,
    prior_samples: Optional[torch.Tensor] = None,
    true_theta: Optional[torch.Tensor] = None,
    n_assessment_samples: int = 10000,
    n_pred_samples: int = 32,
    start_time=None,
) -> dict:
    metrics = dict()
    if start_time is not None:
        metrics["Time"] = time.time() - start_time

    posterior_metrics = evaluate_posterior(
        strategy.posterior,
        posterior_samples,
        prior_samples,
        true_theta,
        n_assessment_samples,
    )
    metrics.update(posterior_metrics)

    with torch.no_grad():
        q_theta_t = strategy.posterior

        if test_data is not None:
            pred_theta_samples = q_theta_t.sample(torch.Size((n_pred_samples,)))
            pred_query = combine_with_parameters(test_data.points, pred_theta_samples)
            pred_like = strategy.model.likelihood(pred_query, pred_theta_samples)
            rmse = (pred_like.mean.mean(0) - test_data.outcomes).pow(2).mean().sqrt()
            metrics["RMSE"] = rmse.cpu().item()
            if true_theta is not None:
                pred_query_true = combine_with_parameters(test_data.points, true_theta)
                pred_like_true = strategy.model.likelihood(pred_query_true, true_theta)
                rmse_true = (
                    (pred_like_true.mean - test_data.outcomes).pow(2).mean().sqrt()
                )
                metrics["RMSE_true"] = rmse_true.cpu().item()

    return metrics


def get_posterior_samples(
    posterior: Distribution, n_samples: int | None = None
) -> torch.Tensor:
    if isinstance(posterior, Delta):
        samples = posterior.mode.view(1, *posterior.event_shape)
    elif isinstance(posterior, Empirical):
        samples = posterior.enumerate_support()
    else:
        assert n_samples is not None
        samples = posterior.sample(torch.Size([n_samples]))
    return samples


def evaluate_ig(
    initial_posterior: Distribution, final_posterior: Distribution, n_samples: int
) -> torch.Tensor:
    initial_samples = get_posterior_samples(initial_posterior, n_samples)
    final_samples = get_posterior_samples(final_posterior, n_samples)
    ig = stats.kl_estimate(final_samples, initial_samples)
    return ig


def evaluate_strategy_ig(
    strategy: FullGPStrategy,
    initial_posterior: Distribution,
    n_assessment_samples,
    n_chains: int = 1,
):
    final_posterior = strategy.mcmc_posterior(
        n_assessment_samples // n_chains, n_chains=n_chains
    )
    return evaluate_ig(
        initial_posterior, final_posterior, n_samples=n_assessment_samples
    )


def sample_posterior(
    prior: Distribution,
    model: JointFullGP,
    gp_inputs: torch.Tensor,
    gp_targets: torch.Tensor,
    gp_noise: torch.Tensor,
    n_samples: int,
    n_chains: int,
    show_progress: bool = False,
    jit_compile: bool = False,
) -> torch.Tensor:
    # Save original data
    (original_inputs,) = model.gp_model.train_inputs
    original_targets = model.gp_model.train_targets
    assert isinstance(model.gp_model.likelihood, FixedNoiseGaussianLikelihood)
    original_noise = model.gp_model.likelihood.noise
    # Set data
    model.gp_model.set_train_data(gp_inputs, gp_targets, strict=False)
    model.gp_model.likelihood.noise = gp_noise
    # Run inference
    pyro_gp = PyroJointGP(prior, model)
    with gpytorch.settings.trace_mode(), gpytorch.settings.detach_test_caches():
        mcmc_kernel = NUTS(
            pyro_gp.model,
            init_strategy=init_to_sample,
            jit_compile=jit_compile,
            ignore_jit_warnings=True,
        )
        mcmc = MCMC(
            mcmc_kernel,
            n_samples // n_chains,
            num_chains=n_chains,
            disable_progbar=not show_progress,
            mp_context="spawn",
        )
        mcmc.run()
        samples = mcmc.get_samples()["theta"]
    # Restore original data
    model.gp_model.set_train_data(original_inputs, original_targets, strict=False)
    model.gp_model.likelihood.noise = original_noise
    # Return posterior samples
    return samples


def evaluate_kls(
    strategies: dict[str, FullGPStrategy],
    n_init: int,
    posterior_samples: Optional[torch.Tensor] = None,
    ref_strategy_name: str = "FullGPRandomStrategy",
    n_samples: int = 2000,
    n_chains: int = 4,
    show_progress: bool = False,
    jit_compile: bool = False,
) -> dict:
    """Estimate KL divergences and entropies for each strategy using hyper-parameters from reference GP model

    Args:
        strategies (dict[str, FullGPStrategy]): Named strategies
        posterior_samples (torch.Tensor): Samples from the true (i.e., known simulator or some other target) posterior
        n_init (int): Number of initial data points to sample initial posterior with
        ref_strategy_name (str, optional): Class name of the reference strategy. Defaults to "FullGPRandomStrategy".
        n_samples (int, optional): Number of MCMC samples. Defaults to 2000.
        n_chains (int, optional): Number of MCMC chains. Defaults to 4.
        show_progress (bool, optional): Enable to show MCMC progress bars. Defaults to False.
        jit_compile (bool, optional): Enable JIT compilation. Defaults to False.

    Returns:
        dict: Dictionary of pairwise-KLs and entropy estimates keyed by strategy name
    """
    # Sample initial posterior based on reference strategy's final GP
    ref_strategy = strategies[ref_strategy_name]
    ref_model = ref_strategy.model
    ref_model.eval().requires_grad_(False)

    initial_samples = sample_posterior(
        ref_strategy.problem.prior,
        ref_model,
        ref_model.gp_model.train_inputs[0][:n_init],
        ref_model.gp_model.train_targets[:n_init],
        ref_model.gp_model.likelihood.noise[:n_init],
        n_samples=n_samples,
        n_chains=n_chains,
        show_progress=show_progress,
        jit_compile=jit_compile,
    )
    estimates = dict()
    final_samples = dict()
    for strategy_name, strategy in strategies.items():
        # Sample posterior based on final data using reference GP
        if isinstance(strategy, FullGPStrategy):
            sim_points = strategy.sim_data.points
            sim_points = torch.cat(
                [sim_points, torch.zeros(sim_points.shape[:-1] + (1,)).to(sim_points)],
                -1,
            )
            final_inputs = ref_model.gp_model.transform_inputs(sim_points)
            if hasattr(ref_model.gp_model, "outcome_transform"):
                final_targets, final_noise = ref_model.gp_model.outcome_transform(
                    strategy.sim_data.outcomes.unsqueeze(-1), strategy.simulation_noise_sd.pow(2).expand_as(strategy.sim_data.outcomes).unsqueeze(-1)
                )
                final_targets = final_targets.squeeze(-1)
                final_noise.squeeze_(-1)
            else:
                final_targets = strategy.sim_data.outcomes
                final_noise = strategy.simulation_noise_sd.pow(2).expand_as(strategy.sim_data.outcomes)
            final_samples[strategy_name] = sample_posterior(
                ref_strategy.problem.prior,
                ref_model,
                gp_inputs=final_inputs,
                gp_targets=final_targets,
                gp_noise=final_noise,
                n_samples=n_samples,
                n_chains=n_chains,
                show_progress=show_progress,
                jit_compile=jit_compile,
            )
            final_metrics = dict()
            distributions = {
                "initial": initial_samples,
                "final": final_samples[strategy_name],
                "true": posterior_samples,
            }
            # Estimate entropies and KL divergence between all distinct pairs
            for p_label, p in distributions.items():
                if p is not None:
                    final_metrics[f"entropy_{p_label}"] = stats.differential_entropy(p)
                    for q_label, q in distributions.items():
                        if p_label != q_label and q is not None:
                            label = f"KL_{p_label}_to_{q_label}"
                            kl = stats.kl_estimate(p, q)
                            final_metrics[label] = kl
            estimates[strategy_name] = final_metrics
    return estimates, final_samples, initial_samples
