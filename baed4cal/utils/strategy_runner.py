import logging
import os
import time
import warnings
from numbers import Number
from typing import Dict, Optional, Tuple

import pandas as pd
import pyro
import torch
from botorch.exceptions import InputDataWarning
from botorch.models import SingleTaskGP
from calibration.models import PyroErrModel
from pyro.distributions import Empirical
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import initialization

import baed4cal.logging_config as logging_config
import baed4cal.problems as problems
from baed4cal.strategies.eig_strategy import (
    FullGPStrategy,
    ObservationsDataset,
    Strategy,
)
from baed4cal.utils.experiments import evaluate_strategy


class StrategyRunner:
    def __init__(
        self,
        problem: problems.BAEDProblem,
        n_batch: int,
        n_experiments: int,
        output_dir: Optional[str] = ".",
        test_data: Optional[torch.Tensor] = None,
        posterior_samples: Optional[torch.Tensor] = None,
        true_theta: Optional[torch.Tensor] = None,
        n_assessment_samples: int = 10000,
        n_samples: int = 32,
        n_chains: int = 1,
        seed: Optional[int] = None,
        save: bool = True,
        verbose: bool = True,
    ):
        """Runner for FullGPStrategy instances which runs and evaluates their performance throughout the experiment

        Args:
            problem (problems.BAEDProblem): BAED calibration problem
            n_batch (int): Number of points in batch of design points to generate each iteration
            n_experiments (int): Number of experiments (i.e., iterations) to run
            output_dir (Optional[str], optional): Output directory where to save files. Defaults to '.'.
            test_data (Optional[torch.Tensor], optional): Test data points for RMSE evaluation. Defaults to None.
            posterior_samples (Optional[torch.Tensor], optional): Samples of the true (MCMC) posterior when the simulator is fully known. Defaults to None.
            true_theta (Optional[torch.Tensor], optional): True parameters or a reference to measure errors against. Defaults to None.
            n_assessment_samples (int, optional): Number of samples to draw from estimated posteriors for performance assessments. Defaults to 10000.
            n_samples (int, optional): Number of predictive samples that the strategy uses. Defaults to 32.
            n_chains (int, optional): Number of MCMC chains to estimate the final posterior. Defaults to 1.
            seed (Optional[int], optional): RNG seed. Defaults to None.
            save (bool, optional): Enables saving results and variable states. Defaults to True.
            verbose (bool, optional): Enables logging. Defaults to True.
        """
        self.problem = problem
        self.output_dir = output_dir
        self.n_batch = n_batch
        self.n_experiments = n_experiments
        self.test_data = test_data
        self.posterior_samples = posterior_samples
        self.prior_samples = problem.prior.sample(torch.Size([n_assessment_samples]))
        self.true_theta = true_theta
        self.n_assessment_samples = n_assessment_samples
        self.n_samples = n_samples
        self.seed = seed
        self._save = save
        self._verbose = verbose
        self.n_chains = n_chains

    def _save_state(
        self,
        strategy_name: str,
        strategy: Strategy,
        metrics_data: pd.DataFrame | None,
    ):
        if self._save:
            rng_state = pyro.util.get_rng_state()
            torch.save(
                rng_state,
                os.path.join(self.output_dir, f"{strategy_name}-rng_state.pth"),
            )
            torch.save(strategy, os.path.join(self.output_dir, f"{strategy_name}.pth"))
            if metrics_data is not None:
                metrics_data.to_csv(
                    os.path.join(self.output_dir, f"{strategy_name}-metrics.csv")
                )

    def _get_posterior(self, strategy: FullGPStrategy) -> dict[str, torch.Tensor]:
        if isinstance(self.problem, problems.SyntheticGPProblem):
            if not isinstance(
                strategy.posterior, Empirical
            ):  # NOTE: We assume the strategy's initial posterior is MCMC-based if it is empirical
                theta_samples = strategy.mcmc_posterior(
                    n_samples=self.n_assessment_samples // self.n_chains,
                    n_chains=self.n_chains,
                ).enumerate_support()
            else:
                theta_samples = strategy.posterior.enumerate_support()
            posterior = {"theta": theta_samples}
        else:
            gp_model = SingleTaskGP(
                strategy.sim_data.points,
                strategy.sim_data.outcomes.unsqueeze(-1),
                train_Yvar=torch.ones_like(strategy.sim_data.outcomes).unsqueeze(-1)
                * 1e-6,
                covar_module=strategy.model.gp_model.covar_module.sim_kernel,
                outcome_transform=strategy.model.gp_model.outcome_transform,
                input_transform=strategy.model.gp_model.input_transform,
            )
            gp_model.eval()
            gp_model.requires_grad_(False)
            pyro_model = PyroErrModel(
                gp_model,
                prior=self.problem.prior,
                design_d=self.problem.design_dimension,
                noise=self.problem.noise_sd,
                err_kernel_class=strategy.model.gp_model.covar_module.err_kernel.base_kernel.__class__.__name__,
            )
            mcmc_kernel = NUTS(
                pyro_model,
                jit_compile=True,
                init_strategy=initialization.init_to_sample,
            )
            mcmc = MCMC(
                mcmc_kernel,
                num_samples=self.n_assessment_samples // self.n_chains,
                num_chains=self.n_chains,
                mp_context="spawn",
            )
            mcmc.run(
                strategy.real_data.points, strategy.real_data.outcomes.unsqueeze(-1)
            )
            posterior = mcmc.get_samples()

        return posterior

    def _evaluate(self, strategy: FullGPStrategy, start_time=None):
        metrics = evaluate_strategy(
            strategy,
            test_data=self.test_data,
            prior_samples=self.prior_samples,
            posterior_samples=self.posterior_samples,
            true_theta=self.true_theta,
            n_assessment_samples=self.n_assessment_samples,
            n_pred_samples=self.n_samples,
            start_time=start_time,
        )
        return metrics

    def __call__(
        self, strategy: FullGPStrategy, strategy_name: str = ""
    ) -> Tuple[pd.DataFrame, Dict[str, Number]]:
        # Setup subprocess environment
        if isinstance(self.problem, problems.SyntheticGPProblem):
            warnings.simplefilter("ignore", category=InputDataWarning)

        torch.set_default_device(strategy.real_data.outcomes.device)
        torch.set_default_dtype(strategy.real_data.outcomes.dtype)

        if self.seed is not None:
            pyro.set_rng_seed(self.seed)

        if not strategy_name:
            strategy_name = strategy.__class__.__name__

        # Setup logger
        if self._verbose:
            if self._save:
                logger = logging_config.make_file_logger(
                    os.path.join(self.output_dir, f"{strategy_name}.log"),
                    logger_name=strategy_name,
                    include_warnings=True,
                )
            else:
                logger = logging.getLogger(strategy_name)

        records = list()
        metrics_data = None
        start_time = None

        try:
            for t in range(self.n_experiments):
                if self._verbose:
                    logger.info(
                        "Running experiment %i/%i for strategy '%s'",
                        t + 1,
                        self.n_experiments,
                        strategy_name,
                    )

                if t == 0:
                    start_time = time.time()
                    estimation_start = time.time()
                    try:
                        # Get initial posterior for information gain evaluation at the end
                        logger.info("Estimating initial posterior...")
                    except Exception:
                        logger.exception(
                            "Unable to obtain initial posterior for performance assessments..."
                        )
                        logger.warning(
                            "Proceeding without initial posterior estimate..."
                        )
                    estimation_time = time.time() - estimation_start
                    start_time += estimation_time  # to compensate for MCMC runtime, which is not part of the strategy

                # Evaluate current fit
                if self._verbose:
                    logger.info("Evaluating strategy..")
                metrics = self._evaluate(strategy, start_time=start_time)
                if self._verbose:
                    for metric_name, metric_value in metrics.items():
                        logger.info(
                            f"({t}/{self.n_experiments}) Metric '{metric_name}': {metric_value}"
                        )

                ## Save evaluation results
                records.append(metrics)
                metrics_data = pd.DataFrame.from_records(records)
                self._save_state(strategy_name, strategy, metrics_data)

                # ============================
                # Run optimisation of the EIG
                # ============================
                if self._verbose:
                    logger.info("Generating candidates...")
                x = strategy.generate(self.n_batch)

                # Collect observation and update dataset
                if self._verbose:
                    logger.info("Running simulations...")
                ## Run new simulation at optimal design point
                try:
                    with torch.no_grad():
                        sim_x_t = torch.atleast_2d(x)
                        sim_y_t = self.problem.simulate(sim_x_t)
                        batch_data = ObservationsDataset(sim_x_t, sim_y_t)
                except RuntimeError:
                    logger.exception("Error occurred during simulation...")
                    logger.warning("Proceeding without new batch")
                    continue

                # ---------------
                # Update strategy
                # ---------------
                if self._verbose:
                    logger.info("Updating strategy...")

                strategy.update(batch_data)

            # ==================
            # Final evaluations
            # ==================
            if self._verbose:
                logger.info("Evaluating final results..")

            ## Evaluate previous metrics
            metrics = evaluate_strategy(
                strategy,
                test_data=self.test_data,
                prior_samples=self.prior_samples,
                posterior_samples=self.posterior_samples,
                true_theta=self.true_theta,
                n_assessment_samples=self.n_assessment_samples,
                n_pred_samples=self.n_samples,
                start_time=start_time,
            )
            records.append(metrics)
            metrics_data = pd.DataFrame.from_records(records)

            ## Log results
            if self._verbose:
                for metric_name, metric_value in metrics.items():
                    logger.info(
                        f"({self.n_experiments}/{self.n_experiments}) Metric '{metric_name}': {metric_value}"
                    )

        except RuntimeError:
            if self._verbose:
                logger.exception(
                    f"Runtime error occurred while running strategy '{strategy_name}'. Aborting..."
                )
        except Exception:
            if self._verbose:
                logger.exception(
                    f"Unexpected error exception while running strategy '{strategy_name}'. Aborting..."
                )

        if self._verbose and self._save:
            logger.info("Saving final results...")
        self._save_state(
            strategy_name,
            strategy,
            metrics_data,
        )
        if self._verbose:
            logger.info("Done")

        return metrics_data, strategy
