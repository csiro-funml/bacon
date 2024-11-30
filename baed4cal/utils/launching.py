import argparse
from collections import defaultdict
from functools import partial
import inspect
import logging
import logging.config
import os
import random
import shutil
import subprocess
from datetime import datetime
from typing import Callable, Dict, Optional
from baed4cal.strategies.observations import ObservationsDataset

import gpytorch
import pyro
import torch
import torch.multiprocessing as mp
import yaml
import copy
import submitit
import pandas as pd

from botorch.models.transforms import Normalize, Standardize

from kohgp.models.kernels import BiFiKernel, ZeroKernel

import baed4cal.models.factories as factories
import baed4cal.logging_config as logging_config
from baed4cal.models.conditional_density_estimation import ConditionalDistributionModel
import baed4cal.problems as problems
import baed4cal.strategies.eig_strategy
from baed4cal.strategies.eig_strategy import (
    FullGPIMSPEStrategy,
    FullGPMCMCStrategy,
    FullGPRandomStrategy,
    FullGPStrategy,
    FullGPEntropyStrategy,
    FullGPVariationalStrategy,
)
from baed4cal.strategies.vbmc import VBMCRunner
from baed4cal.utils.experiments import evaluate_kls
from baed4cal.utils.git import get_git_revision_hash
from baed4cal.utils.strategy_runner import StrategyRunner


def make_gp_kwargs(problem: problems.BAEDProblem):
    kwargs = dict()
    input_transform = None
    outcome_transform = None
    if isinstance(problem, problems.KOHSyntheticGPProblem):
        kernel = copy.deepcopy(problem.simulator.covar_module)
        mean_module = problem.simulator.mean_module
    elif isinstance(problem, problems.SyntheticGPProblem):
        sim_kernel = copy.deepcopy(problem.simulator.covar_module)
        sim_kernel.active_dims = torch.tensor(
            range(problem.dimension), dtype=torch.long
        )
        kernel = BiFiKernel(
            problem.design_dimension,
            problem.calibration_dimension,
            sim_kernel=sim_kernel,
            err_kernel=ZeroKernel(),
            use_scale=False,
        )
        mean_module = problem.simulator.mean_module
    elif isinstance(problem, problems.LocationFindingProblem):
        kernel = BiFiKernel(
            problem.design_dimension,
            problem.calibration_dimension,
            use_ard=True,
            use_scale=False,
            err_kernel=ZeroKernel(),
            nu=0.5,
        )
        mean_module = gpytorch.means.ZeroMean()
        input_transform = Normalize(
            problem.dimension, indices=list(range(problem.dimension))
        )
        outcome_transform = Standardize(1)
    elif isinstance(problem, problems.ClassicTestFunctionProblem):
        kernel = BiFiKernel(
            problem.design_dimension,
            problem.calibration_dimension,
            err_kernel=ZeroKernel(),
            use_ard=True,
            use_scale=False,
        )
        mean_module = gpytorch.means.ConstantMean()
        input_transform = Normalize(
            problem.dimension, indices=list(range(problem.dimension))
        )
        outcome_transform = Standardize(1)
    else:
        kernel = BiFiKernel(
            problem.design_dimension, problem.calibration_dimension, use_ard=True
        )
        mean_module = gpytorch.means.ZeroMean()
        input_transform = Normalize(
            problem.dimension, indices=list(range(problem.dimension))
        )
        outcome_transform = Standardize(1)

    kwargs.update(
        covar_module=kernel,
        mean_module=mean_module,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    return kwargs


CONDITIONAL_MODEL_FACTORIES = {
    "Gaussian": factories.create_conditional_gaussian,
    "CNF": factories.create_conditional_nf,
    "SetCNF": factories.create_conditional_set_nf,
    "Zuko": factories.create_zuko_cnf,
}


def get_conditional_model_factory(
    name: str,
) -> Callable[..., ConditionalDistributionModel]:
    return CONDITIONAL_MODEL_FACTORIES[name]


def make_strategy(
    strategy_name: str,
    problem: problems.BAEDProblem,
    real_data: ObservationsDataset,
    sim_data: ObservationsDataset,
    observation_noise_sd: float,
    n_samples: int,
    n_assessment_samples: int = 2000,
    n_chains: int = 8,
    settings: Optional[dict] = None,
):
    strategy_class = getattr(baed4cal.strategies.eig_strategy, strategy_name)
    strategy = None
    if settings is None:
        settings = dict()

    if "retrain_gp" not in settings:
        settings["retrain_gp"] = not isinstance(problem, problems.SyntheticGPProblem)

    if issubclass(strategy_class, FullGPStrategy):
        gp_kwargs = make_gp_kwargs(problem)
        if strategy_class is FullGPVariationalStrategy:
            conditional_model_factory = None
            conditional_model_settings = settings.pop("conditional", None)
            if conditional_model_settings is not None:
                conditional_model_type = conditional_model_settings.pop(
                    "type", "SetCNF"
                )
                conditional_model_factory = get_conditional_model_factory(
                    conditional_model_type
                )
                if conditional_model_settings:  # check if any settings remain
                    conditional_model_factory = partial(
                        conditional_model_factory, **conditional_model_settings
                    )
            strategy = FullGPVariationalStrategy(
                problem,
                real_data,
                sim_data,
                observation_noise_sd=observation_noise_sd,
                conditional_model_factory=conditional_model_factory,
                n_samples=n_samples,
                n_init_chains=n_chains,
                n_init_samples=n_assessment_samples // n_chains,
                **gp_kwargs,
                **settings,
            )
        elif strategy_class is FullGPRandomStrategy:
            strategy = FullGPRandomStrategy(
                problem,
                real_data,
                sim_data,
                observation_noise_sd=observation_noise_sd,
                **gp_kwargs,
                **settings,
            )
        elif strategy_class is FullGPIMSPEStrategy:
            strategy = FullGPIMSPEStrategy(
                problem,
                real_data,
                sim_data,
                observation_noise_sd=observation_noise_sd,
                n_prediction_designs=n_samples,
                **gp_kwargs,
                **settings,
            )
        elif strategy_class is FullGPMCMCStrategy:
            strategy = FullGPMCMCStrategy(
                problem,
                real_data,
                sim_data,
                observation_noise_sd=observation_noise_sd,
                n_chains=n_chains,
                n_samples=n_samples,
                **gp_kwargs,
                **settings,
            )
        elif strategy_class is FullGPEntropyStrategy:
            strategy = FullGPEntropyStrategy(
                problem,
                real_data,
                sim_data,
                observation_noise_sd=observation_noise_sd,
                n_chains=n_chains,
                n_samples=n_assessment_samples // n_chains,
                **gp_kwargs,
                **settings,
            )
        else:
            raise RuntimeError("Invalid strategy class")
    return strategy


def get_strategies(base_class=FullGPStrategy):
    strategies = []
    for name, obj in inspect.getmembers(baed4cal.strategies.eig_strategy):
        if inspect.isclass(obj):
            if issubclass(obj, base_class):
                if not inspect.isabstract(obj):
                    strategies += [name]
    return strategies


def other_valid_strategies():
    return ["VBMC"]


def common_initialisation(
    cl_args,
    dir_prefix: str,
    logger_name: str,
    log_filename: str = "main.log",
    return_seed: bool = False,
):
    if hasattr(cl_args, "seed"):
        seed = cl_args.seed
        if seed is None:
            rng = random.SystemRandom()
            seed = rng.getrandbits(32)
        pyro.set_rng_seed(seed)
    else:
        seed = None

    base_dir = cl_args.output_directory
    dir_name = dir_prefix + "-" + datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    out_dir = os.path.join(base_dir, dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    args_dict = vars(cl_args)
    with open(os.path.join(out_dir, "cl_args.yaml"), "w") as f:
        yaml.safe_dump(args_dict, f)

    log_file_name = os.path.join(out_dir, log_filename)
    logger = logging_config.make_file_logger(
        log_file_name, logger_name=logger_name, include_warnings=True
    )

    logger.info("Output will be saved to '%s'", out_dir)
    logger.debug("Command-line arguments:")
    for arg, val in args_dict.items():
        logger.debug(f"{arg}: {val}")
    if seed is not None:
        logger.info("Seed: %i", seed)
        with open(os.path.join(out_dir, "seed.csv"), "w") as f:
            f.write(str(seed) + os.linesep)

    try:
        git_rev = get_git_revision_hash()
        logger.debug("Git revision hash: %s", git_rev)
        with open(os.path.join(out_dir, "git_hash.txt"), "w") as f:
            f.write(git_rev + os.linesep)
    except subprocess.CalledProcessError:
        logger.warning("Unable to get git revision hash")

    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
        logger.info(f"SLURM job ID: {job_id}")

    if return_seed:
        return logger, out_dir, seed
    else:
        return logger, out_dir


def generate_test_data(problem: problems.BAEDProblem, n_test: int):
    test_xs = torch.stack(
        torch.meshgrid(
            [
                torch.linspace(lb, ub, n_test)
                for lb, ub in zip(
                    problem.lower_bounds, problem.upper_bounds, strict=True
                )
            ],
            indexing="xy",
        ),
        -1,
    ).view(-1, problem.design_dimension)
    test_outcomes = problem.observe(test_xs)
    test_data = ObservationsDataset(test_xs, test_outcomes)
    return test_data


def build_strategies(cl_args, problem, real_data, sim_data, logger: logging.Logger):
    # Load strategy
    strategy_settings = dict()
    if cl_args.strategy_settings:
        ss_filename = cl_args.strategy_settings
        try:
            with open(ss_filename) as f:
                strategy_settings = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("Provided strategy settings file not found: %s", ss_filename)
        except yaml.YAMLError as ex:
            logger.error(
                f"Formatting error with YAML strategy settings in '{ss_filename}':\n{ex}"
            )
        except Exception:
            logger.exception(
                "Unexpected error while trying to load strategy settings in '%s'",
                ss_filename,
            )
        if not strategy_settings:
            logger.warning("Default strategy settings will be used...")

    # For performance evaluation purposes:
    n_assessment_samples = cl_args.n_assessment

    n_samples = cl_args.n_samples
    n_chains = cl_args.n_chains
    if hasattr(problem, "noise_sd"):
        observation_noise_sd = problem.noise_sd
    elif hasattr(cl_args, "observation_noise"):
        observation_noise_sd = cl_args.observation_noise
    else:
        observation_noise_sd = None
        logger.warning("Observation noise level not set")

    # Create strategies
    strategy_names: list = cl_args.strategy

    strategies = defaultdict(set)
    try:
        for strategy_name in strategy_names:
            if hasattr(baed4cal.strategies.eig_strategy, strategy_name):
                if strategy_name in strategy_settings:
                    logger.info(
                        "Using custom settings for strategy '%s'", strategy_name
                    )
                    settings = strategy_settings[strategy_name]
                else:
                    settings = None
                strategy = make_strategy(
                    strategy_name,
                    problem,
                    real_data,
                    sim_data,
                    observation_noise_sd=observation_noise_sd,
                    n_samples=n_samples,
                    n_assessment_samples=n_assessment_samples,
                    n_chains=n_chains,
                    settings=settings,
                )
                if strategy is None:
                    logger.error(f"Invalid strategy: '{strategy_name}'")
                else:
                    strategies[strategy_name] = strategy
            else:
                strategies[strategy_name] = None
    except Exception:
        logger.exception("Unable to initialise strategies...")
        return None

    return strategies


def prepare_executor(cl_args, out_dir):
    execution_settings = {
        "mem_gb": cl_args.memory,
        "nodes": 1,
        "cpus_per_task": cl_args.n_cpus,
        "tasks_per_node": 1,
        "timeout_min": cl_args.timeout,
        "name": "runner",
    }
    if cl_args.device.startswith("cuda"):
        execution_settings["gpus_per_node"] = 1

    executor = submitit.LocalExecutor(os.path.join(out_dir, "job-%j"))
    executor.update_parameters(**execution_settings)
    return executor


def prepare_runners(
    strategies: dict,
    cl_args,
    problem: problems.BAEDProblem,
    logger: logging.Logger,
    out_dir: str,
    real_data: Optional[torch.Tensor] = None,
    test_data: Optional[torch.Tensor] = None,
    true_theta: Optional[torch.Tensor] = None,
    posterior_samples: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> Dict[str, Callable]:
    rng_state = pyro.util.get_rng_state()
    torch.save(rng_state, os.path.join(out_dir, "rng_state.pth"))

    n_experiments = cl_args.n_iterations
    n_samples = cl_args.n_samples
    n_batch = cl_args.n_batch
    n_real = real_data.points.shape[0]
    n_init = cl_args.n_init
    n_assessment_samples = cl_args.n_assessment
    n_chains = cl_args.n_chains

    runners = dict()

    # Start subprocesses
    for strategy_name, strategy in strategies.items():
        if isinstance(strategy, FullGPStrategy):
            runner = StrategyRunner(
                problem,
                n_batch,
                n_experiments,
                out_dir,
                test_data,
                posterior_samples,
                true_theta,
                n_assessment_samples,
                n_samples,
                n_chains=n_chains,
                seed=seed,
            )
            runners[strategy_name] = partial(runner, strategy, strategy_name)
        elif strategy_name == "VBMC":
            runner = VBMCRunner(
                problem,
                real_data,
                max_eval=(n_init + n_batch * n_experiments) // n_real,
                out_dir=out_dir,
                posterior_samples=posterior_samples,
                n_assessment_samples=n_assessment_samples,
                seed=seed,
            )
            runners[strategy_name] = runner
        else:
            logger.error(f"Skipping unknown strategy '{strategy_name}'...")
            continue
    return runners


def start_strategies(
    runners: Dict[str, Callable], executor: submitit.Executor, logger: logging.Logger
):
    logger.info("Starting strategies...")
    jobs = dict()
    for strategy_name, runner in runners.items():
        job = executor.submit(runner)
        logger.info(
            f"Started '{strategy_name}' strategy subprocess (PID: {job.job_id})..."
        )
        jobs[strategy_name] = job
    return jobs


def launch_strategies(
    strategies: dict,
    cl_args,
    problem: problems.BAEDProblem,
    logger: logging.Logger,
    out_dir: str,
    real_data: Optional[torch.Tensor] = None,
    test_data: Optional[torch.Tensor] = None,
    true_theta: Optional[torch.Tensor] = None,
    posterior_samples: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> Dict[str, mp.Process]:
    processes = prepare_runners(
        strategies,
        cl_args,
        problem,
        logger,
        out_dir,
        real_data,
        test_data,
        true_theta,
        posterior_samples,
        seed,
    )
    start_strategies(processes, logger)
    return processes


def save_copy(
    filename: str | None,
    dest_name: str,
    dest_dir: str,
    logger: logging.Logger | None = None,
):
    if filename is not None:
        try:
            shutil.copyfile(filename, os.path.join(dest_dir, dest_name))
        except OSError as ex:
            if logger is not None:
                logger.error(
                    "Unable to save a copy of file '%s' to '%s'", filename, dest_dir
                )
                logger.debug(ex)


def final_evaluations(
    strategies: dict,
    out_dir: str,
    cl_args: argparse.Namespace,
    logger: logging.Logger,
    posterior_samples: Optional[torch.Tensor] = None,
    show_progress: bool = False,
):
    logger.info("Evaluating KL divergences...")
    ref_strategy = None
    if "FullGPRandomStrategy" in strategies.keys():
        ref_strategy = "FullGPRandomStrategy"
    else:
        for strategy_name in strategies.keys():
            if isinstance(strategies[strategy_name], FullGPStrategy):
                ref_strategy = strategy_name
                break
    if ref_strategy is not None:
        kls, final_samples, initial_samples = evaluate_kls(
            strategies,
            n_init=cl_args.n_init,
            posterior_samples=posterior_samples,
            ref_strategy_name=ref_strategy,
            n_samples=cl_args.n_assessment,
            n_chains=cl_args.n_chains,
            show_progress=show_progress,
        )
        kl_df = pd.DataFrame.from_dict(kls)
        kl_df.to_csv(os.path.join(out_dir, "KL.csv"))
        torch.save(initial_samples, os.path.join(out_dir, "initial_samples.pth"))
        for strategy_name, samples in final_samples.items():
            torch.save(
                samples, os.path.join(out_dir, f"{strategy_name}-final_samples.pth")
            )
            pd.DataFrame(kls[strategy_name], index=[0]).to_csv(
                os.path.join(out_dir, f"{strategy_name}-final.csv")
            )
