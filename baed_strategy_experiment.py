import argparse
import os
from typing import Tuple
import warnings

import numpy as np
import submitit
import torch
import torch.distributions as td
import torch.multiprocessing as mp
from botorch.exceptions import InputDataWarning
from tqdm import trange

import baed4cal.problems as problems
from baed4cal.strategies.eig_strategy import FullGPStrategy
from baed4cal.strategies.observations import ObservationsDataset
from baed4cal.utils.launching import (
    get_strategies,
    common_initialisation,
    build_strategies,
    other_valid_strategies,
    prepare_executor,
    prepare_runners,
    save_copy,
    start_strategies,
    final_evaluations,
)
from baed4cal.utils.sampling import sample_posterior


def generate_initial_data(
    problem: problems.BAEDProblem, n_real, n_init
) -> Tuple[ObservationsDataset, ObservationsDataset]:
    design_dist = td.Uniform(low=problem.lower_bounds, high=problem.upper_bounds)
    real_xs = design_dist.sample(torch.Size([n_real]))
    sim_xs = torch.cat(
        [
            design_dist.sample(torch.Size([n_init])),
            problem.prior.sample(torch.Size([n_init])).view(n_init, -1),
        ],
        -1,
    )

    real_ys = torch.zeros(n_real)
    for i in trange(n_real):
        observation = problem.observe(real_xs[i])
        real_ys[i] = observation

    sim_ys = torch.zeros(n_init)
    for i in trange(n_init):
        observation = problem.simulate(sim_xs[i])
        sim_ys[i] = observation

    sim_data = ObservationsDataset(sim_xs, sim_ys)
    real_data = ObservationsDataset(real_xs, real_ys)
    return sim_data, real_data


def main(cl_args: argparse.Namespace):
    torch.set_default_dtype(torch.double)

    # Initialise logging and output directory
    logger, out_dir, seed = common_initialisation(
        cl_args,
        dir_prefix="strategy",
        logger_name=__name__,
        log_filename="baed_strategy.log",
        return_seed=True,
    )

    device = cl_args.device
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            torch.set_default_device(device)
        else:
            logger.critical("CUDA is not available")
            return
        mp.set_start_method("spawn")
    torch.set_default_device(device)

    # Create problem class
    problem_class_name = cl_args.problem_class
    problem = None
    source_dir = None
    if hasattr(problems, problem_class_name):
        try:
            problem_class = getattr(problems, problem_class_name)
            problem_args = tuple()
            problem_kwargs = dict()
            if issubclass(problem_class, problems.BAEDSyntheticProblem):
                problem_args += (cl_args.dimension,)
                if cl_args.noise is not None:
                    problem_kwargs["noise_sd"] = cl_args.noise
                if issubclass(problem_class, problems.SyntheticGPProblem):
                    warnings.simplefilter("ignore", category=InputDataWarning)
                    problem_kwargs["n_inducing"] = cl_args.n_inducing
            problem = problem_class(*problem_args, **problem_kwargs)
        except Exception:
            logger.exception(
                f"Unable to instantiate problem of class: '{problem_class_name}'."
            )
    elif os.path.isdir(problem_class_name):
        source_dir = problem_class_name
        try:
            problem = torch.load(
                os.path.join(source_dir, "problem.pth"), map_location=device
            )
            assert isinstance(problem, problems.BAEDSyntheticProblem)
        except Exception:
            logger.exception(
                f"Unable to load problem instance from: '{problem_class_name}'"
            )
    else:
        logger.error(f"Unknown problem class: '{problem_class_name}'.")
    if problem is None:
        logger.critical("Aborting due to error while instantiating problem class...")
        return

    # Setup initial data
    n_real = cl_args.n_real
    n_init = cl_args.n_init

    if cl_args.true_theta:
        try:
            true_theta = np.loadtxt(cl_args.true_theta)
            true_theta = torch.from_numpy(true_theta).to(problem.true_parameters)
            problem.true_parameters = true_theta
        except Exception as ex:
            logger.critical(
                "Unable to load true calibration parameter from file: %s",
                cl_args.true_theta,
                exc_info=ex,
            )
            return
    else:
        true_theta = problem.true_parameters

    try:
        torch.save(problem, os.path.join(out_dir, "problem.pth"))
    except Exception:
        logger.warning("Unable to save problem instance")

    # For performance evaluation purposes:
    n_assessment_samples = cl_args.n_assessment

    if source_dir is not None:
        logger.info("Loading initial data from previous run...")
        try:
            # Load initial data from previous run
            posterior_samples = torch.load(
                os.path.join(source_dir, "posterior_samples.pth"), map_location=device
            )
            for strategy_name in get_strategies():
                try:
                    strategy = torch.load(
                        os.path.join(source_dir, f"{strategy_name}.pth"),
                        map_location=device,
                    )
                    if isinstance(strategy, FullGPStrategy):
                        sim_data = ObservationsDataset(
                            strategy.sim_data.points[:n_init],
                            strategy.sim_data.outcomes[:n_init],
                        )
                        real_data = strategy.real_data
                    break
                except FileNotFoundError:
                    continue
        except Exception:
            logger.exception("Unable to load data from previous run")
            logger.critical("Aborting due to error loading data...")
            return 1
    else:
        logger.info("Generating initial data...")
        sim_data, real_data = generate_initial_data(problem, n_real, n_init)
        posterior_samples_file = cl_args.posterior_samples
        if posterior_samples_file is None:
            # Run MCMC
            logger.info("Sampling posterior given simulator...")
            n_chains = cl_args.n_chains
            n_mcmc_samples = n_assessment_samples // n_chains
            posterior_samples = sample_posterior(
                problem,
                real_data.points,
                real_data.outcomes,
                n_samples=n_mcmc_samples,
                n_chains=n_chains,
                show_progress_bar=True,
            )
            torch.save(
                posterior_samples, os.path.join(out_dir, "posterior_samples.pth")
            )
        else:
            logger.info(
                "Loading posterior samples from file '%s'...", posterior_samples_file
            )
            posterior_samples = torch.load(
                posterior_samples_file, map_location=real_data.outcomes.device
            )

    save_copy(cl_args.strategy_settings, "strategy_settings.yaml", out_dir, logger)

    # Generate test points for predictive performance evaluations
    n_test = cl_args.n_test
    if source_dir is not None and os.path.exists(
        os.path.join(source_dir, "test_data.pth")
    ):
        test_data = torch.load(
            os.path.join(source_dir, "test_data.pth"), map_location=device
        )
    else:
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

    torch.save(test_data, os.path.join(out_dir, "test_data.pth"))

    # =========
    # Run BAED
    # =========

    # Create strategies
    logger.info("Initialising strategies...")
    strategies = build_strategies(cl_args, problem, real_data, sim_data, logger)

    # Launch strategies
    if strategies:
        runners = prepare_runners(
            strategies,
            cl_args,
            problem,
            logger,
            out_dir,
            real_data=real_data,
            test_data=test_data,
            true_theta=true_theta,
            posterior_samples=posterior_samples,
            seed=seed,
        )
        logger.info("Waiting for strategies to finish running...")
        # Wait for strategies
        results = dict()
        if len(runners) == 1:
            strategy_name, runner = next(iter(runners.items()))
            logger.info(f"Starting '{strategy_name}' on main thread...")
            results[strategy_name] = runner()
        else:
            executor = prepare_executor(cl_args, out_dir)
            jobs: dict[str, submitit.Job] = start_strategies(runners, executor, logger)
            for strategy_name, job in jobs.items():
                job.wait()
                results[strategy_name] = job.result()

        # Update strategies
        for strategy_name, strategy in strategies.items():
            if isinstance(strategy, FullGPStrategy):
                strategies[strategy_name] = results[strategy_name][-1]

        # Run final performance evaluations
        final_evaluations(
            strategies, out_dir, cl_args, logger, posterior_samples=posterior_samples
        )
    else:
        logger.warning("No strategies to run")

    logger.info("Finished")
    return 0


if __name__ == "__main__":
    valid_strategies = get_strategies() + other_valid_strategies()

    parser = argparse.ArgumentParser(
        description="Bayesian adaptive experimental design on synthetic toy problem"
    )
    parser.add_argument(
        "problem_class",
        help="Class name or results directory of previous run with PyTorch files",
        type=str,
    )
    parser.add_argument(
        "strategy", help="Strategy class", choices=valid_strategies, nargs="+"
    )
    parser.add_argument(
        "-A",
        "--n-assessment",
        help="Number of assessment samples for performance evaluation",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "-B",
        "--n-batch",
        help="Number of points in batch for parallel evaluation",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-C", "--n-chains", help="Number of MCMC chains", type=int, default=8
    )
    parser.add_argument(
        "-I",
        "--n-init",
        help="Number of initial random simulation designs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-M", "--n-inducing", help="Number of inducing points", type=int, default=128
    )
    parser.add_argument(
        "-R",
        "--n-real",
        help="Number of 'real' data points observed",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-T", "--n-iterations", help="Number of iterations", type=int, default=10
    )
    parser.add_argument(
        "-S", "--n-samples", help="Number of samples for SVI", type=int, default=16
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="RNG seed. Default: Uses true random number as seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="Output directory where files will be saved",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-n", "--name", help="Name of the experiment", default=None, type=str
    )
    parser.add_argument(
        "-d",
        "--dimension",
        help="Design (and calibration parameters) dimension",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--strategy-settings",
        help="YAML file with settings for strategy hyper-parameters (constructor args) keyed by class name",
    )
    parser.add_argument(
        "--n-test",
        help="Number of grid points per dimension for prediction error evaluation",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--device",
        help="PyTorch device (+optional index, e.g., 'cuda:1') to use (Default: 'cpu')",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--timeout", help="Timeout (in minutes) for a job", type=int, default=60
    )
    parser.add_argument(
        "--memory",
        help="Memory (in GB) allocated to strategy runs",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--n-cpus",
        help="Number of CPU cores to allocate per job",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--posterior-samples",
        help="Posterior samples torch file to load (given true simulator), instead of running MCMC",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--true-theta",
        help="CSV file with true calibration parameter if it is fixed. Otherwise it will be sampled from the prior.",
        default=None,
    )
    parser.add_argument(
        "--noise",
        help="Custom setting for observation noise std. deviation",
        type=float,
        default=None,
    )

    args = parser.parse_args()
    main(args)
