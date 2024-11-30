# BACON

Code repository for paper titled "Bayesian adaptive calibration and optimal design" (BACON) accepted at NeurIPS 2024.

## Abstract

We consider the problem of calibration of computer simulators from a Bayesian adaptive experimental design (BAED) perspective.
Our focus with this project is on complex simulation problems where the cost of running simulations is too high to allow for
traditional BAED methods, which require many evaluations of the likelihood to sample from the posterior or evaluate predictive
distributions. We, instead, model a simulator as a sample from a Gaussian process and treat the optimal parameters as a latent
input, which can then be inferred via variational inference methods. A lower bound on the expected information gain allows for
joint optimisation of both the design inputs and the variational posterior during the sequential experimentation process.

## Preprint

A preprint is currently available on ArXiv as [2405.14440](https://arxiv.org/abs/).

## Experiments data

Data from the paper results for the synthetic experiment and the location-finding experiment is available under `paper_results/` and can be plotted with the Jupyter notebook `plotting.ipynb`.

## Rerunning experiments

To rerun a trial of each experiment, please follow the instructions below.

### Dependencies

- Python 3.11
- Packages: see `requirements.txt`

### Installation

To run the experiments, you will need to install the Python version above and the packages in `requirements.txt`. One of the most convenient ways to do so is via virtual environment using Python's native virtual environment module `venv` or some higher-level toolbox, like Anaconda. With a suitable version of Python is installed, the environment can be setup with `venv` by running on a terminal under this package's root directory:

```bash
python -m venv  .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### Location-finding experiment

For the location finding experiment, run:

```bash
python baed_strategy_experiment.py LocationFindingProblem\
VBMC FullGPRandomStrategy FullGPIMSPEStrategy FullGPVariationalStrategy FullGPEntropyStrategy\
-A 4000 -B 4 -C 20 -I 20 -R 20 -T 30 -S 256 -d 2 --noise 0.5\
--timeout 480 --strategy-settings config/strategy-location.yaml\
--true-theta config/location-theta.csv -o experiments/location\
```

### Synthetic experiment

The synthetic experiment with simulators drawn from a Gaussian process prior operated with two pre-sampled problems under `data/fixed-problems`. To run the first example, with a unimodal target posterior, run:

```bash
python baed_strategy_experiment.py data/fixed-problems/strategy-20240521-031132.674517\
VBMC FullGPRandomStrategy FullGPIMSPEStrategy FullGPVariationalStrategy FullGPEntropyStrategy\
-A 4000 -B 4 -C 20 -I 20 -R 5 -T 50 -S 256 -d 2 --timeout 480\
--strategy-settings config/strategy-synthetic.yaml -o experiments/synthetic1
```

The second example, with a bimodal posterior, can be run with:

```bash
python baed_strategy_experiment.py data/fixed-problems/strategy-20241025-152913.295464\
VBMC FullGPRandomStrategy FullGPIMSPEStrategy FullGPVariationalStrategy FullGPEntropyStrategy\
-A 4000 -B 4 -C 20 -I 20 -R 5 -T 50 -S 256 -d 2 --timeout 480\
--strategy-settings config/strategy-synthetic.yaml -o experiments/synthetic2
```
