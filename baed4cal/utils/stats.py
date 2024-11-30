import ite
import torch


def differential_entropy(samples: torch.Tensor) -> float:
    co = ite.cost.BHShannon_KnnK()
    ent = co.estimation(samples.detach().cpu().numpy())
    return ent


def kl_estimate(q_samples: torch.Tensor, p_samples: torch.Tensor) -> float:
    co = ite.cost.BDKL_KnnKiTi()
    estimate = co.estimation(q_samples.detach().cpu().numpy(), p_samples.detach().cpu().numpy())
    estimate = max(estimate, 0) # to ensure KL divergence estimate is non-negative
    return estimate


def mmd(q_samples: torch.Tensor, p_samples: torch.Tensor) -> float:
    co = ite.cost.BDMMD_VStat()
    estimate = co.estimation(q_samples.detach().cpu().numpy(), p_samples.detach().cpu().numpy())
    return estimate


def divergence(q_samples: torch.Tensor, p_samples: torch.Tensor, estimator_factory = ite.cost.BDKL_KnnK):
    co = estimator_factory()
    estimate = co.estimation(q_samples.detach().cpu().numpy(), p_samples.detach().cpu().numpy())
    return estimate
