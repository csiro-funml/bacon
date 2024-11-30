import torch
from gpytorch.module import Module


def prior_log_prob(model: Module, shape: torch.Size | tuple = ()):
    # Add log probs of priors on the (functions of) parameters
    log_probs = []
    for name, module, prior, closure, _ in model.named_priors():
        prior_term = prior.log_prob(closure(module)).view(*shape, -1).sum(dim=-1)
        log_probs.append(prior_term)

    total = torch.stack(log_probs, dim=-1).sum(dim=-1)
    return total
