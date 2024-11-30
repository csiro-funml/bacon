import torch
from pyro.distributions import Empirical


class MCMCPosterior(Empirical):
    def __init__(
        self,
        samples: torch.Tensor,
        log_joint: torch.Tensor,
        validate_args: bool | None = None,
    ):
        """Build a wrapper around :class:`pyro.distributions.Empirical` that holds
        the log-joint probabilities of samples without interfering with the sampling
        procedure, ensuring that provided samples are considered as i.i.d., i.e.,
        `log_weights` are kept as equal weights.

        Args:
            samples (torch.Tensor): MCMC samples
            log_joint (torch.Tensor): (unnormalised) log-joint probabilities of samples
            validate_args (bool | None, optional): same as base class. Defaults to None.

        Raises:
            AssertionError: `log_joint` shape must be compatible with `samples` shape
        """
        log_weights = torch.zeros(samples.shape[:-1])
        if log_joint.shape != samples.shape[:-1]:
            raise AssertionError("Invalid shape for `log_joint`")
        self._log_joint = log_joint
        super().__init__(samples, log_weights, validate_args)

    @property
    def log_joint(self):
        """The log-joint (unnormalised posterior) probability of the samples."""
        return self._log_joint
