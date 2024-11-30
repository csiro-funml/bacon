from typing import Optional

from pyro.distributions import Distribution
from torch import Tensor
import torch.nn as nn
import zuko

from .conditional_density_estimation import ConditionalDistributionModel
from .invariant_model import InvariantModel


class ZukoConditionalModel(ConditionalDistributionModel):
    def __init__(
        self,
        conditional_distribution: zuko.flows.Flow,
        encoder: nn.Module,
        encoding_d: int,
        emitter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        if emitter is None:
            emitter = nn.Identity()
        self.set_encoder = InvariantModel(encoder, emitter, encoding_d=encoding_d)
        self.conditional_distribution = conditional_distribution

    def _condition(self, context: Tensor) -> Distribution:
        code = self.set_encoder(context)
        conditioned_distribution = self.conditional_distribution(code)
        return conditioned_distribution
