from pyro.distributions.conditional import ConstantConditionalDistribution
from torch.distributions import Distribution


class BatchableConstantConditionalDistribution(ConstantConditionalDistribution):
    def __init__(self, base_dist: Distribution):
        super().__init__(base_dist)

    def condition(self, context) -> Distribution:
        original_dist = super().condition(context)
        batch_shape = context.shape[:-1]
        expanded_dist = original_dist.expand(batch_shape)
        return expanded_dist
