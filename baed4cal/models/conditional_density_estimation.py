from abc import abstractmethod
from typing import Iterator, Optional

import linear_operator
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import _VariationalDistribution
from pyro.distributions import Distribution, ConditionalTransformedDistribution

from .invariant_model import InvariantModel


class ConditionalDistributionModel(nn.Module):
    @abstractmethod
    def _condition(self, context: torch.Tensor) -> Distribution:
        raise NotImplementedError()

    def _combine_inputs(
        self, points: torch.Tensor, outcomes: torch.Tensor
    ) -> torch.Tensor:
        points = torch.atleast_2d(points)
        outcomes = torch.atleast_1d(outcomes)
        batch_shape = torch.broadcast_shapes(points.shape[:-2], outcomes.shape[:-1])
        combined = torch.cat(
            [
                points.expand(*batch_shape, *points.shape[-2:]),
                outcomes.expand(*batch_shape, *outcomes.shape[-1:]).unsqueeze(-1),
            ],
            -1,
        )
        return combined

    def forward(
        self, outcomes: torch.Tensor, points: Optional[torch.Tensor] = None
    ) -> Distribution:
        """Compute conditional distribution given outcomes and (optionally) points

        Args:
            outcomes (torch.Tensor): (..., N) tensor with N outcomes per batch
            points (Optional[torch.Tensor], optional): (..., N, D) tensor with N D-dimensional points per batch.

        Returns:
            Distribution: The conditioned probability distribution
        """
        if points is None:
            context = outcomes.unsqueeze(-1)
        else:
            context = self._combine_inputs(points, outcomes)
        return self._condition(context)


class GaussianSetConditionalModel(ConditionalDistributionModel):
    def __init__(
        self,
        encoder: nn.Module,
        mean_emitter: nn.Module,
        covariance_factor_emitter: nn.Module,
    ) -> None:
        super().__init__()
        self.mean_model = InvariantModel(encoder, mean_emitter)
        self.covariance_factor_model = InvariantModel(
            encoder, covariance_factor_emitter
        )

    def _condition(self, context: torch.Tensor) -> MultivariateNormal:
        """Conditions on a set of data points, so that the conditional distribution is
           invariant with respect to the ordering of the data points

        Args:
            context (torch.Tensor): (..., N, D) array of data points, where N is the number of points,
                                    and D is their dimensionality

        Returns:
            MultivariateNormal: Gaussian conditional distribution
        """
        context = torch.atleast_2d(context)
        mean = self.mean_model(context)
        covariance_matrix = torch.diagflat(
            torch.exp(self.covariance_factor_model(context))
        )

        mvn = MultivariateNormal(
            mean,
            covariance_matrix=linear_operator.to_linear_operator(covariance_matrix),
        )
        return mvn


class GaussianConditionalModel(ConditionalDistributionModel):
    def __init__(
        self,
        encoder: nn.Module,
        mean_emitter: nn.Module,
        covariance_factor_emitter: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mean_model = mean_emitter
        self.covariance_factor_model = covariance_factor_emitter

    def _condition(self, context: torch.Tensor) -> MultivariateNormal:
        """Conditions each Gaussian on a single data point

        Args:
            context (torch.Tensor): (..., D) possibly-batched data point, where D is its dimensionality

        Returns:
            MultivariateNormal: Gaussian conditional distribution
        """
        context = context.view(*context.shape[:-2], -1)
        encoded = self.encoder(context)
        mean = self.mean_model(encoded)
        covariance_factor = self.covariance_factor_model(encoded)

        mvn = MultivariateNormal(
            mean,
            covariance_matrix=covariance_factor @ covariance_factor.transpose(-1, -2),
        )
        return mvn


class ConditionalNF(ConditionalDistributionModel):
    def __init__(
        self, conditional_distribution: ConditionalTransformedDistribution
    ) -> None:
        super().__init__()
        self.conditional_distribution = conditional_distribution
        for i, transform in enumerate(conditional_distribution.transforms):
            if isinstance(transform, nn.Module):
                self.add_module(f"tf{i}", transform)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return super().parameters(recurse)

    def _condition(self, context: torch.Tensor) -> Distribution:
        x = context.view(*context.shape[:-2], -1)
        return self.conditional_distribution.condition(x)


class ConditionalSetNF(ConditionalNF):
    def __init__(
        self,
        conditional_distribution: ConditionalTransformedDistribution,
        encoder: nn.Module,
        encoding_d: int,
        emitter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(conditional_distribution)
        self.encoder = encoder
        if emitter is None:
            emitter = nn.Identity()
        self.set_encoder = InvariantModel(encoder, emitter, encoding_d=encoding_d)

    def _condition(self, context: torch.Tensor) -> Distribution:
        encoded = self.set_encoder(context)
        return self.conditional_distribution.condition(encoded)


class BaseConditionalVariationalDistribution(_VariationalDistribution):
    @abstractmethod
    def condition(self, x: torch.Tensor):
        raise NotImplementedError()


class ConditionalVariationalDistribution(BaseConditionalVariationalDistribution):
    def __init__(
        self,
        cond_model: GaussianConditionalModel,
        init_conditioner: torch.Tensor,
        num_inducing_points: int,
        batch_shape: torch.Size = torch.Size([]),
    ):
        super().__init__(num_inducing_points, batch_shape)
        self.cond_model = cond_model
        self.conditioner = init_conditioner

    def initialize_variational_distribution(
        self, prior_dist: MultivariateNormal
    ) -> None:
        pass

    def condition(self, conditioner: torch.Tensor):
        self.conditioner = conditioner

    def forward(self) -> MultivariateNormal:
        return self.cond_model(self.conditioner)
