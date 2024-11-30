from functools import partial
from typing import Callable, Optional

import pyro.distributions as pyd
import pyro.distributions.transforms as tf
import torch
import torch.nn as nn
import zuko

from baed4cal.models.conditional_density_estimation import (
    ConditionalVariationalDistribution,
    GaussianConditionalModel,
    ConditionalNF,
    ConditionalSetNF,
)
from baed4cal.models.constant_conditional import (
    BatchableConstantConditionalDistribution,
)
from baed4cal.models.covariance_layers import CholeskyLayer
from baed4cal.models.joint_model import (
    JointSparseGPModel,
    ConditionalOptimalInducingVariableDistribution,
    JointFullGP,
)
from baed4cal.models.zuko_integration import ZukoConditionalModel


def training_points_mapper(
    train_x_full: torch.Tensor, train_x_partial: torch.Tensor, theta: torch.Tensor
):
    cond_x = torch.cat(
        [
            train_x_partial.expand(*theta.shape[:-1], *train_x_partial.shape),
            theta.unsqueeze(-2).expand(
                *theta.shape[:-1], *train_x_partial.shape[:-1], -1
            ),
        ],
        -1,
    )
    full_points = torch.cat(
        [cond_x, train_x_full.expand(*cond_x.shape[:-2], *train_x_full.shape)], -2
    )
    return full_points


def create_model_gp(
    kernel, inducing_points, train_x_full, train_x_partial, train_y
) -> JointSparseGPModel:
    points_mapper = partial(training_points_mapper, train_x_full, train_x_partial)
    conditional = ConditionalOptimalInducingVariableDistribution(
        inducing_points, kernel, train_y.unsqueeze(-1), points_mapper
    )
    gp_model = JointSparseGPModel(conditional, inducing_points, covar_module=kernel)
    return gp_model


def create_model_gp_full(kernel, sim_x, sim_y, real_x, real_y, likelihood=None):
    model = JointFullGP(sim_x, sim_y, real_x, real_y, kernel, likelihood=likelihood)
    return model


def init_net_parameters(net: nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("tanh"))
            if m.bias is not None:
                m.bias.detach().zero_()


def create_model_nn(
    kernel, inducing_points, unknown_dim, n_hidden=32
) -> JointSparseGPModel:
    n_inducing = inducing_points.shape[0]

    encoder = nn.Sequential(nn.Linear(unknown_dim, n_hidden), nn.Tanh())
    mean_emitter = nn.Sequential(nn.Linear(n_hidden, n_inducing))
    cov_factor_emitter = nn.Sequential(
        nn.Linear(n_hidden, n_inducing * (n_inducing + 1) // 2),
        CholeskyLayer(n_inducing),
    )
    gaussian_conditional = GaussianConditionalModel(
        encoder, mean_emitter, cov_factor_emitter
    )
    init_net_parameters(gaussian_conditional)

    variational_conditional = ConditionalVariationalDistribution(
        gaussian_conditional, torch.randn(unknown_dim), num_inducing_points=n_inducing
    )
    gp_model = JointSparseGPModel(
        variational_conditional, inducing_points, covar_module=kernel
    )
    return gp_model


def get_cnf_factory(
    flow_type: str, dimension: int = 1
) -> Callable[..., pyd.transforms.Transform]:
    """Get Pyro callable for conditional normalising flow factory

    Args:
        flow_type (str): Type of CNF. Available: 'Spline', 'IAF' and 'RealNVP'
        dimension (int, optional): Dimension of the sample space, which, for spline flows, determines whether to use the autorgressive version (D > 1) or not. Defaults to 1.

    Raises:
        ValueError: If flow type is not among 'Spline', 'IAF' and 'RealNVP'

    Returns:
        Callable[..., pyd.transforms.Transform]: Pyro callable that produces a conditional transform given by the normalising flow architecture
    """
    if flow_type == "Spline":
        if dimension == 1:
            return tf.conditional_spline
        else:
            return tf.conditional_spline_autoregressive
    elif flow_type == "IAF":
        return tf.conditional_affine_autoregressive
    elif flow_type == "RealNVP":
        return tf.conditional_affine_coupling
    else:
        raise ValueError(
            "Unrecognised flow type. Available are 'Spline', 'IAF' and 'RealNVP'"
        )


def create_conditional_nf(
    theta_d: int,
    obs_d: int = 1,
    n_obs: int = 1,
    n_transforms: int = 2,
    flow_type: str = "Spline",
    **flow_kwargs
):
    base_dist = pyd.MultivariateNormal(torch.zeros(theta_d), torch.eye(theta_d))
    context_d = obs_d * n_obs
    flow_factory = get_cnf_factory(flow_type)
    cond_tfs = [
        flow_factory(theta_d, context_dim=context_d, **flow_kwargs)
        for _ in range(n_transforms)
    ]
    cond_dist = pyd.ConditionalTransformedDistribution(base_dist, cond_tfs)
    cond_nf = ConditionalNF(cond_dist)
    init_net_parameters(cond_nf)
    return cond_nf


def create_mlp(
    input_d: int,
    output_d: int,
    n_hidden: int,
    n_layers: int = 1,
    activation: str = "ReLU",
    output_activation: Optional[str] = None,
) -> nn.Module:
    activation_cls = getattr(nn, activation)
    layers = [nn.Linear(input_d, n_hidden), activation_cls()]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(activation_cls())
    layers.append(nn.Linear(n_hidden, output_d))
    if output_activation is not None:
        layers.append(getattr(nn, output_activation)())
    mlp = nn.Sequential(*layers)
    return mlp


def create_conditional_set_nf(
    theta_d: int,
    obs_d: int = 1,
    n_obs: int = 1,
    n_transforms: int = 2,
    n_hidden: int = 32,
    encoding_d: int = 16,
    activation: str = "ReLU",
    n_layers: int = 2,
    flow_type: str = "Spline",
    **flow_kwargs
):
    # Base distribution for NF
    unbatched_base_dist = pyd.MultivariateNormal(
        torch.zeros(theta_d), torch.eye(theta_d)
    )
    base_dist = BatchableConstantConditionalDistribution(unbatched_base_dist)
    # Build encoder NN for deep set conditioning model
    encoder = create_mlp(obs_d, encoding_d, n_hidden, n_layers, activation)
    # Build CNF transform layers
    flow_factory = get_cnf_factory(flow_type, theta_d)
    cond_tfs = [
        flow_factory(theta_d, context_dim=encoding_d, **flow_kwargs)
        for _ in range(n_transforms)
    ]
    cond_dist = pyd.ConditionalTransformedDistribution(base_dist, cond_tfs)
    cond_nf = ConditionalSetNF(cond_dist, encoder, encoding_d=encoding_d)
    # init_net_parameters(cond_nf)
    return cond_nf


def create_conditional_gaussian(
    theta_d: int,
    obs_d: int,
    n_obs: int = 1,
    n_hidden: int = 32,
    encoding_d: int = 32,
    activation: str = "Tanh",
    n_layers: int = 1,
):
    context_d = obs_d * n_obs
    encoder = create_mlp(
        context_d,
        encoding_d,
        n_hidden,
        n_layers=n_layers,
        activation=activation,
        output_activation=activation,
    )
    mean_emitter = nn.Sequential(nn.Linear(encoding_d, theta_d))
    cov_factor_emitter = nn.Sequential(
        nn.Linear(encoding_d, theta_d * (theta_d + 1) // 2), CholeskyLayer(theta_d)
    )
    gaussian_conditional = GaussianConditionalModel(
        encoder, mean_emitter, cov_factor_emitter
    )
    init_net_parameters(gaussian_conditional)
    return gaussian_conditional


def create_zuko_cnf(
    theta_d: int,
    obs_d: int = 1,
    n_obs: int = 1,
    n_transforms: int = 2,
    n_hidden: int = 32,
    encoding_d: int = 16,
    activation: str = "ReLU",
    n_layers: int = 2,
    flow_type: str = "NSF",
    **flow_kwargs
):
    flow_class = getattr(zuko.flows, flow_type)
    flow = flow_class(
        features=theta_d, context=encoding_d, transforms=n_transforms, **flow_kwargs
    )
    encoder = create_mlp(obs_d, encoding_d, n_hidden, n_layers, activation)
    cond_nf = ZukoConditionalModel(flow, encoder, encoding_d=encoding_d)
    return cond_nf
