from typing import Optional

import torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior


def get_matern_kernel_with_gamma_prior(
    ard_num_dims: int,
    batch_shape: Optional[torch.Size] = None,
    active_dims: Optional[tuple[int]] = None,
    scale_output: bool = True,
    nu=2.5,
) -> ScaleKernel | MaternKernel:
    r"""Constructs the Scale-Matern kernel that is used by default by
    several models. This uses a Gamma(3.0, 6.0) prior for the lengthscale
    and a Gamma(2.0, 0.15) prior for the output scale.

    .. note:: Modified from BoTorch original implementation to include 'active_dims'

    Args:
        ard_num_dims (int): Number of input coordinates to use automatic relevant determination (ARD)
        batch_shape (Optional[torch.Size], optional): Batch shape. Defaults to None.
        active_dims (Optional[tuple[int]], optional): Active dimensions, i.e., indexes to apply the kernel on.
        scale_output (bool): If enabled, applies :class:`gpytorch.kernels.ScaleKernel` to output. Defaults to True.
        nu (float, optional): Kernel moothness parameter. Defaults to 2.5.

    Returns:
        ScaleKernel: Scaled Matern kernel
    """
    base_kernel = MaternKernel(
        nu=nu,
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        active_dims=active_dims,
        lengthscale_prior=GammaPrior(3.0, 6.0),
        lengthscale_constraint=GreaterThan(1e-4),
    )

    if scale_output:
        kernel = ScaleKernel(
            base_kernel=base_kernel,
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
    else:
        kernel = base_kernel

    return kernel
