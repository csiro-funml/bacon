from typing import Optional, Tuple, Union
from gpytorch.constraints import Interval, Positive
from kohgp.models.utils import get_matern_kernel_with_gamma_prior

import torch
from gpytorch.kernels import (
    AdditiveKernel,
    Kernel,
)
from gpytorch.priors import LogNormalPrior, Prior
from linear_operator.operators import LinearOperator
from torch import Tensor
from torch.nn import Parameter


class ZeroKernel(Kernel):
    """A convenience kernel class for problems which use no simulation error kernel."""

    def __init__(self, *args, **kwargs):
        """Arguments are passed directly to the base class. This module has no parameters."""
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        batch_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
        res = torch.zeros([], device=x1.device, dtype=x1.dtype).expand(
            *batch_shape, x1.shape[-2], x2.shape[-2]
        )
        return res


class ConstantKernel(Kernel):
    """A convenience kernel class for problems which use no scaling kernel."""

    def __init__(self, *args, **kwargs):
        """Arguments are passed directly to the base class. This module has no parameters."""
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        batch_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
        res = torch.ones([], device=x1.device, dtype=x1.dtype).expand(
            *batch_shape, x1.shape[-2], x2.shape[-2]
        )
        return res


class BiFiIndexKernel(Kernel):
    r"""
    A Kronecker delta kernel for use within a bi-fidelity kernel, like `BiFiKernel` below.

    .. math::

        k_\delta (s, s') = s \cdot s', \quad s, s' \in \{0,1\}
    """

    def __init__(self, fidelity_coord: int = -1):
        """
        Args:
            fidelity_coord (int, optional): Coordinate corresponding to the fidelity parameter. Defaults to -1.
        """
        self.has_lengthscale = False
        self.fidelity_coord = fidelity_coord
        super().__init__()

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Union[Tensor, LinearOperator]:
        return torch.einsum(
            "...n,...m->...nm",
            x1[..., self.fidelity_coord],
            x2[..., self.fidelity_coord],
        )


class BiFiScalingKernel(Kernel):
    r"""Bi-Fidelity Scaling kernel"""

    def __init__(
        self,
        scale_prior: Prior | None = None,
        scale_constraint: Interval | None = None,
        batch_shape: Optional[torch.Size] = None,
        active_dims: Optional[Tuple[int, ...]] = None,
    ):
        """This kernel applies scaling dependent on its input, which is treated as a fidelity parameter.
        It is assumed that low-fidelity components affect high-fidelity outcomes via linear scaling.

        Args:
            scale_prior (Prior | None, optional): _description_. Defaults to None.
            scale_constraint (Interval | None, optional): _description_. Defaults to None.
            batch_shape (Optional[torch.Size], optional): _description_. Defaults to None.
            active_dims (Optional[Tuple[int, ...]], optional): _description_. Defaults to None.
        """
        super().__init__(batch_shape=batch_shape, active_dims=active_dims)
        self.register_parameter("raw_scale", Parameter(torch.zeros(self.batch_shape)))

        if scale_prior is not None:
            self.register_prior(
                "scale_prior",
                scale_prior,
                BiFiScalingKernel._get_scale,
                BiFiScalingKernel._set_scale,
            )

        if scale_constraint is None:
            scale_constraint = Positive()
        self.register_constraint("raw_scale", scale_constraint)

    @property
    def scale(self) -> Tensor:
        r"""Scale parameter :math:`\rho`"""
        return self.raw_scale_constraint.transform(self.raw_scale)

    @scale.setter
    def scale(self, value):
        self._set_scale(value)

    def _set_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_scale)
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))

    def _get_scale(self):
        return self.scale

    @property
    def var(self) -> Tensor:
        return self.scale.pow(2)

    def forward(self, i1, i2, **params):
        i1, i2 = i1.long(), i2.long()  # convert to integers

        # Match input batch shapes with internal parameters' batch shape
        batch_shape = torch.broadcast_shapes(
            i1.shape[:-2], i2.shape[:-2], self.batch_shape
        )
        i1 = i1.expand(batch_shape + i1.shape[-2:])
        i2 = i2.expand(batch_shape + i2.shape[-2:])

        # Compute kernel matrix
        scale = self.scale.view(*self.scale.shape, 1, 1)
        res = (1 + i1 * (scale - 1)) @ (1 + i2 * (scale - 1)).transpose(-1, -2)
        return res


class BiFiKernel(AdditiveKernel):
    r"""
    A bi-fidelity kernel for Kennedy & O'Hagan (2001) style GP models.

    We model functions as:

    .. math::

        f(x) = \rho h(x, \theta^*) + \varepsilon(x)

    where :math:`h` corresponds to a simulator/low-fidelity model, :math:`x` represents design points,
    :math:`\theta^*` is an unknown calibration parameter, and :math:`\varepsilon` models the error
    between simulations and observed outcomes. When we have access to both simulations :math:`h(x,\theta)` and
    real observations :math:`f(x)`, this function can be modelled as a single Gaussian process with an additional
    fidelity input parameter :math:`s`, as:

    .. math::

        g(x, \theta, s) = \begin{cases}
                            h(x, \theta), \quad &s = 0 \\
                            \rho h(x, \theta) + \varepsilon(x), \quad &s = 1
                        \end{cases}
    Low-fidelity inputs are assumed to have a fidelity parameter set to 0, while high-fidelity inputs are assumed to
    have fidelity 1. The corresponding kernel for the combined function :math:`g` is given by:

    .. math::

        k((x, \theta, s), (x',\theta', s')) := k_\rho(s, s') k_h((x, \theta), (x', \theta')) + s s' k_\varepsilon(x, x')

    where the base kernels for both the low-fidelity component :math:`k_h` and the error model :math:`k_\varepsilon`
    are by default constructed as a Matern kernel with smoothness parameter set to 2.5, the kernel :math:`k_\rho` is a
    :class:`BiFiScalingKernel`, which models the scaling of the simulator output in the high-fidelity outcomes.
    """

    def __init__(
        self,
        design_d: int,
        calibration_d: int,
        sim_kernel: Optional[Kernel] = None,
        err_kernel: Optional[Kernel] = None,
        scale_prior: Optional[Prior] = None,
        use_scale: bool = True,
        use_ard: bool = False,
        fidelity_coord: int = -1,
        nu: float = 2.5,
        batch_shape: Optional[torch.Size] = None,
    ):
        """Constructor arguments.

        Args:
            design_d (int): Number of coordinates corresponding to the design :math:`x` variables.
            calibration_d (int): Number of coordinates corresponding to the calibration :math:`\theta` variables, assumed to follow after the design variables.
            sim_kernel (Optional[Kernel], optional): Simulation kernel. If `None`, a Matern kernel will be applied. Defaults to None.
            err_kernel (Optional[Kernel], optional): Error kernel. If `None`, a Matern kernel will be applied. Defaults to None.
            scale_prior (Optional[Prior], optional): Prior for simulation outputs scaling parameter. Defaults to None.
            use_ard (bool, optional): If `True`, independent lengthscales are assigned to each input coordinate (except fidelity input). Defaults to False.
            fidelity_coord (int, optional): Coordinate corresponding to the fidelity parameter. Defaults to -1.
            nu (float, optional): Smoothness parameter for the default Matern kernel, used if `sim_kernel` or `err_kernel` are not provided. Defaults to 2.5.
        """
        assert design_d >= 0, "Must be a non-negative integer"
        assert calibration_d >= 0, "Must be a non-negative integer"
        input_d = design_d + calibration_d
        if fidelity_coord < 0:
            fidelity_coord = input_d + 1 + fidelity_coord
        sim_dims = tuple([i for i in range(input_d)])
        err_dims = tuple([i for i in range(design_d)])

        if sim_kernel is None:
            sim_kernel = get_matern_kernel_with_gamma_prior(
                ard_num_dims=input_d if use_ard else 1,
                active_dims=sim_dims,
                nu=nu,
                batch_shape=batch_shape,
            )

        if err_kernel is None:
            err_kernel = get_matern_kernel_with_gamma_prior(
                ard_num_dims=design_d if use_ard else 1,
                active_dims=err_dims,
                nu=nu,
                batch_shape=batch_shape,
            )

        if use_scale:
            if scale_prior is None:
                scale_prior = LogNormalPrior(1.0, 1.0)

            scaling_sim_kernel = BiFiScalingKernel(
                scale_prior=scale_prior,
                active_dims=[fidelity_coord],
                batch_shape=batch_shape,
            )
        else:
            scaling_sim_kernel = ConstantKernel(batch_shape=batch_shape)

        fid_kernel = BiFiIndexKernel(fidelity_coord=fidelity_coord)

        super().__init__(scaling_sim_kernel * sim_kernel, fid_kernel * err_kernel)

    @property
    def sim_kernel(self):
        """The simulation kernel."""
        return self.kernels[0].kernels[-1]

    @property
    def err_kernel(self):
        """The error kernel"""
        return self.kernels[1].kernels[-1]
