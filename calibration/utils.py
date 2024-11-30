from typing import Optional
import torch


def combine_inputs(
    real_designs: torch.Tensor,
    theta: torch.Tensor,
    sim_inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine real design points with calibration parameters, accounting for batch dimensions

    Args:
        real_designs (torch.Tensor): Design points with shape (N, D)
        theta (torch.Tensor): Calibration parameters with (possibly batched) shape (..., d)
        sim_points (Optional[torch.Tensor], optional): Combined design and calibration parameters with shape (M, D+d).

    Returns:
        torch.Tensor: Combined design and calibration parameters with shape (..., N [+ M], D+d)
    """
    cond_x = torch.cat(
        [
            real_designs.expand(theta.shape[:-1] + real_designs.shape),
            theta.view(
                theta.shape[:-1] + (1,) * (real_designs.ndim - 1) + theta.shape[-1:]
            ).expand(theta.shape[:-1] + real_designs.shape[:-1] + theta.shape[-1:]),
        ],
        -1,
    )
    if sim_inputs is not None:
        sim_inputs = torch.cat(
            [sim_inputs, torch.zeros(*sim_inputs.shape[:-1], 1).to(sim_inputs)], -1
        )
        full_points = torch.cat(
            [
                cond_x,
                sim_inputs.expand(*cond_x.shape[:-2], *sim_inputs.shape).to(cond_x),
            ],
            -2,
        )
    else:
        full_points = cond_x
    return full_points


def combine_outputs(
    real_outcomes: torch.Tensor, sim_outcomes: Optional[torch.Tensor] = None
):
    if sim_outcomes is None:
        return real_outcomes
    else:
        return torch.cat([real_outcomes, sim_outcomes], -1)
