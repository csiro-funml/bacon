from typing import Optional
import torch


def combine_with_parameters(
    designs: torch.Tensor,
    theta: torch.Tensor,
    combined_points: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine design points with calibration parameters, accounting for batch dimensions

    Args:
        designs (torch.Tensor): Design points with shape (N, D)
        theta (torch.Tensor): Calibration parameters with (possibly batched) shape (..., d)
        combined_points (Optional[torch.Tensor], optional): Combined design and calibration parameters with shape (M, D+d). Defaults to None.

    Returns:
        torch.Tensor: Combined design and calibration parameters with shape (..., N [+ M], D+d)
    """
    cond_x = torch.cat(
        [
            designs.expand(*theta.shape[:-1], *designs.shape),
            theta.unsqueeze(-2).expand(*theta.shape[:-1], *designs.shape[:-1], -1),
        ],
        -1,
    )
    if combined_points is not None:
        full_points = torch.cat(
            [
                cond_x,
                combined_points.expand(*cond_x.shape[:-2], *combined_points.shape),
            ],
            -2,
        )
    else:
        full_points = cond_x
    return full_points

