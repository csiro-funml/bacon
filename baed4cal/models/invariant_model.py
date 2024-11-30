from typing import Optional
import torch
import torch.nn as nn


class InvariantModel(nn.Module):
    # NOTE: Inspired by Deep Sets paper code implementation
    def __init__(
        self, encoder: nn.Module, emitter: nn.Module, encoding_d: Optional[int] = None
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.emitter = emitter
        self.encoding_d = encoding_d

    def forward(self, points: torch.Tensor):
        points = torch.atleast_2d(points)
        if points.shape[-2] == 0:  # special case for an empty set
            if self.encoding_d is None:
                raise ValueError(
                    "Encoding dimension must be provided to condition on empty set"
                )
            encoding = torch.zeros(1, self.encoding_d)
        else:
            encoding: torch.Tensor = self.encoder(points)
        out = self.emitter(encoding.sum(dim=-2))
        return out
