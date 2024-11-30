from torch import Tensor
import torch
from torch.utils.data import TensorDataset


class ObservationsDataset(TensorDataset):
    def __init__(self, points: Tensor, outcomes: Tensor) -> None:
        super().__init__(points, outcomes)

    @property
    def points(self) -> Tensor:
        return self.tensors[0]

    @property
    def outcomes(self) -> Tensor:
        return self.tensors[1]

    def __getitem__(self, index):
        xs, ys = super().__getitem__(index)
        return ObservationsDataset(points=xs, outcomes=ys)


def concatenate(*datasets: ObservationsDataset) -> ObservationsDataset:
    points = []
    outcomes = []
    for d in datasets:
        points.append(d.points)
        outcomes.append(d.outcomes)
    points = torch.cat(points, -2)
    outcomes = torch.cat(outcomes, -1)
    result = ObservationsDataset(points, outcomes)
    return result
