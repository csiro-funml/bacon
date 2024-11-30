import torch


class LocationFindingSimulator:
    """ This class simulates the location finding problem in Foster et al. (2021).

        References:
        - Foster, A., Ivanova, D. R., Malik, I., & Rainforth, T. (2021).
        Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design.
        In M. Meila & T. Zhang (Eds.),
        Proceedings of the 38th International Conference on Machine Learning
        (ICML 2021) (Vol. 139, pp. 3384-3395). PMLR.
    """

    def __init__(self,
                 n_sources: int, spatial_d: int, max_signal: float = 1e-4,
                 background_signal: float = 1e-1, alpha: float = 1.
                 ) -> None:
        self.n_sources = n_sources
        self.spatial_d = spatial_d
        self.max_signal = max_signal
        self.background_signal = background_signal
        self.alpha = alpha

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Noise-free evaluation

        Args:
            x (torch.Tensor): input composed of design/measurement location
            followed by source-location parameters, concatenated along the last axis

        Returns:
            torch.Tensor: log signal intensities for each input
        """
        measurement_location = x[..., :self.spatial_d]
        source_locations = torch.stack(
            x[..., self.spatial_d:].split(self.spatial_d, dim=-1),
            dim=0
        )
        sq_distances = (measurement_location - source_locations).pow(2).sum(-1)
        intensities = self.alpha / (self.max_signal + sq_distances)
        total_intensity = self.background_signal + intensities.sum(0)

        return total_intensity.log()
