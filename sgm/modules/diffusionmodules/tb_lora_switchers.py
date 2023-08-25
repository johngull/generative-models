from abc import abstractmethod, ABC

import torch


class TBLoRaSwitcher:
    def __init__(
            self,
            total_segments: int
    ):
        self.total_segments = total_segments
        self.segment = 0

    def segments_count(self) -> int:
        return self.total_segments

    def current_segment(self) -> int:
        return self.segment

    # @abstractmethod
    def select_segment(self, c_noise: torch.Tensor) -> None:
        raise NotImplementedError


    # def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    #     self.sigma_min = sigma_min
    #     self.sigma_max = sigma_max
    #     self.rho = rho
    #
    # def get_sigmas(self, n, device="cpu"):
    #     ramp = torch.linspace(0, 1, n, device=device)
    #     min_inv_rho = self.sigma_min ** (1 / self.rho)
    #     max_inv_rho = self.sigma_max ** (1 / self.rho)
    #     sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
    #     return sigmas


class EDMTBLoRaSwitcher(TBLoRaSwitcher):
    def __init__(
            self,
            total_segments: int = 1,
            sigma_min: float = 0.002,
            sigma_max: float = 80,
            rho=7.0
            # min_sigma: float = -6.2146,     # log(0.002)
            # max_sigma: float = 4.3820       # log(80)
    ):
        super().__init__(total_segments)

        self.min_inv_rho = sigma_min ** (1. / rho)
        self.max_inv_rho = sigma_max ** (1. / rho)
        self.d_inv_rho = self.min_inv_rho - self.max_inv_rho
        self.rho = rho

        # self.min_sigma = min_sigma
        # self.step = (max_sigma-min_sigma) / total_segments

    def select_segment(self, c_noise: torch.Tensor) -> None:

        sigmas = torch.exp(4 * c_noise)

        p = (sigmas ** (1./self.rho) - self.max_inv_rho) / self.d_inv_rho
        self.segment = torch.clip((p * self.total_segments).long(), 0, self.total_segments-1)

        # # sigma = torch.exp(4 * c_noise)
        # sigma = 4 * c_noise
        # self.segment = (sigma-self.min_sigma) / self.step
        # self.segment = torch.clip(self.segment.long(), 0, self.total_segments-1)

