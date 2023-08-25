import torch

from ...util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()


class EDMUniformSampling:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.rho = rho
        min_inv_rho = sigma_min ** (1 / rho)
        self.max_inv_rho = sigma_max ** (1 / rho)
        self.d_inv_rho = min_inv_rho - self.max_inv_rho

    def __call__(self, n_samples, rand=None):
        r = default(rand, torch.rand((n_samples,)))
        sigmas = (self.max_inv_rho + r * self.d_inv_rho) ** self.rho
        return sigmas


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = default(
            rand,
            torch.randint(0, self.num_idx, (n_samples,)),
        )
        return self.idx_to_sigma(idx)
