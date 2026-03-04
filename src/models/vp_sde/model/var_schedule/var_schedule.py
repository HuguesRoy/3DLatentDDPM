import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union

class NoiseSchedule(ABC):
    """
    Abstract interface for continuous-time noise schedules.

    A schedule maps a time parameter t (or step) to a noise level sigma,
    and defines how sigma is sampled during training.

    Any diffusion / score model can depend on this interface
    without caring whether the schedule is EDM, VP, VE, etc.
    """

    @abstractmethod
    def sigma(self, t: torch.Tensor | float) -> torch.Tensor:
        """
        Noise level

        Parameters
        ----------
        t : Tensor or float
            Continuous time parameter (in [0, 1]).

        Returns
        -------
        Tensor
            Noise scale, broadcastable to data shape.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample sigma for training (usually per batch element).

        Parameters
        ----------
        x : Tensor
            Input data tensor (used only for shape/device).

        Returns
        -------
        Tensor
            Sampled σ with shape (B, 1, ..., 1).
        """
        raise NotImplementedError


class NoiseScheduleEDM(NoiseSchedule):
    """
    Abstract EDM-style schedule interface.

    Subclasses must define how σ behaves and how
    EDM reparameterization coefficients are computed.
    """

    @abstractmethod
    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EDMSchedule(NoiseSchedule):
    """
    Karras-style EDM power schedule:

        sigma(t) = [ sigma_max^{1/rho} + t (sigma_min^{1/rho} - sigma_max^{1/rho}) ]^rho,  t ∈ [0, 1]

    Additionally, training-time sigma is sampled log-normally:

        log sigma ~ N(P_mean, P_std^2)
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ) -> None:
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
    
    # Sampling 

    def time_steps(self, step : Union[torch.Tensor, float]) -> torch.Tensor:
        # can be scalar or tensor (in [0,1])
        t_tensor = torch.as_tensor(step, dtype=torch.float32)
        # broadcast-friendly powers:
        smax_rho = self.sigma_max ** (1.0 / self.rho)
        smin_rho = self.sigma_min ** (1.0 / self.rho)
        return (smax_rho + t_tensor * (smin_rho - smax_rho)) ** self.rho

    # Training

    def sample_sigma(self, x):
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.randn(shape, device=x.device)
        return torch.exp(self.P_mean + self.P_std*noise)  # return only sigma

    def loss_weight(self, sigma: torch.Tensor):
        return (self.sigma_data**2 + sigma**2) / (sigma * self.sigma_data) ** 2

    # Network and preconditioning

    def c_skip(self, sigma: torch.Tensor):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return (sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return self.mean_factor(sigma)

    def c_noise(self,sigma):
        return 0.25 * torch.log(sigma) 

import torch
from typing import Union


class VPLinearSchedule:
    """
    EDM "VP" schedule used in NVlabs EDM ablation sampler.

    sigma(t) = sqrt(exp(0.5*beta_d*t^2 + beta_min*t) - 1)
    s(t)     = 1 / sqrt(1 + sigma(t)^2)

    beta_d and beta_min are typically chosen to match (sigma_min, sigma_max)
    over t in [epsilon_s, 1].
    """

    def __init__(
        self,
        beta_min: float,
        beta_max: float,
        epsilon: float = 1e-3,
        M: int = 1000,
    ):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.epsilon_s = float(epsilon)
        self.M = int(M)

        beta_d = self.beta_max - self.beta_min

        self.beta_d = float(beta_d)
    
    def time_steps(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
        return 1 + t * (self.epsilon_s -1)

    def beta_t(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
        t = torch.as_tensor(t)
        return self.beta_min + self.beta_d * t

    def sigma(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
        t = torch.as_tensor(t)
        exponent = 0.5 * self.beta_d * t**2 + self.beta_min * t
        return torch.sqrt(torch.clamp(torch.exp(exponent) - 1.0, min=1e-20))

    def sigma_derivative(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
        t = torch.as_tensor(t)
        sigma = self.sigma(t)
        return 0.5 * self.beta_t(t) * (sigma + 1.0 / sigma)

    def sigma_inv(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(sigma)
        disc = self.beta_min**2 + 2.0 * self.beta_d * torch.log(
            torch.clamp(sigma**2 + 1.0, min=1.0)
        )
        t = (torch.sqrt(disc) - self.beta_min) / self.beta_d
        return t.clamp(0.0, 1.0)

    def scaling(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
        t = torch.as_tensor(t)
        sigma = self.sigma(t)
        return 1.0 / torch.sqrt(1.0 + sigma**2)

    def scaling_derivative(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
        t = torch.as_tensor(t)
        sigma = self.sigma(t)
        sigma_d = self.sigma_derivative(t)
        s = 1.0 / torch.sqrt(1.0 + sigma**2)
        return -sigma * sigma_d * (s**3)

    def sample_sigma(self, x: torch.Tensor) -> torch.Tensor:
        # sample t form Uniform(eps, 1)
        batch = x.shape[0]
        shape = (batch,) + (1,) * (x.ndim - 1)
        t = self.epsilon_s + (1.0 - self.epsilon_s) * torch.rand(
            shape, device=x.device, dtype=x.dtype
        )
        return self.sigma(t)

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(sigma)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return -sigma

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(1.0 + sigma**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return (self.M - 1) * self.sigma_inv(sigma)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / (sigma**2)
