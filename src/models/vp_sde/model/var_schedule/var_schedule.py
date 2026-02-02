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
    def sample_t_sigma(self, x: torch.Tensor) -> torch.Tensor:
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

    def time_steps(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
        # can be scalar or tensor (in [0,1])
        t_tensor = torch.as_tensor(t, dtype=torch.float32)
        # broadcast-friendly powers:
        smax_rho = self.sigma_max ** (1.0 / self.rho)
        smin_rho = self.sigma_min ** (1.0 / self.rho)
        return (smax_rho + t_tensor * (smin_rho - smax_rho)) ** self.rho

    # Training

    def sample_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log-normal sigma sampling, one sigma per batch element.

        Shape: (B, 1, ..., 1) to broadcast over x.
        """
        batch = x.shape[0]
        # Make shape like (B, 1, ..., 1) with same ndims as x
        shape = (batch,) + (1,) * (x.ndim - 1)
        eps = torch.randn(shape, device=x.device, dtype=x.dtype)
        return torch.ones(shape, device= x.device, dtype = x.dtype), torch.exp(self.P_mean + self.P_std * eps)

    def loss_weight(self, sigma: torch.Tensor):
        return (self.sigma_data**2 + sigma**2) / (sigma * self.sigma_data) ** 2

    # Network and preconditioning

    def c_skip(self, sigma: torch.Tensor):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return (sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return self.mean_factor(sigma)

    def c_noise(self,sigma,t):
        return 0.25 * torch.log(sigma) 


class VPLinearSchedule(NoiseSchedule):
    """
    Variance-Preserving (VP) schedule with linear beta(t):
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 19.9,
        M : int = 1000,
        epsilon: float = 1e-5
    ) -> None:
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.epsilon = float(epsilon)
        self.M = int(M)

    def sigma(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
        # t in [0, 1]
        t_tensor = torch.as_tensor(t, dtype=torch.float32)

        beta_bar = (
            self.beta_min * t_tensor
            + 0.5 * (self.beta_max - self.beta_min) * t_tensor**2
        )
        alpha_bar = torch.exp(-beta_bar)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-20))
        return sigma

    def sample_t_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample t \sim Uniform(0, 1) per batch element and map to \sigma(t).

        Shape: (B, 1, ..., 1) to broadcast over x.
        """
        batch = x.shape[0]
        shape = (batch,) + (1,) * (x.ndim - 1)
        t = self.epsilon + (1.0 - self.epsilon) * torch.rand(
            shape, device=x.device, dtype=x.dtype
        )

        return t, self.sigma(t)
    
    def loss_weight(self, sigma: torch.Tensor):
        return 1/sigma**2

    # Network and preconditioning

    def c_skip(self, sigma: torch.Tensor):
        return torch.ones_like(sigma)

    def c_out(self, sigma: torch.Tensor):
        return - sigma

    def c_in(self, sigma: torch.Tensor):
        return 1/torch.sqrt(sigma**2 + 1.)

    def c_noise(self, sigma: torch.Tensor, t : torch.Tensor):
        return t * (self.M - 1)
