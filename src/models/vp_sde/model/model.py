
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict

class SDE(nn.Module):

    def __init__(self,
                network : nn.Module,
                schedule,
                model_config):
        super().__init__()

        self._init_config(model_config)
        self.network = network
        self.schedule = schedule


    def _init_config(self,model_config):
        self.loss_type = model_config.loss_type

    def train_step(self, dict_ord):

        x = dict_ord["image"]

        sigma = self.schedule.sample_sigma(x).to(x.device)

        c_skip = self.schedule.c_skip(sigma)
        c_out = self.schedule.c_out(sigma)
        c_in = self.schedule.c_in(sigma)
        c_noise = self.schedule.c_noise(sigma)

        noise = torch.randn_like(x)

        x_noise = x + sigma * noise

        f_pred = self.network(c_in * x_noise, c_noise.view(-1,))

        
        if self.loss_type == "score":
            f_target = (x - c_skip * x_noise) / c_out
            loss_weight = self.schedule.loss_weight(sigma)

            reduce_dims = tuple(range(1, f_pred.ndim))
            mse = (f_pred - f_target).pow(2).mean(dim=reduce_dims)
            loss = (loss_weight * mse).mean()
        elif self.loss_type == "noise":
            reduce_dims = tuple(range(1, f_pred.ndim))
            mse = (f_pred - noise).pow(2).mean(dim=reduce_dims)
            loss = mse.mean()


        if not torch.isfinite(loss):
            print(f"[LOSS] Non-finite loss encountered: {loss.item():.3e}")
            return torch.tensor(0.0, device=x.device)

        return loss
    
    @torch.no_grad()
    def drift(self, x, sigma):
        """
        EDM probability-flow ODE drift:
            dx/dsigma = -(d(x,sigma) - x) / sigma
        """

        # Get EDM coefficients
        c_skip = self.schedule.c_skip(sigma)
        c_out  = self.schedule.c_out(sigma)
        c_in   = self.schedule.c_in(sigma)
        c_noise = self.schedule.c_noise(sigma)

        # Predict residual f
        print((c_in).shape)
        f_pred = self.network(c_in * x, c_noise.view(-1,))

        # EDM denoiser
        d_pred = c_skip * x + c_out * f_pred

        # Probability-flow ODE drift
        drift = (x - d_pred) / (sigma + 1e-6)

        return drift

    @torch.no_grad()
    def sample_ode(self, batch_size, H=128,W =128, sigma_min=0.002, sigma_max=80, steps=100):
        """
        Classical EDM probability-flow ODE sampler.
        Deterministic DDIM-like sampling.

        Returns:
            x0 : sampled images (N,C,H,W)
        """
        device = next(self.network.parameters()).device

        # Create noise schedule
        sigmas = []
        for t in torch.linspace(0, 1, steps, device=device):
            s = self.schedule.time_steps(t)
            sigmas.append(s)
        sigmas = torch.stack(sigmas).view(steps, 1, 1, 1, 1)  # [steps,1,1,1,1]

        # Initial sample x_T = sigma_max * N(0, I)
        x = sigmas[0] * torch.randn(batch_size, 1, H, W, device=device)

        # ODE solve: Euler integration
        for i in range(steps - 1):
            sigma_i = sigmas[i]
            sigma_next = sigmas[i + 1]
            d_sigma = sigma_next - sigma_i

            # EDM drift: dx/dsigma
            drift = self.drift(x, sigma_i)

            # Euler step
            x = x + drift * d_sigma

        return x
    
    @torch.no_grad()
    def sample_ode_from_xt(self, x_t, t_star, steps=4):
        """
        Deterministic EDM ODE sampler applied locally around (x_t, sigma).

        Inputs:
            x_t   : noisy input at noise level sigma (B,C,H,W)
            t_star : starting time [1.,0]
            steps : number of small ODE refinement steps

        Output:
            x0 : deterministic reconstruction through local ODE integration
        """

        device = x_t.device
        batch_size = x_t.shape[0]
        data_ndim = x_t.ndim - 1
        times = torch.linspace(t_star, 1., steps, device=x_t.device)
        sigma_steps = self.schedule.time_steps(times).view(steps, *([1] * (data_ndim)))

        x = x_t.clone()

        for i in range(steps - 1):

            sigma_i = sigma_steps[i]           # shape: [1, 1, ..., 1]
            sigma_next = sigma_steps[i + 1]    # shape: [1, 1, ..., 1]
            ds = sigma_next - sigma_i          # [1, 1, ..., 1]
            drift = self.drift(x, sigma_i)   # dx/dsigma
            x = x + drift * ds               # Euler step

        return x
    
    def sample_from_x0_t(self, x0, t_star, steps = 100, pfode = True):

        batch_size = x0.shape[0]
        data_ndim = x0.ndim - 1
        
        sigma_scalar = self.schedule.time_steps(t_star).to(x0.device)
        # [1, 1, ..., 1]
        sigma_base = sigma_scalar.view(1, *([1] * data_ndim))
        # [B, 1, ..., 1]
        sigma_t = sigma_base.expand(batch_size, *([1] * data_ndim))

        x_t = x0 + sigma_t * torch.randn_like(x0)

        if pfode:
            x  = self.sample_ode_from_xt(x_t, t_star, steps=steps)
        else:
            x  = self.sample_sde_from_xt(x_t, sigma_t, steps=steps)
        
        return x
    