
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
        self.loss_type = model_config.loss_types

    def train_step(self, dict_ord):

        x = dict_ord["image"]

        print(x.shape)
        t, sigma = self.schedule.sample_t_sigma(x)
        t = t.to(x.device)
        sigma = sigma.to(x.device)
        c_skip = self.schedule.c_skip(sigma)
        c_out = self.schedule.c_out(sigma)
        c_in = self.schedule.c_in(sigma)
        c_noise = self.schedule.c_noise(sigma, t)

        noise = torch.randn_like(x)

        x_noise = x + sigma * noise

        f_pred = self.network(c_in * x_noise, c_noise.view(-1,))

        

        if self.losse_type == "score":
            f_target = (x - c_skip * x_noise) / c_out
            loss_weight = self.schedule.loss_weight(sigma)

            reduce_dims = tuple(range(1, f_pred.ndim))
            mse = (f_pred - f_target).pow(2).mean(dim=reduce_dims)
            loss = (loss_weight * mse).mean()
        elif self.loss_type == "noise":
            reduce_dims = tuple(range(1, f_pred.ndim))
            mse = (noise - f_target).pow(2).mean(dim=reduce_dims)
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
    def sample_ode_from_xt(self, x_t, sigma, steps=4):
        """
        Deterministic EDM ODE sampler applied locally around (x_t, sigma).

        Inputs:
            x_t   : noisy input at noise level sigma (B,C,H,W)
            sigma : (B,1,1,1) noise level corresponding to t
            steps : number of small ODE refinement steps

        Output:
            x0 : deterministic reconstruction through local ODE integration
        """

        # Make a list of noise levels from sigma -> sigma_min
        # but only refine locally with small strides
        sigma_scalar = sigma.view(-1)[0].item()   # extract value
        sigma_min = self.schedule.sigma_min

        # For small steps, linearly interpolate σ
        sigma_steps = torch.linspace(
            sigma_scalar, sigma_min, steps, device=x_t.device
        ).view(-1, 1, 1, 1)

        x = x_t.clone()

        for i in range(steps - 1):

            sigma_i = sigma_steps[i].view(x.shape[0],1,1,1)
            sigma_next = sigma_steps[i+1].view(x.shape[0],1,1,1)
            ds = sigma_next - sigma_i

            drift = self.drift(x, sigma_i)   # dx/dsigma
            x = x + drift * ds               # Euler step

        return x




    