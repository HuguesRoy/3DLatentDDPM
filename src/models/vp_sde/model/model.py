
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
        
        if self.loss_type == "edm_loss":
            f_pred = self.network(c_in * x_noise, c_noise.view(-1,))
            f_target = (x - c_skip * x_noise) / c_out
            loss_weight = self.schedule.loss_weight(sigma)

            mse = (f_pred - f_target).pow(2)
            loss = (loss_weight * c_out.pow(2) * mse).mean()


        if not torch.isfinite(loss):
            print(f"[LOSS] Non-finite loss encountered: {loss.item():.3e}")
            return torch.tensor(0.0, device=x.device)

        return loss
    
    @torch.no_grad()
    def drift(self, x, t):
        """
        EDM probability-flow ODE drift:
            dx/dsigma = -(d(x,sigma) - x) / sigma
        """

        sigma = self.schedule.sigma(t)
        sigma_derivative = self.schedule.sigma_derivative(t)
        scaling = self.schedule.scaling(t)
        scaling_derivative = self.schedule.scaling_derivative(t)


        # Get EDM coefficients
        c_skip = self.schedule.c_skip(sigma)
        c_out  = self.schedule.c_out(sigma)
        c_in   = self.schedule.c_in(sigma)
        c_noise = self.schedule.c_noise(sigma)

        # Predict residual f
        if self.loss_type == "edm_loss":

            f_pred = self.network(
                c_in * x / scaling,
                c_noise.view(
                    -1,
                ),
            )
            d_pred = c_skip * x / scaling + c_out * f_pred


        # Probability-flow ODE drift

        drift_1 = (sigma_derivative/sigma + scaling_derivative/scaling) * x
        drift_2 = - sigma_derivative * scaling * d_pred / sigma
        drift = drift_1 + drift_2

        return drift

    @torch.no_grad()
    def sample_ode(self, batch_size : int = 1, dims : tuple =  (1,16), steps=10, device="cuda"):
        """
        Deterministic EDM ODE sampler.

        Inputs:
            batch_size : batch size
            dim :
            steps : number of small ODE refinement steps

        Output:
            x0 : deterministic reconstruction through local ODE integration
        """
        data_ndim = len(dims)

        ratios = torch.linspace(0,1,steps, device=device)
        times = self.schedule.time_steps(ratios)
        sigmas = self.schedule.sigma(times)
        times_hat = self.schedule.sigma_inv(sigmas).view(steps,)

        times_hat = torch.cat([times_hat, torch.zeros_like(times_hat[:1])])
        sigma0 = self.schedule.sigma(times_hat[0])

        s0 = self.schedule.scaling(times_hat[0])
        x = torch.randn(batch_size, *dims, device=device) * sigma0 * s0


        for i in range(steps - 1):
            
            time_i = times_hat[i]  # shape: [1, 1, ..., 1]
            time_next = times_hat[i + 1]  # shape: [1, 1, ..., 1]

            dt = time_next - time_i          # [1, 1, ..., 1]
            drift = self.drift(x, time_i) 
            x = x + drift * dt               # Euler step
        return x
    
    @torch.no_grad()
    def sample_ode_from_xt(self, x_0, t_star, steps=4):
        """
        Deterministic EDM ODE sampler applied locally around (x_t, sigma).

        Inputs:
            x_t   : noisy input at noise level sigma (B,C,H,W)
            t_star : starting time [1.,0]
            steps : number of small ODE refinement steps

        Output:
            x0 : deterministic reconstruction through local ODE integration
        """

        device = x_0.device
        ratios = torch.linspace(t_star, 1, steps, device=device)
        times = self.schedule.time_steps(ratios)
        sigmas = self.schedule.sigma(times)
        times_hat = self.schedule.sigma_inv(sigmas).view(
            steps,
        )

        sigma0 = self.schedule.sigma(times_hat[0])

        s0 = self.schedule.scaling(times_hat[0])
        x = torch.randn_like(x_0) * sigma0 * s0 + x_0 * s0

        for i in range(steps - 1):
            time_i = times_hat[i]  # shape: [1, 1, ..., 1]
            time_next = times_hat[i + 1]  # shape: [1, 1, ..., 1]

            dt = time_next - time_i  # [1, 1, ..., 1]
            drift = self.drift(x, time_i)
            x = x + drift * dt  # Euler step
        return x
    
    @torch.no_grad()
    def sample_sde_from_xt(self, x_0, t_star, steps=4):
        """
        Stochastic EDM SDE sampler applied locally around (x_t, sigma).

        Inputs:
            x_t   : noisy input at noise level sigma (B,C,H,W)
            t_star : starting time [1.,0]
            steps : number of small ODE refinement steps

        Output:
            x0 : deterministic reconstruction through local ODE integration
        """

        device = x_0.device
        ratios = torch.linspace(t_star, 1, steps, device=device)
        times = self.schedule.time_steps(ratios)
        sigmas = self.schedule.sigma(times)
        times_hat = self.schedule.sigma_inv(sigmas).view(
            steps,
        )

        sigma0 = self.schedule.sigma(times_hat[0])

        s0 = self.schedule.scaling(times_hat[0])
        x = torch.randn_like(x_0) * sigma0 * s0 + x_0 * s0

        for i in range(steps - 1):
            time_i = times_hat[i]  # shape: [1, 1, ..., 1]
            time_next = times_hat[i + 1]  # shape: [1, 1, ..., 1]

            dt = time_next - time_i  # [1, 1, ..., 1]
            drift = self.drift(x, time_i)
            x = x + drift * dt  # Euler step
        return x
    
    def sample_from_x0_t(self, x_0, t_star, steps = 100, pfode = True):

        if pfode:
            x = self.sample_ode_from_xt(x_0, t_star, steps=steps)
        else:
            x = self.sample_sde_from_xt(x_0, t_star, steps=steps)
        
        return x
    