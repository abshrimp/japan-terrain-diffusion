#!/usr/bin/env python3
"""EDM (Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based
Generative Models") preconditioning, loss, and Heun sampler.

D_theta(x; sigma, cond) = c_skip(sigma) x + c_out(sigma) F(c_in(sigma) x; c_noise(sigma), cond)
Training: sigma ~ LogNormal(P_mean, P_std); loss = lambda(sigma) ||D - y||^2.
Sampling: deterministic 2nd-order Heun on sigma schedule (sigma_max..sigma_min, rho).
"""
import numpy as np
import torch
import torch.nn as nn


class EDM(nn.Module):
    def __init__(self, net, sigma_data=0.5, p_mean=-1.2, p_std=1.2):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std

    def precond(self, x, sigma, cond=None):
        sd = self.sigma_data
        sigma = sigma.view(-1, 1, 1, 1)
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4.0
        F = self.net((c_in * x), c_noise, cond=cond)
        return c_skip * x + c_out * F

    def loss(self, y, cond=None):
        B = y.shape[0]
        rnd = torch.randn(B, device=y.device)
        sigma = (rnd * self.p_std + self.p_mean).exp()
        s = sigma.view(-1, 1, 1, 1)
        n = torch.randn_like(y) * s
        D = self.precond(y + n, sigma, cond=cond)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return (weight.view(-1, 1, 1, 1) * (D - y) ** 2).mean()

    @torch.no_grad()
    def sample(self, shape, device, cond=None, steps=32, sigma_min=0.002,
               sigma_max=80.0, rho=7.0, S_churn=0.0, S_min=0.0, S_max=float("inf"),
               S_noise=1.0, seed=None):
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(shape, device=device, generator=g) * sigma_max
        else:
            x = torch.randn(shape, device=device) * sigma_max
        i = torch.arange(steps, device=device)
        t = (sigma_max ** (1 / rho) + i / (steps - 1) *
             (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t = torch.cat([t, torch.zeros_like(t[:1])])  # t[N]=0
        for j in range(steps):
            s_cur, s_next = t[j], t[j + 1]
            gamma = (min(S_churn / steps, np.sqrt(2) - 1)
                     if S_min <= s_cur <= S_max else 0.0)
            s_hat = s_cur * (1 + gamma)
            if gamma > 0:
                x = x + (s_hat ** 2 - s_cur ** 2).sqrt() * S_noise * torch.randn_like(x)
            sig = torch.full((shape[0],), float(s_hat), device=device)
            d = self.precond(x, sig, cond=cond)
            dx = (x - d) / s_hat
            x_next = x + (s_next - s_hat) * dx
            if s_next > 0:  # 2nd-order correction
                sig2 = torch.full((shape[0],), float(s_next), device=device)
                d2 = self.precond(x_next, sig2, cond=cond)
                dx2 = (x_next - d2) / s_next
                x_next = x + (s_next - s_hat) * 0.5 * (dx + dx2)
            x = x_next
        return x


class EMA:
    """Exponential moving average of model params."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow
