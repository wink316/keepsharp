from __future__ import annotations

import torch


class DDPMScheduler:
    def __init__(self, timesteps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        }

    def to(self, device: torch.device) -> "DDPMScheduler":
        self.register = {k: v.to(device) for k, v in self.register.items()}
        return self

    def _extract(self, name: str, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        value = self.register[name][t]
        return value.view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract("sqrt_alphas_cumprod", t, x0.shape) * x0
            + self._extract("sqrt_one_minus_alphas_cumprod", t, x0.shape) * noise
        )

    def ddim_step(
        self,
        xt: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> torch.Tensor:
        ac_t = self.register["alphas_cumprod"][t]
        ac_prev = self.register["alphas_cumprod"][t_prev] if t_prev >= 0 else xt.new_ones(())
        x0 = (xt - torch.sqrt(1.0 - ac_t) * noise_pred) / torch.sqrt(ac_t)
        x0 = x0.clamp(-1.0, 1.0)
        return torch.sqrt(ac_prev) * x0 + torch.sqrt(1.0 - ac_prev) * noise_pred
