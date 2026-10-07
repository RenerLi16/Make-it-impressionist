"""DDPM forward noising and full ancestral reverse sampling, implemented locally."""

import torch
from torch import nn
from tqdm.auto import tqdm
from impressionist.models.conditional_unet import ConditionalUNet


class GaussianDiffusion(nn.Module):
    """Linear beta schedule and epsilon-prediction DDPM.

    Index 0 denotes the first noising step. Its reverse posterior has zero
    variance. Sampling visits every trained timestep; there is no step skipping.
    """

    def __init__(self, model: ConditionalUNet, timesteps: int = 1000,
                 beta_start: float = 0.0001, beta_end: float = 0.02):
        super().__init__()
        if not isinstance(timesteps, int) or timesteps < 2 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("Require timesteps >= 2 and 0 < beta_start <= beta_end < 1.")
        self.model, self.timesteps = model, timesteps
        # Compute schedules in float64 on CPU, store float32 for MPS support.
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
        values = dict(betas=betas, alphas=alphas, alpha_bars=alpha_bars,
                      sqrt_alpha_bars=alpha_bars.sqrt(), sqrt_one_minus_alpha_bars=(1-alpha_bars).sqrt(),
                      posterior_variance=betas * (1-previous) / (1-alpha_bars),
                      posterior_mean_x0=betas * previous.sqrt() / (1-alpha_bars),
                      posterior_mean_xt=(1-previous) * alphas.sqrt() / (1-alpha_bars))
        for name, value in values.items():
            self.register_buffer(name, value.float())

    def _validate(self, image: torch.Tensor, t: torch.Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Images must be floating-point [B,3,H,W] tensors.")
        if t.dtype != torch.long or t.shape != (image.shape[0],):
            raise ValueError("Timesteps must be int64 [B] tensors.")
        if t.device != image.device or image.device != self.betas.device:
            raise ValueError("Images, timesteps, and diffusion must be on the same device.")
        if (t < 0).any() or (t >= self.timesteps).any():
            raise ValueError(f"Timesteps must be in [0, {self.timesteps-1}].")

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return values[t][:, None, None, None]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon."""
        self._validate(x0, t)
        noise = torch.randn_like(x0) if noise is None else noise
        if noise.shape != x0.shape or noise.device != x0.device:
            raise ValueError("Noise must match x0 shape and device.")
        return self._extract(self.sqrt_alpha_bars, t) * x0 + self._extract(self.sqrt_one_minus_alpha_bars, t) * noise

    def forward(self, xt: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        self._validate(xt, t)
        return self.model(xt, t, condition)

    def p_mean_variance(self, xt: torch.Tensor, t: torch.Tensor,
                        condition_features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        predicted_noise = self.model(xt, t, condition_features=condition_features)
        x0 = ((xt - self._extract(self.sqrt_one_minus_alpha_bars, t) * predicted_noise)
              / self._extract(self.sqrt_alpha_bars, t)).clamp(-1, 1)
        mean = self._extract(self.posterior_mean_x0, t) * x0 + self._extract(self.posterior_mean_xt, t) * xt
        return mean, self._extract(self.posterior_variance, t)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, seed: int = 42, progress: bool = True) -> torch.Tensor:
        """Generate from Gaussian noise, caching the fixed condition features.

        Use a local CPU RNG and transfer draws to the model device. This avoids
        MPS generator limitations and does not consume training RNG state.
        """
        if condition.device != self.betas.device:
            raise ValueError("Condition and diffusion must be on the same device.")
        was_training = self.training
        self.eval()
        try:
            features = self.model.encode_condition(condition)
            shape = (condition.shape[0], 3, *condition.shape[-2:])
            generator = torch.Generator(device="cpu").manual_seed(seed)
            def draw() -> torch.Tensor:
                return torch.randn(shape, generator=generator, dtype=torch.float32).to(condition.device)
            xt = draw()
            for step in tqdm(range(self.timesteps - 1, -1, -1), total=self.timesteps, disable=not progress, desc="Sampling"):
                t = torch.full((shape[0],), step, device=xt.device, dtype=torch.long)
                mean, variance = self.p_mean_variance(xt, t, features)
                xt = mean + variance.sqrt() * draw() if step > 0 else mean
            return xt.clamp(-1, 1)
        finally:
            self.train(was_training)


def build_diffusion(config: dict) -> GaussianDiffusion:
    return GaussianDiffusion(ConditionalUNet(**config["model"]), **config["diffusion"])
