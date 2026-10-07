"""Reference fidelity metrics, computed per image in [0, 1] RGB space."""

import torch
from torch.nn import functional as F


def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return [B] L1/MSE/PSNR/SSIM; SSIM uses an 11px Gaussian window.

    Inputs must already be in [0,1]. SSIM uses valid windows, sigma=1.5,
    C1=0.01^2, C2=0.03^2, and averages RGB channels. Identical images have
    infinite PSNR. These metrics measure reference fidelity, not artistic merit.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Metrics expect matching [B,3,H,W] tensors.")
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("SSIM requires images at least 11x11.")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Metrics require finite images.")
    if prediction.min() < 0 or prediction.max() > 1 or target.min() < 0 or target.max() > 1:
        raise ValueError("Metrics expect values in [0,1].")
    error = prediction - target
    mse = error.square().mean((1, 2, 3))
    coordinates = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    gaussian = torch.exp(-coordinates.square() / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, 11, 11).contiguous()
    def average(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, groups=3)
    mean_x, mean_y = average(prediction), average(target)
    var_x = (average(prediction.square()) - mean_x.square()).clamp_min(0)
    var_y = (average(target.square()) - mean_y.square()).clamp_min(0)
    covariance = average(prediction * target) - mean_x * mean_y
    ssim = ((2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)
            / ((mean_x.square() + mean_y.square() + 0.01**2) * (var_x + var_y + 0.03**2)))
    return {"l1": error.abs().mean((1, 2, 3)), "mse": mse,
            "psnr": -10 * torch.log10(mse), "ssim": ssim.mean((1, 2, 3))}


# Future research hooks: add separate metrics with explicit input contracts for
# edge preservation, color distributions, spatial frequencies, brushstroke /
# texture statistics, and human ratings. No pretrained metric network is used.
