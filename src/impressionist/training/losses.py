"""The baseline objective is only mean squared noise-prediction error."""

import torch
from torch.nn import functional as F


def noise_prediction_loss(prediction: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    if prediction.shape != noise.shape:
        raise ValueError("Predicted and actual noise shapes must match.")
    return F.mse_loss(prediction, noise)
