"""Explicit CPU, CUDA, and Apple Silicon device selection."""

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA, then MPS, then CPU; fail clearly for unavailable requests."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or auto.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Use --device cpu or auto.")
    return device
