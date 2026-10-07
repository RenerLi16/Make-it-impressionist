"""RGB loading and conversions between [-1, 1] model space and image space."""

from pathlib import Path
from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TF


def load_rgb(path: str | Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode image {path}: {exc}") from exc


def to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu().add(1).div(2).clamp(0, 1)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("save_image expects a [3, H, W] RGB tensor.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(to_unit_range(tensor)).save(path)
