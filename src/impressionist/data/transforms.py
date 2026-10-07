"""Apply geometry once to both images; perturb only the realistic input."""

import math
import random
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class PairedTransform:
    """Aspect-preserving resize/crop to square RGB tensors in [-1, 1].

    A crop scale controls the fraction of the short edge kept. Inputs must
    depict aligned scenes with matching aspect ratios (within 1%).
    """

    def __init__(self, image_size: int, training: bool = False, **options: float):
        self.size, self.training = image_size, training
        defaults = dict(horizontal_flip=0.0, rotation=0.0, crop_scale_min=1.0,
                        brightness=0.0, contrast=0.0, saturation=0.0, blur=0.0, noise=0.0)
        unknown = options.keys() - defaults.keys()
        if unknown:
            raise ValueError(f"Unknown augmentation options: {sorted(unknown)}")
        self.options = defaults | options
        if image_size < 1 or not 0 < self.options["crop_scale_min"] <= 1:
            raise ValueError("image_size must be positive and crop_scale_min in (0, 1].")
        if any(v < 0 for v in self.options.values()) or self.options["horizontal_flip"] > 1 or self.options["blur"] > 1:
            raise ValueError("Augmentations must be nonnegative; flip and blur are probabilities in [0, 1].")

    def __call__(self, source: Image.Image, target: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if abs((source.width / source.height) / (target.width / target.height) - 1) > 0.01:
            raise ValueError("Paired images must have matching aspect ratios and aligned content.")
        o = self.options
        scale = random.uniform(o["crop_scale_min"], 1.0) if self.training else 1.0
        resize_factor = self.size / (min(target.size) * scale)
        height, width = math.ceil(target.height * resize_factor), math.ceil(target.width * resize_factor)
        images = [TF.resize(i, [height, width], InterpolationMode.BILINEAR, antialias=True) for i in (source, target)]
        top = random.randint(0, height-self.size) if self.training else (height-self.size)//2
        left = random.randint(0, width-self.size) if self.training else (width-self.size)//2
        images = [TF.crop(i, top, left, self.size, self.size) for i in images]
        if self.training:
            if random.random() < o["horizontal_flip"]:
                images = [TF.hflip(i) for i in images]
            if o["rotation"]:
                angle = random.uniform(-o["rotation"], o["rotation"])
                images = [TF.rotate(i, angle, InterpolationMode.BILINEAR, fill=(127, 127, 127)) for i in images]
            for name, adjust in (("brightness", TF.adjust_brightness), ("contrast", TF.adjust_contrast), ("saturation", TF.adjust_saturation)):
                if o[name]:
                    images[0] = adjust(images[0], random.uniform(max(0, 1-o[name]), 1+o[name]))
            if random.random() < o["blur"]:
                images[0] = TF.gaussian_blur(images[0], 3, random.uniform(0.1, 1.0))
        source_tensor, target_tensor = [TF.to_tensor(i) for i in images]
        if self.training and o["noise"]:
            source_tensor = (source_tensor + torch.randn_like(source_tensor) * o["noise"]).clamp(0, 1)
        return source_tensor * 2 - 1, target_tensor * 2 - 1
