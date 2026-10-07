"""Small deterministic CPU fixtures without external test dependencies."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
torch.set_num_threads(1)


def make_raw(root: Path, paintings: int = 8, reconstructions: int = 3) -> None:
    generator = np.random.default_rng(7)
    targets = root / "data/raw/paintings"
    targets.mkdir(parents=True)
    for index in range(paintings):
        pid = f"painting_{index:03d}"
        folder = root / "data/raw/reconstructions" / pid
        folder.mkdir(parents=True)
        array = generator.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        image.save(targets / f"{pid}.png")
        for reconstruction in range(reconstructions):
            image.save(folder / f"realistic_{reconstruction:02d}.png")
