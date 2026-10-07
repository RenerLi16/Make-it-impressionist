"""Labeled visual grids for alignment checks and generated comparisons."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from impressionist.utils.image import to_unit_range


def save_comparison_grid(inputs: torch.Tensor, targets: torch.Tensor, path: str | Path,
                         generated: torch.Tensor | None = None,
                         labels: list[str] | None = None) -> None:
    columns = [("Realistic input", inputs), ("Target painting", targets)]
    if generated is not None:
        columns.insert(1, ("Generated output", generated))
    if len(inputs) == 0 or any(batch.shape != inputs.shape for _, batch in columns):
        raise ValueError("Grid batches must be nonempty and have matching shapes.")
    fig, axes = plt.subplots(len(inputs), len(columns), figsize=(3*len(columns), 3*len(inputs)), squeeze=False)
    for col, (title, batch) in enumerate(columns):
        for row, image in enumerate(to_unit_range(batch)):
            axes[row, col].imshow(image.permute(1, 2, 0).numpy())
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title)
            if col == 0 and labels:
                axes[row, col].set_ylabel(labels[row])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
