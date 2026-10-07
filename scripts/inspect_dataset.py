"""Save aligned realistic-input / painting rows, optionally with augmentation."""

import argparse
from pathlib import Path
import torch
from _common import ROOT
from impressionist.data.dataset import PairedPaintingDataset
from impressionist.data.transforms import PairedTransform
from impressionist.evaluation.visualization import save_comparison_grid
from impressionist.utils.config import load_config, project_path
from impressionist.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline_128.yaml"))
    parser.add_argument("--csv", help="Override CSV, relative to data.root")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dataset_inspection.png")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    seed_everything(args.seed)
    config = load_config(args.config)
    data = config["data"]
    dataset = PairedPaintingDataset(args.csv or data["train_csv"], project_path(data["root"]),
        PairedTransform(data["image_size"], training=args.augment, **config["augmentation"]))
    indices = torch.randperm(len(dataset))[:args.count].tolist()
    items = [dataset[index] for index in indices]
    save_comparison_grid(torch.stack([i["input"] for i in items]), torch.stack([i["target"] for i in items]),
                         args.output, labels=[i["painting_id"] for i in items])
    print(f"Saved {args.output} ({len(items)} pairs, augmentation={args.augment})")


if __name__ == "__main__":
    main()
