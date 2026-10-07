"""Generate held-out outputs, per-pair metrics, painting-macro means, and a grid."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import torch
from tqdm.auto import tqdm
from _common import ROOT
from impressionist.data.dataset import PairedPaintingDataset
from impressionist.data.pairing import assert_disjoint, read_metadata
from impressionist.evaluation.metrics import image_metrics
from impressionist.evaluation.visualization import save_comparison_grid
from impressionist.training.checkpointing import load_for_inference
from impressionist.utils.config import project_path
from impressionist.utils.device import select_device
from impressionist.utils.image import save_image, to_unit_range
from impressionist.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", help="Defaults to checkpoint config's test CSV")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-items", type=int, help="Debug only: evaluate a prefix")
    args = parser.parse_args()
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive")
    seed_everything(args.seed)
    diffusion, config = load_for_inference(args.checkpoint, select_device(args.device))
    data = config["data"]
    csv_path = args.csv or data.get("test_csv")
    if not csv_path:
        parser.error("Provide --csv or data.test_csv in the checkpoint config")
    root = project_path(data["root"])
    dataset = PairedPaintingDataset(csv_path, root, image_size=data["image_size"])
    train_path = root / data["train_csv"]
    if not train_path.exists():
        raise FileNotFoundError(f"Training metadata is needed to check held-out evaluation: {train_path}")
    splits = {"train": read_metadata(train_path), "evaluation": dataset.rows}
    assert_disjoint({name: [r | {"target_path": str((root / r["target_path"]).resolve())} for r in rows]
                     for name, rows in splits.items()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, visual_items = [], []
    count = min(len(dataset), args.max_items or len(dataset))
    for index in tqdm(range(count), desc="Evaluating"):
        item = dataset[index]
        # Individual seeded samples remain reproducible independently of batch size.
        generated = diffusion.sample(item["input"][None].to(diffusion.betas.device), seed=args.seed + index, progress=False).cpu()
        values = image_metrics(to_unit_range(generated), to_unit_range(item["target"][None]))
        records.append({"pair_id": item["pair_id"], "painting_id": item["painting_id"], "seed": args.seed+index,
                        **{name: value.item() for name, value in values.items()}})
        save_image(generated[0], args.output_dir / "images" / f"{index:06d}.png")
        if len(visual_items) < 6:
            visual_items.append((item, generated[0]))
    with (args.output_dir / "per_pair.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    by_painting: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_painting[record["painting_id"]].append(record)
    keys = ("l1", "mse", "psnr", "ssim")
    def mean(rows: list[dict], key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)
    painting_means = [{key: mean(rows, key) for key in keys} for rows in by_painting.values()]
    def json_number(value: float) -> float | str:
        return value if math.isfinite(value) else str(value)
    summary = {"checkpoint": str(args.checkpoint.resolve()), "csv": str((root / csv_path).resolve()),
               "seed": args.seed, "pairs": len(records), "paintings": len(by_painting),
               "pair_mean": {key: json_number(mean(records, key)) for key in keys},
               "painting_macro_mean": {key: json_number(mean(painting_means, key)) for key in keys},
               "note": "RGB [0,1]; higher PSNR/SSIM and lower L1/MSE indicate reference fidelity, not artistic quality."}
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    save_comparison_grid(torch.stack([i["input"] for i, _ in visual_items]),
                         torch.stack([i["target"] for i, _ in visual_items]),
                         args.output_dir / "comparison.png", torch.stack([g for _, g in visual_items]),
                         [i["painting_id"] for i, _ in visual_items])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
