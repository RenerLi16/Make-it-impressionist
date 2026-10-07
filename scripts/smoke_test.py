"""Exercise preparation, two optimizer updates with resume, sampling, and evaluation."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from _common import ROOT
from impressionist.data.pairing import assert_disjoint, read_metadata
from impressionist.models.diffusion import build_diffusion
from impressionist.training.checkpointing import read_checkpoint
from impressionist.utils.config import load_config, save_config
from impressionist.utils.device import select_device
from impressionist.utils.seed import seed_everything


def create_synthetic_data(root: Path) -> None:
    """Eight distinct synthetic targets and three input variants per target."""
    rng = np.random.default_rng(123)
    targets = root / "data/raw/paintings"
    targets.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:40, :48]
    for painting in range(8):
        pid = f"synthetic_{painting:03d}"
        folder = root / "data/raw/reconstructions" / pid
        folder.mkdir(parents=True)
        array = np.stack([(xx*5 + painting*29) % 256, (yy*6 + painting*41) % 256,
                          ((xx+yy)*3 + painting*17) % 256], axis=-1).astype(np.uint8)
        image = Image.fromarray(array)
        image.save(targets / f"{pid}.png")
        for variant in range(3):
            source = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
            source = source.filter(ImageFilter.GaussianBlur(0.2 * variant))
            source.save(folder / f"realistic_{variant:02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keep-data", action="store_true", help="Retain synthetic raw files and CSVs for inspection")
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_everything(123)
    name = f"smoke_{uuid.uuid4().hex[:8]}"
    output = ROOT / "outputs" / name
    output.mkdir(parents=True)
    temporary = None if args.keep_data else tempfile.TemporaryDirectory(prefix="impressionist-smoke-")
    data_root = output / "dataset" if args.keep_data else Path(temporary.name)
    commands: list[list[str]] = []
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MPLCONFIGDIR=str(output / ".matplotlib"))
    def run(script: str, *arguments: object) -> None:
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
        commands.append(command)
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    try:
        create_synthetic_data(data_root)
        run("prepare_dataset.py", "--root", data_root)
        splits = {s: read_metadata(data_root / "data/processed" / f"{s}.csv") for s in ("train", "val", "test")}
        assert_disjoint(splits)
        assert [len(splits[s]) for s in ("train", "val", "test")] == [18, 3, 3]
        config = load_config(ROOT / "configs/baseline_128.yaml")
        config["experiment"] = {"name": name, "seed": 123}
        config["data"].update(root=str(data_root), image_size=16)
        config["model"].update(base_channels=8, channel_multipliers=[1, 2], condition_channels=4)
        config["diffusion"]["timesteps"] = 8
        config["training"].update(batch_size=1, epochs=1, save_every=1, sample_every=1, num_samples=1)
        config_path = output / "config.yaml"
        save_config(config, config_path)
        device = select_device(args.device)
        model = build_diffusion(config).to(device)
        x = torch.zeros(1, 3, 16, 16, device=device)
        with torch.no_grad():
            predicted = model(x, torch.tensor([0], device=device), x)
        assert predicted.shape == x.shape and torch.isfinite(predicted).all()
        del model
        run("inspect_dataset.py", "--config", config_path, "--augment", "--output", output / "inspection.png", "--count", 3)
        common = ("--config", config_path, "--device", args.device, "--max-train-batches", 1, "--max-val-batches", 1)
        run("train.py", *common)
        checkpoint = ROOT / "checkpoints" / name / "latest.pt"
        state = read_checkpoint(checkpoint)
        assert state["epoch"] == 1 and state["global_step"] == 1 and state["optimizer_state"]["state"]
        run("train.py", *common, "--resume", checkpoint, "--epochs", 2)
        state = read_checkpoint(checkpoint)
        assert state["epoch"] == 2 and state["global_step"] == 2
        source = data_root / splits["test"][0]["input_path"]
        run("sample.py", "--checkpoint", checkpoint, "--input", source, "--output", output / "sample.png", "--seed", 123, "--device", args.device)
        with Image.open(output / "sample.png") as image:
            assert image.size == (16, 16)
        run("evaluate.py", "--checkpoint", checkpoint, "--output-dir", output / "evaluation", "--max-items", 1, "--device", args.device)
        metrics = json.loads((output / "evaluation/metrics.json").read_text())
        assert metrics["pairs"] == 1
        report = {"status": "passed", "device": str(device), "torch": str(torch.__version__),
                  "paintings": 8, "pairs": 24, "split_pairs": {s: len(rows) for s, rows in splits.items()},
                  "optimizer_updates": state["global_step"], "resume_checked": True,
                  "synthetic_data_retained": args.keep_data, "commands": commands,
                  "note": "Wiring test only: 16px, 8 diffusion steps, random synthetic data; output quality is meaningless."}
        (output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"SMOKE TEST PASSED: {output}")
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
