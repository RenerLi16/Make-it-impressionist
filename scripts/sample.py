"""Generate one painting from a realistic input and a trained checkpoint."""

import argparse
import json
from pathlib import Path
from _common import ROOT
from impressionist.data.transforms import PairedTransform
from impressionist.training.checkpointing import load_for_inference
from impressionist.utils.device import select_device
from impressionist.utils.image import load_rgb, save_image
from impressionist.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    seed_everything(args.seed)
    diffusion, config = load_for_inference(args.checkpoint, select_device(args.device))
    image = load_rgb(args.input)
    source, _ = PairedTransform(config["data"]["image_size"])(image, image)
    result = diffusion.sample(source[None].to(diffusion.betas.device), seed=args.seed)
    save_image(result[0], args.output)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()), "input": str(args.input.resolve()),
        "seed": args.seed, "image_size": config["data"]["image_size"],
        "timesteps": diffusion.timesteps, "device": str(diffusion.betas.device)}, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
