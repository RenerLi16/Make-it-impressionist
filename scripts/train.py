"""Train the conditional DDPM from scratch or resume an epoch checkpoint."""

import argparse
from _common import ROOT
from impressionist.training.trainer import Trainer
from impressionist.utils.config import load_config, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline_128.yaml"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, help="Total desired epochs, including completed epochs")
    parser.add_argument("--max-train-batches", type=int, help="Debug only: truncate each training epoch")
    parser.add_argument("--max-val-batches", type=int, help="Debug only: truncate each validation pass")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    validate_config(config)
    Trainer(config, args.device, args.resume).fit(args.max_train_batches, args.max_val_batches)


if __name__ == "__main__":
    main()
