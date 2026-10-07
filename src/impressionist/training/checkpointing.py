"""Atomic tensor/state-dictionary checkpoints with epoch-boundary RNG restore."""

from pathlib import Path
import torch
from impressionist.models.diffusion import GaussianDiffusion, build_diffusion
from impressionist.utils.seed import capture_rng_state, restore_rng_state


def save_checkpoint(path: str | Path, diffusion: GaussianDiffusion,
                    optimizer: torch.optim.Optimizer, epoch: int, global_step: int,
                    config: dict, loader_generator: torch.Generator,
                    best_val_loss: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format_version": 1, "model_state": diffusion.state_dict(),
             "optimizer_state": optimizer.state_dict(), "epoch": epoch, "global_step": global_step,
             "config": config, "seed": config["experiment"]["seed"],
             "rng_state": capture_rng_state(), "loader_rng_state": loader_generator.get_state(),
             "best_val_loss": best_val_loss}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def read_checkpoint(path: str | Path) -> dict:
    """Load only tensor/primitive containers; do not load arbitrary model objects."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    for key in ("model_state", "optimizer_state", "epoch", "global_step", "config"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing {key}.")
    return checkpoint


def load_for_inference(path: str | Path, device: torch.device) -> tuple[GaussianDiffusion, dict]:
    checkpoint = read_checkpoint(path)
    model = build_diffusion(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint["config"]


def restore_training(checkpoint: dict, model: GaussianDiffusion, optimizer: torch.optim.Optimizer,
                     loader_generator: torch.Generator) -> None:
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    loader_generator.set_state(checkpoint["loader_rng_state"].cpu())
    restore_rng_state(checkpoint["rng_state"])
