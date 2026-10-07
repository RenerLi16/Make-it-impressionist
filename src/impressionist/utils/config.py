"""YAML config with paths consistently resolved relative to the repository."""

from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    for section in ("experiment", "data", "augmentation", "model", "diffusion", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    name = config["experiment"].get("name", "")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("experiment.name must be a single directory name.")
    factor = 2 ** (len(config["model"]["channel_multipliers"]) - 1)
    size = config["data"]["image_size"]
    if not isinstance(size, int) or size < factor or size % factor:
        raise ValueError(f"data.image_size must be a positive multiple of {factor}.")
    train = config["training"]
    for key in ("batch_size", "epochs", "save_every", "sample_every"):
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer.")
    if train["num_workers"] < 0 or train["learning_rate"] <= 0 or train["weight_decay"] < 0:
        raise ValueError("Invalid worker count, learning rate, or weight decay.")
    if train.get("num_samples", 4) < 1 or train.get("grad_clip", 1.0) <= 0:
        raise ValueError("training.num_samples and grad_clip must be positive.")


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
