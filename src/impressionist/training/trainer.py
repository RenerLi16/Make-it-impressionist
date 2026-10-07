"""A compact float32 training loop with held-out loss and sample grids."""

import json
import math
from itertools import islice
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from impressionist.data.dataset import PairedPaintingDataset
from impressionist.data.pairing import assert_disjoint, read_metadata
from impressionist.data.transforms import PairedTransform
from impressionist.evaluation.visualization import save_comparison_grid
from impressionist.models.diffusion import build_diffusion
from impressionist.training.checkpointing import read_checkpoint, restore_training, save_checkpoint
from impressionist.training.losses import noise_prediction_loss
from impressionist.utils.config import project_path, save_config
from impressionist.utils.device import select_device
from impressionist.utils.seed import seed_everything, seed_worker


class Trainer:
    """Resume at epoch boundaries; logs use example-weighted mean MSE."""

    def __init__(self, config: dict, device: str = "auto", resume: str | Path | None = None):
        self.config, self.device = config, select_device(device)
        seed_everything(config["experiment"]["seed"])
        data, train = config["data"], config["training"]
        root = project_path(data["root"])
        self.train_data = PairedPaintingDataset(data["train_csv"], root,
            PairedTransform(data["image_size"], training=True, **config["augmentation"]))
        self.val_data = PairedPaintingDataset(data["val_csv"], root, image_size=data["image_size"])
        splits = {"train": self.train_data.rows, "val": self.val_data.rows}
        if data.get("test_csv") and (root / data["test_csv"]).exists():
            splits["test"] = read_metadata(root / data["test_csv"])
        # Normalize paths as well as checking IDs, including hand-authored CSVs.
        normalized = {name: [row | {"target_path": str((root / row["target_path"]).resolve())}
                             for row in rows] for name, rows in splits.items()}
        assert_disjoint(normalized)
        self.loader_generator = torch.Generator().manual_seed(config["experiment"]["seed"])
        self.val_generator = torch.Generator().manual_seed(config["experiment"]["seed"] + 1)
        loader_args = dict(batch_size=train["batch_size"], num_workers=train["num_workers"],
                           worker_init_fn=seed_worker, pin_memory=self.device.type == "cuda")
        self.train_loader = DataLoader(self.train_data, shuffle=True, generator=self.loader_generator, **loader_args)
        self.val_loader = DataLoader(self.val_data, shuffle=False, generator=self.val_generator, **loader_args)
        self.diffusion = build_diffusion(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=train["learning_rate"],
                                           weight_decay=train["weight_decay"])
        self.epoch, self.global_step, self.best_val_loss = 0, 0, math.inf
        name = config["experiment"]["name"]
        self.checkpoint_dir = project_path(f"checkpoints/{name}")
        self.output_dir = project_path(f"outputs/{name}")
        self.log_dir = project_path(f"logs/{name}")
        if resume:
            state = read_checkpoint(resume)
            for section in ("data", "augmentation", "model", "diffusion", "experiment"):
                if config[section] != state["config"][section]:
                    raise ValueError(f"Resume config differs in {section}; use the checkpoint's saved config.")
            for key in ("batch_size", "learning_rate", "weight_decay", "num_workers"):
                if train[key] != state["config"]["training"][key]:
                    raise ValueError(f"Resume requires unchanged training.{key}.")
            restore_training(state, self.diffusion, self.optimizer, self.loader_generator)
            self.epoch, self.global_step = state["epoch"], state["global_step"]
            self.best_val_loss = state["best_val_loss"]
            if train["epochs"] <= self.epoch:
                raise ValueError(f"Checkpoint completed epoch {self.epoch}; increase --epochs to continue.")
        elif self.checkpoint_dir.exists() and any(self.checkpoint_dir.glob("*.pt")):
            raise FileExistsError(f"Experiment {name} already has checkpoints. Use --resume or a new experiment.name.")
        for path in (self.checkpoint_dir, self.output_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        save_config(config, self.log_dir / "config.yaml")
        save_config(config, self.checkpoint_dir / "config.yaml")
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            pass
        else:
            self.writer = SummaryWriter(str(self.log_dir), purge_step=self.global_step if resume else None)
        count = sum(p.numel() for p in self.diffusion.parameters())
        print(f"Device: {self.device} | parameters: {count:,} | train pairs: {len(self.train_data)} | val pairs: {len(self.val_data)}")

    def train_epoch(self, max_batches: int | None = None) -> float:
        self.diffusion.train()
        total, count = 0.0, 0
        batches = islice(self.train_loader, max_batches) if max_batches is not None else self.train_loader
        progress = tqdm(batches, total=min(len(self.train_loader), max_batches or len(self.train_loader)), desc=f"Epoch {self.epoch+1}")
        for batch in progress:
            source, target = batch["input"].to(self.device), batch["target"].to(self.device)
            t = torch.randint(self.diffusion.timesteps, (target.shape[0],), device=self.device)
            noise = torch.randn_like(target)
            xt = self.diffusion.q_sample(target, t, noise)
            self.optimizer.zero_grad(set_to_none=True)
            loss = noise_prediction_loss(self.diffusion(xt, t, source), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.config["training"].get("grad_clip", 1.0), error_if_nonfinite=True)
            self.optimizer.step()
            self.global_step += 1
            total += loss.item() * len(source)
            count += len(source)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if self.writer:
                self.writer.add_scalar("loss/train_step", loss.item(), self.global_step)
        return total / count

    @torch.no_grad()
    def validate(self, max_batches: int | None = None) -> float:
        self.diffusion.eval()
        seed = self.config["experiment"]["seed"] + 10000
        generator = torch.Generator().manual_seed(seed)
        self.val_generator.manual_seed(seed)
        total, count = 0.0, 0
        batches = islice(self.val_loader, max_batches) if max_batches is not None else self.val_loader
        for batch in batches:
            source, target = batch["input"].to(self.device), batch["target"].to(self.device)
            t = torch.randint(self.diffusion.timesteps, (len(source),), generator=generator).to(self.device)
            noise = torch.randn(target.shape, generator=generator).to(self.device)
            loss = noise_prediction_loss(self.diffusion(self.diffusion.q_sample(target, t, noise), t, source), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss.")
            total += loss.item() * len(source)
            count += len(source)
        return total / count

    def generate_validation_samples(self) -> None:
        count = min(self.config["training"].get("num_samples", 4), len(self.val_data))
        # Select distinct paintings first so multiple reconstructions do not
        # fill every qualitative row with the same target.
        indices, seen = [], set()
        for index, row in enumerate(self.val_data.rows):
            if row["painting_id"] not in seen:
                seen.add(row["painting_id"])
                indices.append(index)
            if len(indices) == count:
                break
        items = [self.val_data[i] for i in indices]
        inputs = torch.stack([i["input"] for i in items])
        targets = torch.stack([i["target"] for i in items])
        generated = self.diffusion.sample(inputs.to(self.device), seed=self.config["experiment"]["seed"])
        save_comparison_grid(inputs, targets, self.output_dir / f"epoch_{self.epoch:04d}.png",
                             generated.cpu(), [i["painting_id"] for i in items])

    def save(self, name: str) -> None:
        save_checkpoint(self.checkpoint_dir / name, self.diffusion, self.optimizer,
                        self.epoch, self.global_step, self.config, self.loader_generator, self.best_val_loss)

    def fit(self, max_train_batches: int | None = None, max_val_batches: int | None = None) -> None:
        for value in (max_train_batches, max_val_batches):
            if value is not None and value < 1:
                raise ValueError("Batch limits must be positive.")
        training = self.config["training"]
        try:
            while self.epoch < training["epochs"]:
                train_loss = self.train_epoch(max_train_batches)
                val_loss = self.validate(max_val_batches)
                self.epoch += 1
                improved = val_loss < self.best_val_loss
                self.best_val_loss = min(self.best_val_loss, val_loss)
                if self.epoch % training["sample_every"] == 0:
                    self.generate_validation_samples()
                self.save("latest.pt")
                if improved:
                    self.save("best.pt")
                if self.epoch % training["save_every"] == 0:
                    self.save(f"epoch_{self.epoch:04d}.pt")
                record = dict(epoch=self.epoch, global_step=self.global_step, train_loss=train_loss,
                              val_loss=val_loss, max_train_batches=max_train_batches, max_val_batches=max_val_batches)
                with (self.log_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record))
                if self.writer:
                    self.writer.add_scalar("loss/train_epoch", train_loss, self.global_step)
                    self.writer.add_scalar("loss/val", val_loss, self.global_step)
                    self.writer.flush()
        finally:
            if self.writer:
                self.writer.close()
