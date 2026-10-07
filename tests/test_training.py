import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import _helpers
import torch
from impressionist.data.pairing import scan_pairs, split_by_painting, write_metadata
from impressionist.training.checkpointing import load_for_inference
from impressionist.training.trainer import Trainer


class TrainingTests(unittest.TestCase):
    def test_epoch_resume_matches_uninterrupted_next_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _helpers.make_raw(root, paintings=4, reconstructions=1)
            rows = scan_pairs(root, root / "data/raw/paintings", root / "data/raw/reconstructions")
            for name, subset in split_by_painting(rows).items():
                write_metadata(subset, root / f"{name}.csv")
            config = {
                "experiment": {"name": "resume_test", "seed": 123},
                "data": {"root": str(root), "train_csv": "train.csv", "val_csv": "val.csv", "test_csv": "test.csv", "image_size": 16},
                "augmentation": {"horizontal_flip": 0.5, "noise": 0.02},
                "model": {"base_channels": 8, "channel_multipliers": [1, 2], "condition_channels": 4},
                "diffusion": {"timesteps": 4, "beta_start": 0.0001, "beta_end": 0.02},
                "training": {"batch_size": 1, "epochs": 2, "learning_rate": 0.001, "weight_decay": 0.0,
                             "num_workers": 0, "save_every": 1, "sample_every": 1, "num_samples": 1},
            }
            with patch("impressionist.training.trainer.project_path", side_effect=lambda p: root / p):
                trainer = Trainer(config, "cpu")
                self.addCleanup(lambda: trainer.writer.close() if trainer.writer else None)
                trainer.train_epoch()
                trainer.epoch = 1
                trainer.best_val_loss = trainer.validate()
                trainer.save("resume.pt")
                checkpoint = root / "checkpoints/resume_test/resume.pt"
                expected_loss = trainer.train_epoch()
                expected_state = {k: v.clone() for k, v in trainer.diffusion.state_dict().items()}
                resumed = Trainer(config, "cpu", checkpoint)
                self.addCleanup(lambda: resumed.writer.close() if resumed.writer else None)
                actual_loss = resumed.train_epoch()
                self.assertEqual(expected_loss, actual_loss)
                self.assertEqual(resumed.global_step, trainer.global_step)
                for key, expected in expected_state.items():
                    torch.testing.assert_close(resumed.diffusion.state_dict()[key], expected, rtol=0, atol=0)
                loaded, saved_config = load_for_inference(checkpoint, torch.device("cpu"))
                self.assertEqual(saved_config, config)
                self.assertFalse(loaded.training)


if __name__ == "__main__":
    unittest.main()
