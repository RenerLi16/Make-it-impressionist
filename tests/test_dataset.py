import tempfile
import unittest
from pathlib import Path
import _helpers
import torch
from impressionist.data.dataset import PairedPaintingDataset
from impressionist.data.pairing import assert_disjoint, scan_pairs, split_by_painting, write_metadata
from impressionist.data.transforms import PairedTransform
from impressionist.utils.image import load_rgb
from impressionist.utils.seed import seed_everything


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        _helpers.make_raw(self.root)
        self.rows = scan_pairs(self.root, self.root / "data/raw/paintings", self.root / "data/raw/reconstructions")
        write_metadata(self.rows, self.root / "metadata.csv")

    def test_dataset_shapes_and_many_to_one_targets(self):
        dataset = PairedPaintingDataset("metadata.csv", self.root, image_size=16)
        self.assertEqual(len(dataset), 24)
        self.assertEqual(dataset[0]["input"].shape, (3, 16, 16))
        self.assertTrue(torch.equal(dataset[0]["input"], dataset[0]["target"]))
        self.assertEqual(dataset[0]["painting_id"], dataset[1]["painting_id"])
        self.assertTrue(torch.equal(dataset[0]["target"], dataset[1]["target"]))
        self.assertTrue(-1 <= dataset[0]["target"].min() <= dataset[0]["target"].max() <= 1)

    def test_geometry_preserves_alignment_across_seeds(self):
        image = load_rgb(self.root / self.rows[0]["target_path"])
        transform = PairedTransform(16, training=True, crop_scale_min=0.6, horizontal_flip=0.5, rotation=8)
        for seed in range(10):
            seed_everything(seed)
            source, target = transform(image, image)
            self.assertTrue(torch.equal(source, target))

    def test_photographic_augmentation_never_changes_target(self):
        image = load_rgb(self.root / self.rows[0]["target_path"])
        geometry = dict(crop_scale_min=0.7, horizontal_flip=0.5, rotation=5)
        seed_everything(2)
        clean_source, clean_target = PairedTransform(16, True, **geometry)(image, image)
        seed_everything(2)
        source, target = PairedTransform(16, True, **geometry, brightness=0.5, contrast=0.5,
                                         saturation=0.5, blur=1.0, noise=0.1)(image, image)
        self.assertTrue(torch.equal(target, clean_target))
        self.assertFalse(torch.equal(source, clean_source))

    def test_split_disjoint_complete_and_reproducible(self):
        splits = split_by_painting(self.rows)
        self.assertEqual(splits, split_by_painting(self.rows))
        self.assertEqual(sum(map(len, splits.values())), 24)
        self.assertEqual([len(splits[s]) for s in ("train", "val", "test")], [18, 3, 3])
        ids = {s: {r["painting_id"] for r in rows} for s, rows in splits.items()}
        self.assertFalse(ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
        for pid in {r["painting_id"] for r in self.rows}:
            self.assertEqual(sum(pid in subset for subset in ids.values()), 1)

    def test_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint({"train": self.rows[:1], "val": self.rows[:1]})

    def test_too_few_paintings_rejected(self):
        with self.assertRaisesRegex(ValueError, "Too few"):
            split_by_painting(self.rows[:3])

    def test_duplicate_target_pixels_rejected(self):
        image = load_rgb(self.root / self.rows[0]["target_path"])
        image.save(self.root / "data/raw/paintings/painting_001.png")
        with self.assertRaisesRegex(ValueError, "Identical target"):
            scan_pairs(self.root, self.root / "data/raw/paintings", self.root / "data/raw/reconstructions")

    def test_missing_file_and_bad_aspect_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "aspect ratios"):
            image = load_rgb(self.root / self.rows[0]["target_path"])
            PairedTransform(16)(image.resize((24, 24)), image)
        (self.root / self.rows[0]["input_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing input_path"):
            PairedPaintingDataset("metadata.csv", self.root)


if __name__ == "__main__":
    unittest.main()
