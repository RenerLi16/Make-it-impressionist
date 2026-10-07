"""CSV-backed many-reconstructions-to-one-painting dataset."""

from pathlib import Path
from torch.utils.data import Dataset
from impressionist.data.pairing import read_metadata
from impressionist.data.transforms import PairedTransform
from impressionist.utils.image import load_rgb


class PairedPaintingDataset(Dataset):
    """CSV image paths are relative to root, never relative to the CSV file."""

    def __init__(self, csv_path: str | Path, root: str | Path,
                 transform: PairedTransform | None = None, image_size: int = 128):
        self.root = Path(root).resolve()
        csv_path = Path(csv_path)
        self.rows = read_metadata(csv_path if csv_path.is_absolute() else self.root / csv_path)
        if not self.rows:
            raise ValueError(f"Dataset {csv_path} is empty. Prepare a nonempty split first.")
        self.transform = transform or PairedTransform(image_size)
        for row in self.rows:
            for key in ("input_path", "target_path"):
                if not (self.root / row[key]).is_file():
                    raise FileNotFoundError(f"Pair {row['pair_id']}: missing {key}: {self.root / row[key]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        source, target = self.transform(load_rgb(self.root / row["input_path"]),
                                        load_rgb(self.root / row["target_path"]))
        return {"input": source, "target": target, "painting_id": row["painting_id"],
                "pair_id": row["pair_id"], "artist": row.get("artist", ""), "title": row.get("title", "")}
