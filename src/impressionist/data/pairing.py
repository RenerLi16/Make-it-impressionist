"""Raw file matching and leakage-safe painting-level splits."""

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Iterable
from impressionist.utils.image import load_rgb

FIELDS = ["pair_id", "painting_id", "input_path", "target_path", "artist", "title"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS[:4]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
        rows = list(reader)
    seen: set[str] = set()
    targets: dict[str, str] = {}
    target_owners: dict[str, str] = {}
    for row in rows:
        if any(not row.get(k, "").strip() for k in FIELDS[:4]):
            raise ValueError(f"{path}: required values cannot be empty: {row}")
        if row["pair_id"] in seen:
            raise ValueError(f"Duplicate pair_id: {row['pair_id']}")
        seen.add(row["pair_id"])
        pid, target = row["painting_id"], row["target_path"]
        if pid in targets and targets[pid] != target:
            raise ValueError(f"Painting {pid} has more than one target.")
        if target in target_owners and target_owners[target] != pid:
            raise ValueError(f"Target {target} is assigned to multiple painting IDs.")
        targets[pid], target_owners[target] = target, pid
        row.setdefault("artist", "")
        row.setdefault("title", "")
    return rows


def write_metadata(rows: Iterable[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def scan_pairs(root: Path, paintings: Path, reconstructions: Path,
               annotations: Path | None = None) -> list[dict[str, str]]:
    """Match paintings/<ID>.<ext> to reconstructions/<ID>/*.<ext>.

    IDs are stable target filename stems. Reject undecodable images, missing
    matches, duplicate target pixels, and aspect ratios differing by >1%.
    """
    root, paintings, reconstructions = root.resolve(), paintings.resolve(), reconstructions.resolve()
    if not paintings.is_dir() or not reconstructions.is_dir():
        raise ValueError("Raw paintings and reconstructions directories must exist. See data/README.md.")
    labels: dict[str, dict] = {}
    if annotations:
        with annotations.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "painting_id" not in (reader.fieldnames or []):
                raise ValueError("Annotations CSV requires painting_id.")
            for row in reader:
                if row["painting_id"] in labels:
                    raise ValueError(f"Duplicate annotation ID: {row['painting_id']}")
                labels[row["painting_id"]] = row
    targets: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for path in sorted(paintings.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        if path.stem in targets:
            raise ValueError(f"More than one target with ID {path.stem}")
        targets[path.stem] = path
    if not targets:
        raise ValueError("No target paintings found. Add images as documented in data/README.md.")
    orphan_ids = {p.name for p in reconstructions.iterdir() if p.is_dir() and not p.name.startswith('.')} - targets.keys()
    if orphan_ids:
        raise ValueError(f"Reconstruction folders have no painting: {sorted(orphan_ids)}")
    if any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in reconstructions.iterdir()):
        raise ValueError("Place reconstruction images in reconstructions/<painting_id>/ folders.")
    rows = []
    for pid, target in targets.items():
        target_image = load_rgb(target)
        digest = hashlib.sha256(str(target_image.size).encode() + target_image.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Identical target pixels under IDs {hashes[digest]} and {pid}; merge their reconstructions.")
        hashes[digest] = pid
        folder = reconstructions / pid
        inputs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if folder.is_dir() else []
        if not inputs:
            raise ValueError(f"No reconstructions found for painting {pid} in {folder}")
        for source in inputs:
            image = load_rgb(source)
            if abs((image.width / image.height) / (target_image.width / target_image.height) - 1) > 0.01:
                raise ValueError(f"Aspect ratio mismatch: {source} and {target}. Align pairs before preparation.")
            try:
                input_path, target_path = source.relative_to(root), target.relative_to(root)
            except ValueError as exc:
                raise ValueError("Raw data must be inside --root so metadata stays portable.") from exc
            rows.append(dict(pair_id=f"{len(rows)+1:06d}", painting_id=pid,
                             input_path=input_path.as_posix(), target_path=target_path.as_posix(),
                             artist=labels.get(pid, {}).get("artist", ""),
                             title=labels.get(pid, {}).get("title", "")))
    return rows


def assert_disjoint(splits: dict[str, list[dict[str, str]]]) -> None:
    """Reject painting IDs or literal target paths shared across partitions."""
    for key in ("painting_id", "target_path"):
        owners: dict[str, str] = {}
        for name, rows in splits.items():
            for row in rows:
                value = row[key]
                if value in owners and owners[value] != name:
                    raise ValueError(f"Split leakage: {key} {value!r} occurs in {owners[value]} and {name}")
                owners[value] = name


def split_by_painting(rows: list[dict[str, str]], val_fraction: float = 0.15,
                      test_fraction: float = 0.15, seed: int = 42) -> dict[str, list[dict[str, str]]]:
    """Shuffle unique painting IDs, then assign all their pairs together."""
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be nonnegative and sum to less than one.")
    ids = sorted({r["painting_id"] for r in rows})
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction)) if val_fraction else 0
    n_test = max(1, round(len(ids) * test_fraction)) if test_fraction else 0
    if len(ids) <= n_val + n_test:
        raise ValueError("Too few paintings for nonempty requested splits; add paintings or reduce fractions.")
    names = {pid: "val" for pid in ids[:n_val]}
    names.update({pid: "test" for pid in ids[n_val:n_val+n_test]})
    result: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        result[names.get(row["painting_id"], "train")].append(row)
    assert_disjoint(result)
    return result


def statistics(rows: list[dict[str, str]]) -> dict:
    counts = Counter(row["painting_id"] for row in rows)
    return {"pairs": len(rows), "paintings": len(counts),
            "reconstructions_per_painting": dict(sorted(counts.items()))}
