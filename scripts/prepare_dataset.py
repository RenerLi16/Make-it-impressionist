"""Validate raw pairs and write painting-disjoint CSV partitions."""

import argparse
import json
from pathlib import Path
from _common import ROOT
from impressionist.data.pairing import scan_pairs, split_by_painting, statistics, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--paintings", type=Path, default=Path("data/raw/paintings"))
    parser.add_argument("--reconstructions", type=Path, default=Path("data/raw/reconstructions"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--annotations", type=Path, help="Optional CSV with painting_id,artist,title")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = scan_pairs(root, root / args.paintings, root / args.reconstructions,
                      root / args.annotations if args.annotations else None)
    splits = split_by_painting(rows, args.val_fraction, args.test_fraction, args.seed)
    output = root / args.output_dir
    write_metadata(rows, output / "metadata.csv")
    for name, subset in splits.items():
        write_metadata(subset, output / f"{name}.csv")
    report = {"seed": args.seed, "val_fraction": args.val_fraction, "test_fraction": args.test_fraction,
              "all": statistics(rows), "splits": {name: statistics(part) for name, part in splits.items()}}
    (output / "statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, subset in {"all": rows, **splits}.items():
        stats = statistics(subset)
        print(f"{name:5s}: {stats['paintings']} paintings, {stats['pairs']} pairs")
    print(f"Wrote metadata to {output}; zero painting overlap across splits.")


if __name__ == "__main__":
    main()
