# Dataset contract

There are no real training images in this repository. Supply authorized source
paintings and realistic reconstructions depicting the same scene and composition.

## Raw layout

```text
data/raw/
├── paintings/
│   ├── monet_001.jpg
│   └── monet_002.png
└── reconstructions/
    ├── monet_001/
    │   ├── realistic_01.jpg
    │   ├── realistic_02.jpg
    │   └── realistic_03.jpg
    └── monet_002/
        └── realistic_01.png
```

The painting filename stem is its stable `painting_id`. A reconstruction folder
must have exactly that name. Each painting has exactly one target and at least
one reconstruction; file extensions may differ. Use globally unique IDs across
artists. The scanner is deliberately strict about unmatched folders, multiple
targets per ID, corrupt images, and identical decoded target pixels under
different IDs. It ignores hidden/non-image files. JPEG, PNG, WebP, BMP and TIFF
are accepted. EXIF orientation is applied before RGB conversion.

Images in each pair must have matching aspect ratios (within 1%). They can have
different resolutions; they must already be geometrically aligned. Preparation
does not register images or fix composition changes introduced by a generator.
The duplicate check finds identical decoded pixels, not re-encoded, cropped or
near-duplicate paintings. Curate those into one ID before splitting.

Optional annotations:

```csv
painting_id,artist,title
monet_001,Claude Monet,Example title
monet_002,Claude Monet,Another example
```

```bash
python scripts/prepare_dataset.py --annotations data/raw/annotations.csv
```

Without annotations, artist/title are blank. The script does not guess them.

## Processed layout

Preparation writes `metadata.csv`, `train.csv`, `val.csv`, `test.csv`, and
`statistics.json` to `data/processed/`. Images remain in raw storage; processed
files are manifests, avoiding redundant image copies and retaining originals.

```csv
pair_id,painting_id,input_path,target_path,artist,title
000001,monet_001,data/raw/reconstructions/monet_001/realistic_01.jpg,data/raw/paintings/monet_001.jpg,Claude Monet,Example title
000002,monet_001,data/raw/reconstructions/monet_001/realistic_02.jpg,data/raw/paintings/monet_001.jpg,Claude Monet,Example title
```

All image and CSV paths in the data configuration resolve against `data.root`.
The root itself resolves against the repository, unless absolute. The preparation
script's `--paintings`, `--reconstructions`, `--output-dir`, and `--annotations`
resolve against `--root`. Raw images must stay inside that root for portable CSVs.

## Splits and research hygiene

The default split allocates approximately 70%/15%/15% of **unique painting IDs**
to train/validation/test using seed 42. Fractions are rounded, with at least one
painting per requested nonzero split and at least one training painting. Too few
paintings is an error. Eight paintings become six/one/one paintings, regardless
of the number of reconstructions. Every reconstruction follows its painting ID.

Never split CSV rows independently. A target in both train and validation would
make validation partially a memorization test. Training rejects cross-split
painting IDs and shared target paths; evaluation also rejects overlap with train.
Keep validation for model choices and reserve test for final evaluation.

Store a copy/hash of final manifests for each scientific run. Re-preparing after
adding data can change assignments; freeze the corpus before comparing models.
The current sampler is uniform over pairs, so paintings with more reconstructions
receive more training weight. Evaluation reports both pair means and a painting
macro mean (mean within each painting, then mean across paintings).

## Augmentation and range

Training first resizes both images to identical dimensions without changing
aspect ratio, then applies the same crop coordinates, flip decision, and optional
rotation angle. Rotation fills exposed corners with neutral gray. Brightness,
contrast, saturation, blur and noise affect only the input. Noise standard
deviation is measured in [0,1] space. Disabling photographic augmentation is
useful when visually checking geometric alignment.

Validation, evaluation and sampling resize the short edge and take a deterministic
center square crop. This can discard peripheral scene content. All model tensors
are float32 RGB `[3,H,W]`, normalized to `[-1,1]`. Metrics use `[0,1]`.

```bash
python scripts/inspect_dataset.py
python scripts/inspect_dataset.py --augment --output outputs/augmented_pairs.png
```

Both commands write labeled grids that can be opened in an image viewer without
requiring a desktop plotting backend.

Raw data, generated manifests, model checkpoints, logs and outputs are ignored by
Git; directory placeholders and this documentation are tracked.
