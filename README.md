# Make It Impressionist

A small, readable research baseline for learning **realistic scene → Impressionist
painting** from paired images. All model weights are randomly initialized.
The condition encoder, U-Net, forward diffusion and reverse sampler are local
PyTorch implementations. There are no pretrained weights, Diffusers, LoRAs,
external image-generation APIs, or web UI.

## Scientific motivation and scope

This project asks whether paired supervision can teach a generative model to
preserve a scene's content while learning the color, simplification and texture
of Impressionist paintings. Multiple realistic reconstructions may supervise
the same painting. An external model may eventually help create those input
images, but it is not part of this repository's training or inference system.

The baseline learns a conditional distribution `p(painting | realistic input)`
through noise prediction. It does not learn a universal artistic-style filter by
definition: its behavior depends on the artists, scenes, alignment and artifacts
in the training corpus. Reconstruction-generator artifacts may become shortcuts;
generalization to real photographs must be measured separately.

## Setup

Python 3.11+ is required. From the repository:

```bash
cd "/Users/renerli/Documents/GitHub/Make it impressionist"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The code needs no editable package install: each script explicitly adds `src/`
to its import path. For interactive imports, use `PYTHONPATH=src python`.
For CUDA, choose the PyTorch/torchvision wheel pair recommended by the
[official PyTorch installer](https://pytorch.org/get-started/locally/) for your
driver/platform, then install the remaining requirements. `tensorboard` is
optional; install it with `python -m pip install tensorboard`.

## First execution: synthetic smoke test

```bash
python scripts/smoke_test.py --device cpu --keep-data
# Optional on Apple Silicon:
python scripts/smoke_test.py --device mps --keep-data
```

This generates **8 synthetic paintings × 3 reconstructions**, prepares metadata,
checks the 18/3/3 pair split (6/1/1 paintings), runs a model forward pass, writes
an augmented inspection grid, and performs **exactly two optimizer updates**:
one before checkpointing and one after resume. It then invokes the standalone
sampling and evaluation commands and checks their outputs. It uses 16×16 images,
base width 8, batch size 1, and eight diffusion steps; this only tests wiring.
The short schedule does not approach a Gaussian terminal marginal and is not
a meaningful training configuration. Generated image quality is meaningless.

Each run gets a unique `smoke_<id>` experiment. Reports, exact invoked commands,
images and the smoke config are under `outputs/smoke_<id>/`; checkpoints are in
`checkpoints/smoke_<id>/`; loss logs are in `logs/smoke_<id>/`. `--keep-data`
retains the synthetic raw files and CSVs under that output directory. Omit it to
use a temporary dataset removed at exit; that run's saved data paths then expire.
No real dataset is needed and the real raw directories are untouched.

## Data preparation

See [data/README.md](data/README.md) for the full contract. The expected layout is:

```text
data/raw/paintings/monet_001.jpg
data/raw/reconstructions/monet_001/realistic_01.jpg
data/raw/reconstructions/monet_001/realistic_02.jpg
```

```bash
python scripts/prepare_dataset.py --seed 42
python scripts/inspect_dataset.py
python scripts/inspect_dataset.py --augment --output outputs/augmented_pairs.png
```

Preparation validates every image and writes a metadata CSV plus train/val/test
CSVs. Multiple rows may share a target. Artist/title are optional metadata, never
model inputs in this baseline. Dataset items contain `input`, `target`,
`painting_id`, `pair_id`, `artist`, and `title`.

**Split by painting ID, never by pair.** Otherwise reconstructions of the same
target can enter different splits, leaking the target painting into evaluation.
The implementation shuffles unique painting IDs and assigns every associated row
together. It checks split disjointness again before training and evaluation.

Shared geometry consists of aspect-preserving resize, crop, optional horizontal
flip and optional small rotation. Photographic brightness/contrast/saturation,
blur and noise affect only the input. Evaluation uses deterministic center crops.

## Model and diffusion objective

Let `c` be the realistic conditioning image and `x₀` the target painting, both RGB
in `[-1,1]`. A linear schedule defines βₜ, αₜ = 1 − βₜ and
ᾱₜ = ∏ₛ₌₀ᵗ αₛ. Indices run from 0 to T−1; index 0 is the first noisy step.
For a uniformly drawn timestep and independent ε ~ N(0,I):

```math
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon
```

A small CNN extracts condition features at each U-Net resolution. The U-Net
concatenates these with noisy-target features along its down path. Residual
blocks receive sinusoidal timestep embeddings; skip connections feed the up path.
There is no attention or transformer. The RGB output predicts the added noise:

```math
\mathcal{L}(\theta) = \mathbb{E}_{x_0,c,t,\epsilon}
\left[\|\epsilon - \epsilon_\theta(x_t,t,c)\|_2^2\right]
```

The implementation averages the squared error over batch, channels and pixels.
For generation, start with Gaussian noise and visit all T reverse steps. Predict
and clip `x₀` to `[-1,1]`, then use the DDPM posterior mean and variance. The final
step adds no noise. Condition features are computed once for the sampling loop.
This follows the noise-prediction formulation in
[Ho et al., Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

The baseline uses **32 base channels**, condition width 16, multipliers
`[1,2,4,8]`, and 1,000 steps. A width of 64 is configurable, but 32 is the
conservative consumer-hardware starting point. The 128px config uses batch 4;
the 256px config uses batch 1. Reduce batch size first if memory is tight.
Spatial dimensions must be divisible by `2 ** (number_of_levels - 1)`.

## Training and resume

```bash
python scripts/train.py --config configs/baseline_128.yaml
# Explicit backend:
python scripts/train.py --config configs/baseline_128.yaml --device mps
# Resume to a total of 150 epochs:
python scripts/train.py --config configs/baseline_128.yaml \
  --resume checkpoints/baseline_128/latest.pt --epochs 150
```

Automatic device selection prefers CUDA, then MPS, then CPU. The implementation
uses float32, GroupNorm (supports batch size 1), AdamW, gradient clipping, and
`num_workers: 0` by default for straightforward macOS operation. Increasing
worker count is supported. There is no mixed precision, EMA or LR scheduler yet.

Training prints progress and logs example-weighted training/validation noise MSE
to `logs/<name>/metrics.jsonl`; TensorBoard is used if installed. Validation
timesteps and noise are fixed across epochs to reduce metric fluctuation.
Every configured sample interval produces `input | generated | target` grids
with a fixed seed. Full 1,000-step grids can be slow on consumer hardware.

`latest.pt` is saved every epoch, `best.pt` on improved validation loss, and
numbered checkpoints at `save_every`. Checkpoints include the diffusion/model
state (including schedule buffers), optimizer, completed epoch, global step,
config, seed, best validation loss, Python/NumPy/PyTorch/backend RNG states, and
the training DataLoader generator state. Configs are also saved as YAML beside
checkpoints and logs. Checkpoint writes use a temporary file then an atomic rename.
Loading uses tensor/state dictionaries and
[`torch.load(..., weights_only=True)`](https://docs.pytorch.org/docs/stable/generated/torch.load.html).

Resume occurs at an **epoch boundary**, not halfway through an epoch. Model,
data, augmentation, seed, batch size, worker count and optimizer settings must
match. Increase the total epoch count with `--epochs`; a new experiment name is
required for a fresh run once checkpoints exist. Data files must remain unchanged
for equivalent resume. Seeds do not promise bitwise equality across devices,
platforms or PyTorch versions. CPU resume equivalence is covered by a unit test.

`--max-train-batches 1 --max-val-batches 1` truncates epochs for debugging only;
these limits are recorded in the loss log. Do not use truncated runs for reported
research results. Optional TensorBoard: `tensorboard --logdir logs`.

## Sampling

```bash
python scripts/sample.py \
  --checkpoint checkpoints/baseline_128/best.pt \
  --input example.jpg \
  --output outputs/result.png \
  --seed 42
```

Architecture, image size and diffusion schedule come from the checkpoint.
Sampling resizes/center-crops the input to the configured square resolution and
saves a PNG plus a JSON sidecar containing seed, paths, steps and device. A local
CPU random generator supplies the noise draws on every backend, including MPS;
this does not disturb training RNG state. The seed is repeatable on the same
software/hardware setup. Sampling always uses the trained number of steps.

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/baseline_128/best.pt \
  --csv data/processed/test.csv \
  --output-dir outputs/test_evaluation --seed 42
```

The checkpoint's data root resolves the CSV and image paths. Evaluation requires
the training manifest to verify the evaluated painting IDs/targets are held out.
It saves generated images, a three-column grid, `per_pair.csv`, and `metrics.json`.
Metrics are pixel L1, MSE, PSNR and a local Gaussian-window SSIM in `[0,1]` RGB.
SSIM uses an 11×11 valid Gaussian window, sigma 1.5, and RGB-channel averaging;
images must be at least 11×11. Identical images have infinite PSNR (serialized as
`"inf"` in JSON). This explicit convention may differ from other SSIM packages.

Both pair-weighted and painting-macro summaries are provided. One seeded sample
is generated per pair, so uncertainty requires later repeated-seed experiments.
Pixel fidelity is not a complete measure of style or perceptual quality. No
pretrained CLIP/FID network is included. `metrics.py` leaves hooks for edge
preservation, color-distribution similarity, spatial frequencies, brushstroke /
texture statistics and human evaluation.

## Repository map

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── baseline_128.yaml
│   └── baseline_256.yaml
├── data/
│   ├── README.md
│   ├── raw/{paintings,reconstructions}/
│   └── processed/
├── checkpoints/
├── outputs/
├── logs/
├── scripts/
│   ├── _common.py
│   ├── prepare_dataset.py
│   ├── inspect_dataset.py
│   ├── train.py
│   ├── sample.py
│   ├── evaluate.py
│   └── smoke_test.py
├── src/impressionist/
│   ├── __init__.py
│   ├── data/{__init__,dataset,transforms,pairing}.py
│   ├── models/{__init__,blocks,embeddings,conditional_unet,diffusion}.py
│   ├── training/{__init__,trainer,losses,checkpointing}.py
│   ├── evaluation/{__init__,metrics,visualization}.py
│   └── utils/{__init__,config,device,image,seed}.py
└── tests/
    ├── _helpers.py
    ├── test_dataset.py
    ├── test_diffusion.py
    ├── test_model_shapes.py
    ├── test_metrics.py
    └── test_training.py
```

| Module | Responsibility |
|---|---|
| `data/pairing.py` | Raw validation, stable painting IDs, CSV I/O, group splits and leakage checks |
| `data/dataset.py` | Load many-to-one paired rows as tensors and metadata |
| `data/transforms.py` | Shared geometry and input-only photographic perturbations |
| `models/blocks.py` | GroupNorm residual blocks and independent condition feature pyramid |
| `models/embeddings.py` | Sinusoidal timestep embeddings |
| `models/conditional_unet.py` | Multiscale conditioning and RGB noise prediction |
| `models/diffusion.py` | Schedules, forward noising, posterior equations and ancestral sampling |
| `training/trainer.py` | Optimization, deterministic validation noise, logging and sample generation |
| `training/losses.py` | Explicit noise-prediction MSE |
| `training/checkpointing.py` | Atomic checkpoints, resume and inference reconstruction |
| `evaluation/metrics.py` | Dependency-light L1/MSE/PSNR/SSIM and future metric hooks |
| `evaluation/visualization.py` | Labeled pair/comparison grids |
| `utils/` | Config/path rules, backend choice, RGB conversions and RNG management |
| `scripts/` | Small command-line entry points plus the end-to-end synthetic test |
| `tests/` | Alignment, leakage, shapes, diffusion equations, conditioning gradients, metrics and exact CPU resume |

## Limitations and next milestones

This is an untrained pixel-space baseline. It needs enough diverse aligned pairs
to learn useful results; a smoke pass establishes plumbing, not scientific
validity. Full DDPM sampling is slow, square cropping discards scene edges, and
pair-weighted training favors paintings with more reconstructions. Exact-pixel
duplicate detection does not replace corpus-level near-duplicate review.
Dataset version hashes, artist-balanced sampling and multi-seed evaluation are
not automated yet. CPU/MPS/CUDA performance and reproducibility can differ.

1. **Establish the corpus and freeze splits.** Track provenance and usage rights,
   curate alignment and near duplicates, record manifest hashes, and audit
   reconstruction artifacts and artist coverage.
2. **Validate learning at 128px.** Deliberately overfit a small training subset,
   then run a held-out baseline with fixed settings and multiple seeds. Compare
   generated grids, input/target fidelity and failure cases before scaling up.
3. **Build controlled comparisons.** Add painting-balanced sampling and research
   metrics, then compare RGB versus RGB+edges/depth and augmentation strategies.
   Keep the same frozen painting splits, training budget and evaluation seeds.

The condition encoder is a separate module, and its configurable input-channel
count and multiscale feature interface leave room for future modalities. The
current dataset and scripts supply only RGB; adding edges/depth will require
explicit data-pipeline changes. Artist subsets, reconstruction counts and encoder
variants can be studied independently. Latent diffusion should be a separate
future experiment with an explicit autoencoder design and training policy.
