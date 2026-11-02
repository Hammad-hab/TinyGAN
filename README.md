# TinyGAN

A 3D GAN built from scratch in PyTorch. It trains a generator to emit 32x32x32
voxel volumes of 3D objects (from ModelNet10) and renders the results as meshes
via PyVista. The project is now trained as a **Wasserstein GAN with Gradient
Penalty (WGAN-GP)** after the earlier objective proved unstable.

The repository is laid out as a proper Python package: `dsets/`, `models/`,
`proc/`, and `utils/` live under the root, and every source file imports each
other with absolute package paths (`from dsets.ModelNet10 import ...`). Run all
commands from the repository root so the root is on `sys.path`.

## What's in this repo

| Path | Role |
| ---- | ---- |
| `gan-train.py` | Entrypoint: wires `GAN` + `TrainProcesss` + `VersionManager` and starts the loop |
| `view.py` | CLI: voxel `.npz` -> mesh (marching cubes + smoothing) |
| `SampleGeneration.py` | One-shot sampling from the latest checkpoint -> `output.npz` |
| `dsets/ModelNet10.py` | PyTorch Geometric `ModelNet10` loader + `Voxelize` transform |
| `dsets/SolidWood.py` | Unused dataset stub (left from the parent repo) |
| `models/GAN.py` | `GAN` container: generator + critic, D/G loss, gradient penalty |
| `models/Generator.py` | 7-strided `ConvTranspose3d` generator: `latent_dim=100` -> `(1, 32, 32, 32)` |
| `models/Discriminator.py` | *Critic*: 5 strided `Conv3d` -> 1 logit (spectral norm **not** applied) |
| `models/UNET.py` | Unused stub (left from the parent repo) |
| `proc/GANTrainProcess.py` | WGAN-GP training loop + TensorBoard writers |
| `utils/VersionManager.py` | Checkpoint save / load (`versions/tiny-gan-<epoch>.pth`) |
| `utils/__init__.py` | `get_device()` (torch CUDA / MPS / CPU auto-detection) |

## Install

A `requirements.txt` is now shipped. Create a venv and install from it:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No GPU is required — `utils.get_device()` auto-detects CUDA and Apple MPS; CPU is
supported but slow. The archived run went to epoch 851.

## Quick start

### Train

```
python gan-train.py
```

Trains for 1000 epochs by default. Logs go to `runs/tiny-gan/` and checkpoints
to `versions/tiny-gan-<epoch>.pth`. Override the epoch count by editing the
`TrainProcesss` constructor in `gan-train.py` (currently `1000`).

### Sample

```
python SampleGeneration.py
```

Loads the latest checkpoint and writes `output.npz` (array key `voxels`,
shape `(1, 32, 32, 32)`).

### View

```
python view.py output.npz                 # interactive PyVista window
python view.py output.npz --save sample.ply   # headless export
```

## Training details

- Optimizer: Adam (`lr=1e-4`, `betas=(0.0, 0.9)`) for both critic and generator.
- Loss: `BCEWithLogitsLoss` on the critic logits.
- Critic/generator alternation: critic does **3 inner steps**, then 1 generator
  step, so 1 "epoch" = `len(loader)` critic-steps, each with 3 critic steps +
  1 generator step.
  - critic loss = `d_fake - d_real + 10 * gradient_penalty(real, fake)` (GP weight 10)
  - generator loss = `-D(G(z)).mean()`
  - reported Wasserstein estimate = `d_real - d_fake` (mean critic logits)
- LR schedulers: `CosineAnnealingLR` with `T_max=epochs`, critic `eta_min=1e-8`,
  generator `eta_min=1e-7`.
- Batch size: 32.

## Checkpoints

`VersionManager` saves `versions/tiny-gan-<epoch>.pth` (model + optimizer state)
with a failed-save protocol (`.f.pth`) so an interrupted epoch can be recomputed
on resume. `load_latest(True, True)` anywhere in the code picks the latest epoch,
skipping stray `.f` leftovers, and returns the epoch to resume from. The stale
`tiny-gan-True.pth` left behind by the failed-protocol path is ignored by
`load_latest`.

## Visualization pipeline (`view.py`)

`create_mesh(vol, level=None, sigma=1.0, smooth_iters=30, keep_largest=True)`:

1. auto iso-level: `0.0` if the volume has negatives, else `0.5`;
2. zero-pad the field by 2 so the surface closes at the borders;
3. optional Gaussian blur (`--sigma`);
4. `skimage.measure.marching_cubes`;
5. keep the largest connected component, drop floating specks;
6. Taubin smooth (`--smooth` iterations) and compute normals.

CLI options: `--key`, `--index`, `--level`, `--save`, `--sigma`, `--smooth`,
`--keep-all`.

## Project layout

```
TinyGAN/
├── README.md
├── requirements.txt
├── pyrightconfig.json
├── gan-train.py
├── view.py
├── SampleGeneration.py
├── dsets/
│   ├── __init__.py
│   ├── ModelNet10.py
│   └── SolidWood.py
├── models/
│   ├── __init__.py
│   ├── GAN.py
│   ├── Generator.py
│   ├── Discriminator.py
│   └── UNET.py
├── proc/
│   ├── __init__.py
│   └── GANTrainProcess.py
├── utils/
│   ├── __init__.py
│   └── VersionManager.py
├── texture/               # leftover files from an earlier project
│   ├── ForwardDiffusion.py
│   ├── Scheduler.py
│   ├── TrainProcess.py
│   └── UNET.py
├── bin/ lib/ include/ share/   # virtualenv
├── data/                        # ModelNet10 dataset (git-ignored)
├── runs/                        # TensorBoard logs (git-ignored)
└── versions/                    # checkpoints (git-ignored)
```

- `data/` — dataset (untracked in `.gitignore`).
- `runs/` — TensorBoard logs (untracked).
- `versions/` — checkpoints (untracked).
- `texture/` — leftover files from an earlier project (ForwardDiffusion.py,
  Scheduler.py, TrainProcess.py, UNET.py); untracked and unused by TinyGAN.

## Renamed / moved files

| Old path      | New path      |
| ------------- | ------------- |
| `util.py`     | `utils/__init__.py` (contains `get_device()`) |
| `Dataset.py`  | `dsets/ModelNet10.py` |
| `GAN.py`      | `models/GAN.py` |
| `Generator.py`| `models/Generator.py` |
| `Discriminator.py` | `models/Discriminator.py` |
| `TrainProcess.py` | `proc/GANTrainProcess.py` |
| `VersionManager.py` | `utils/VersionManager.py` |

The import statements in `gan-train.py`, `SampleGeneration.py`, and every
module were updated to match. If you copy any file out of the package, update
its imports accordingly.

## Known quirks & limitations

- **Discriminator spectral norm is not applied.** `spectral_norm` is imported in
  `models/Discriminator.py` but the conv layers are plain `nn.Conv3d` — the
  "add spectral norm" commit only added the import. Wrap the conv layers with
  `spectral_norm(...)` to enable it.
- **Generator has no final activation**, so output voxels are a soft real-valued
  density field (roughly `[-1, 2]` on the archived sample), not a hard 0/1 mask.
  Visualization relies on the iso-level and blur to turn it into a surface.
- **Iso-level can fall outside the data range after blur** for some samples.
  If `view.py` raises `Level fell outside range after blur`, lower `--sigma` or
  pass `--level` (on negative-value volumes, `0.0` works).
- `dsets/ModelNet10.py` imports `open3d` (used only in its `__main__` block);
  install it if you want the interactive dataset demo to run.
- Single-GPU, single-process (no DDP).
- `models/UNET.py` and `dsets/SolidWood.py` are leftover stubs; they are not
  imported by the training path.
- No `LICENSE` file is present; the repo is under the author's rights unless a
  license is added.

## License

See the note above. To add a license, place a `LICENSE` file at the repo root.
