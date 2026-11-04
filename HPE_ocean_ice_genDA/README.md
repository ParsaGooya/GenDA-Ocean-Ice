# GAE training starting point

Copied from the current working copy of `../APPA`, including local edits. See
`UPSTREAM.txt` for the recorded upstream revision and `COPY_MANIFEST.json` for the
file inventory and extraction changes. Existing Pixi files were preserved.

## Included

- `experiments/autoencoder/train.py`: distributed training, validation plots,
  logging, gradient accumulation, checkpoints, resuming, and Slurm submission.
- `experiments/autoencoder/configs/`: training, graph architecture (`ico7to*`),
  optimizer, and retained convolutional architecture configurations.
- `appa/nn/gae.py`, `graph.py`, `layers.py`, `triggers.py`, and `appa/grid.py`:
  graph autoencoder and its supporting operations.
- `appa/data/`, `date.py`, `losses.py`, `optim.py`, `soap.py`, `save.py`,
  `config/`, and `diagnostics/plots.py`: data loading, standardization, loss,
  optimization, checkpointing, configuration, and training visualization.
- `scripts/data/download_era5.py`, `data_stats.py`, and their configurations:
  source data download and mean/std preparation.
- `pyproject.toml`: Python package and dependency declarations. The `all` extra
  includes the training launcher, plotting, and W&B dependencies.

Diffusion models, diffusion experiments, sampling, latent preparation for
diffusion, and unrelated diagnostics were excluded. Shared copied files can
still contain helpers/classes beyond the training path. The convolutional
model is retained because the architecture/checkpoint helpers import it.

## Changes from the source

- Training defaults to `ae: ico7to5` (GraphAE), instead of `cae_f32`.
- `PROJECT` resolves to this folder; the original used `/path/to/appa`.
- Removed the diffusion import and `load_denoiser` from `appa/save.py`.
- Removed unused torch_harmonics/torchvision dependencies and explicitly
  declared cloudpickle/pandas. Other upstream version constraints are retained.

## Setup and launch

Use a Linux CUDA environment with Slurm and compatible NVIDIA drivers. The
existing macOS-only `pixi.toml` has not been configured for this GPU workload.
The copied dependency versions have not been installation-tested here.

From this folder, install the package and training extras:

```sh
python -m pip install -e '.[all]'
```

Before launching, set the dataset/statistics paths in `appa/config/__init__.py`
and `slurm_account` / `wandb_entity` in the training configuration. Authenticate
W&B as needed. From the training directory, submit with:

```sh
cd experiments/autoencoder
python train.py --gpus 4 ae=ico7to5
```

This command submits a Slurm job; it is not a local CPU training command.
Outputs are written under this project's `autoencoders/` folder.

## Before adapting to ocean and ice

The copy preserves the ERA5 training behavior. It does not yet train directly
on your own data. Adapt these together:

1. Dataset reader, state/context variable names, pressure levels, and date
   splits (`appa/data/datasets.py`, `const.py`, and the training script).
2. Spatial grid and flattening order. `GraphAE` currently calls `create_N320()`
   in `appa/grid.py`, which actually constructs a fixed regular 721 x 1440
   latitude/longitude grid. Match graph coordinates to your input ordering.
3. Ocean/ice masks, missing-data treatment, loss weighting, and validation plots.
4. Statistics computed for your variables using your training split. The copied
   statistics configuration spans 1993–2021; change it to your training dates.
5. Cluster settings, paths, and suitable graph architecture sizes.

Keep `trajectory_size=1` while using the current validation subsampling:
`num_samples` selects timestamps before trajectories are formed.

## Verification

Python syntax, local import resolution, configuration references, and copy
integrity were checked without importing GPU dependencies. Training and runtime
imports were not executed: this environment lacks torch, Hydra, dawgz, xformers,
and W&B, and no project dataset or Slurm allocation was supplied.
