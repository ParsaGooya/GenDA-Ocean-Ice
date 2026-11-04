# appa2: regional ocean and ice GAE development

Copied from the current working copy of `../APPA`, including local edits. See
`UPSTREAM.txt` for the recorded upstream revision and `COPY_MANIFEST.json` for the
file inventory and extraction changes. The package is now named `appa2`, with separate Mac and Linux GPU environments.

## Included

- `experiments/autoencoder/train.py`: distributed training, validation plots,
  logging, gradient accumulation, checkpoints, resuming, and Slurm submission.
- `experiments/autoencoder/configs/`: training, graph architecture (`ico7to*`),
  optimizer, and retained convolutional architecture configurations.
- `appa2/nn/gae.py`, `graph.py`, `layers.py`, `triggers.py`, and `appa2/grid.py`:
  graph autoencoder and its supporting operations.
- `appa2/data/`, `date.py`, `losses.py`, `optim.py`, `soap.py`, `save.py`,
  `config/`, and `diagnostics/plots.py`: data loading, standardization, loss,
  optimization, checkpointing, configuration, and training visualization.
- `scripts/data/download_era5.py`, `data_stats.py`, and their configurations:
  source data download and mean/std preparation.
- `pyproject.toml`: Python package and dependency declarations. The `dev` extra
  includes development and plotting tools; `gpu` adds xformers, dawgz, and W&B.

Diffusion models, diffusion experiments, sampling, latent preparation for
diffusion, and unrelated diagnostics were excluded. Shared copied files can
still contain helpers/classes beyond the training path. The convolutional
model is retained because the architecture/checkpoint helpers import it.

## Changes from the source

- Training defaults to `ae: ico7to5` (GraphAE), instead of `cae_f32`.
- `PROJECT` resolves to this folder; the original used `/path/to/appa`.
- Removed the diffusion import and `load_denoiser` from `appa2/save.py`.
- Removed unused torch_harmonics/torchvision dependencies and explicitly
  declared cloudpickle/pandas. Other upstream version constraints are retained.

## Environments and configuration

Use Pixi >=0.83 on both machines. One `pixi.toml` defines two independent
environments that install the same local `appa2` source in editable mode:

| Environment | Platform | Package extras | Purpose |
|---|---|---|---|
| `mac-dev` | Apple Silicon macOS (`osx-arm64`) | `dev` | CPU development and checks |
| `gpu-dev` | x86-64 Linux with NVIDIA GPUs | `dev`, `gpu` | GPU development and HPC training |

- `pyproject.toml` defines package metadata, common dependencies, and optional
  dependency groups. The `gpu` group adds xformers, dawgz, and W&B.
- `pixi.toml` selects the platform, Python 3.11, and extras for each environment.
- `pixi.lock`, once generated, records resolved versions for reproducibility.

Editable installation means source changes take effect without reinstalling.
Changing dependencies requires updating the environment.

### Mac: CPU development

From the `HPE_ocean_ice_genDA` project directory:

```sh
pixi install -e mac-dev
pixi run -e mac-dev python -c "import appa2; print(appa2.__file__)"
pixi run -e mac-dev python -c "import torch; print(torch.ones(3, device='cpu'))"
```

To work in an activated shell:

```sh
pixi shell -e mac-dev
python -c "from appa2.grid import create_icosphere; print(create_icosphere(1)[0].shape)"
exit
```

Use this environment to inspect data, build small graphs, test CPU operations,
and develop code. Set tensors/models to `cpu` for these checks. Graph attention
falls back to PyTorch attention when xformers is absent. The original training
launcher remains CUDA/NCCL- and Slurm-specific; CPU training support has not
been added. Full-resolution graph construction can require substantial memory,
so use small graphs for local checks.

### HPC: GPU development and training

After pulling the repository, enter the `HPE_ocean_ice_genDA` project directory:

```sh
pixi install -e gpu-dev
pixi shell -e gpu-dev
```

The GPU environment assumes x86-64 Linux and declares CUDA 12.4 compatibility
for the retained PyTorch 2.5.1 stack. Confirm compatibility with the HPC's driver
and site module requirements before installing. Pixi does not install the
system NVIDIA driver or Slurm.

Within a GPU allocation, check GPU access:

```sh
pixi run -e gpu-dev python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A login node without an allocated GPU may report no CUDA device; perform the
GPU check on an allocated compute node according to your HPC's procedures.

Before training, configure:

- Dataset and statistics paths in `appa2/config/__init__.py`.
- `slurm_account` and `wandb_entity` in
  `experiments/autoencoder/configs/train.yaml`.
- W&B authentication and cluster resource arguments as required by your site.

From the activated `gpu-dev` shell, submit training:

```sh
cd experiments/autoencoder
python train.py --gpus 4 ae=ico7to5
```

This submits a Slurm job. Ensure the submitted job uses the `gpu-dev` environment
on the compute nodes; check the job environment against your site's Slurm setup.
The model defaults to the graph autoencoder. Outputs go under this project's
`autoencoders/` folder. Dataset paths and ERA5 assumptions still need adaptation
before training on RIOPS data.

### Git synchronization and dependency changes

Commit source code, `pixi.toml`, `pyproject.toml`, and `pixi.lock` when generated.
Keep `.pixi/` environments local; do not commit datasets, credentials, or large
training outputs. Git synchronizes configuration and code, not installed packages.

After pulling changes, run `pixi install -e mac-dev` on the Mac or
`pixi install -e gpu-dev` on the HPC to update that machine's environment.
Use environment names explicitly with `pixi install`, `pixi run`, and `pixi shell`.
The manifest also has Pixi's implicit default environment, which does not install
`appa2`; select one of the named environments above for project work.

Keep shared Python requirements in `pyproject.toml` and GPU-only requirements
in its `gpu` group. Use Pixi for environment/platform settings and avoid ad hoc
`pip install` changes that are not recorded in configuration.


## Cross-platform locking

Pixi resolves all environment/platform combinations when updating `pixi.lock`,
even when installing only `mac-dev`. Both named environments therefore support
a host-compatible variant on both platforms, so PyPI metadata builds can run
on either machine. `gpu-dev` selects `dev` extras on macOS and `dev` plus `gpu`
extras on Linux. Continue to use `mac-dev` on the Mac and `gpu-dev` on the HPC;
this does not install Linux GPU packages on the Mac.

## Verification status

Cross-platform dependency resolution succeeded and `pixi.lock` was generated.
The `mac-dev` environment installed successfully. Runtime checks passed for the
editable `appa2` import, a PyTorch CPU tensor operation, importing graph attention
without xformers, and creating a small 42-node icosphere. GPU installation and
training still need verification on the HPC.
