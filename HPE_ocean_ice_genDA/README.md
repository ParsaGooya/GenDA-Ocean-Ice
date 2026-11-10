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
  includes development and plotting tools; `gpu` adds dawgz and W&B, plus
  xformers on x86-64 Linux only. Castner uses the existing PyTorch graph-attention fallback.

Diffusion models, diffusion experiments, sampling, latent preparation for
diffusion, and unrelated diagnostics were excluded. Shared copied files can
still contain helpers/classes beyond the training path. The convolutional
model is retained because the architecture/checkpoint helpers import it.

## Changes from the source

- Training defaults to `ae: ico7to5` (GraphAE), instead of `cae_f32`.
- `PROJECT` resolves to this folder; the original used `/path/to/appa`.
- Removed the diffusion import and `load_denoiser` from `appa2/save.py`.
- Removed unused torch_harmonics/torchvision dependencies and explicitly
  declared cloudpickle/pandas. PyTorch remains 2.5.1 on macOS; ARM Linux uses 2.6.0 for its CUDA wheel.
  Other upstream version constraints are retained.

## Environments and configuration

Use Pixi >=0.83 on both machines. One `pixi.toml` defines two independent
environments that install the same local `appa2` source in editable mode:

| Environment | Platform | Package extras | Purpose |
|---|---|---|---|
| `mac-dev` | Apple Silicon macOS (`osx-arm64`) | `dev` | CPU development and checks |
| `gpu-dev-castner` | Castner ARM Linux with GH200 GPUs | `dev`, `gpu` | GPU development and HPC training |

- `pyproject.toml` defines package metadata, common dependencies, and optional
  dependency groups. The `gpu` group adds dawgz and W&B; its legacy
  xformers dependency is restricted to x86-64 Linux.
- `pixi.toml` selects the platform, Python 3.11, and extras for each environment.
- `pixi.lock`, once generated, records resolved versions for reproducibility.

Editable installation means source changes take effect without reinstalling.
Changing dependencies requires updating the environment.

### Mac: CPU development

From the `HPE_ocean_ice_genDA` project directory:

```sh
pixi install --locked -e mac-dev
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

#### Why the Castner environment exists

`gpu-dev-castner` is a compatibility configuration for Castner, not a
Castner-specific or performance-tuned software build. Castner reports an ARM
CPU (`aarch64`) and NVIDIA GH200 GPUs. The previous GPU target used
`linux-64` (Intel/AMD CPUs), which cannot run on this machine even though both
systems use NVIDIA GPUs. The new target therefore selects ARM Linux packages
and an explicitly CUDA-enabled PyTorch wheel.

The name identifies the system that motivated these choices; the packages
are standard upstream builds, not HPE-customized binaries. Other ARM Linux
systems may use the same environment if their operating-system libraries,
GPU hardware, and NVIDIA driver meet its requirements. It is not a generic
environment for every HPC system, and the Linux target does not support
Intel/AMD CPUs. Slurm partitions, account settings, GPU counts, and training
performance still need to be configured and validated separately on Castner.

After pulling the repository, enter the `HPE_ocean_ice_genDA` project directory:

```sh
pixi install --locked -e gpu-dev-castner
pixi shell -e gpu-dev-castner
```

The `linux-castner` target uses `linux-aarch64` and a CUDA 12.6 baseline.
ARM Linux explicitly installs the official Python 3.11 PyTorch 2.6.0+cu126
wheel rather than a CPU-only ARM wheel. macOS retains PyTorch 2.5.1.
The reported Castner driver, 580.65.06, supports this CUDA runtime; CUDA 13.0
shown by `nvidia-smi` is the driver's supported level, not an installed toolkit.
The wheel requires a compatible Linux runtime (PyTorch's ARM builds use
manylinux 2.28 / glibc 2.28 or newer). Pixi does not install the host driver or Slurm.

The pinned legacy xformers release is excluded on ARM. Graph attention already
has a PyTorch fallback; its performance and memory usage at full resolution
need verification on Castner before large training runs.

Within a GPU allocation, check GPU access:

```sh
pixi run -e gpu-dev-castner python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A login node without an allocated GPU may report no CUDA device; perform the
GPU check on an allocated compute node according to your HPC's procedures.

Before training, configure:

- Dataset and statistics paths in `appa2/config/__init__.py`.
- `slurm_account` and `wandb_entity` in
  `experiments/autoencoder/configs/train.yaml`.
- W&B authentication and cluster resource arguments as required by your site.

From the activated `gpu-dev-castner` shell, submit training:

```sh
cd experiments/autoencoder
python train.py --gpus 4 ae=ico7to5
```

This submits a Slurm job. Ensure the submitted job uses the `gpu-dev-castner` environment
on the compute nodes; check the job environment against your site's Slurm setup.
The model defaults to the graph autoencoder. Outputs go under this project's
`autoencoders/` folder. Dataset paths and ERA5 assumptions still need adaptation
before training on RIOPS data.

### Git synchronization and dependency changes

Commit source code, `pixi.toml`, `pyproject.toml`, and `pixi.lock` when generated.
Keep `.pixi/` environments local; do not commit datasets, credentials, or large
training outputs. Git synchronizes configuration and code, not installed packages.

After pulling changes, run `pixi install --locked -e mac-dev` on the Mac or
`pixi install --locked -e gpu-dev-castner` on the HPC to synchronize that machine's environment to the committed lockfile.
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
on either machine. `gpu-dev-castner` selects `dev` extras on macOS and `dev` plus `gpu`
extras on Linux. Continue to use `mac-dev` on the Mac and `gpu-dev-castner` on the HPC;
this does not install Linux GPU packages on the Mac.

## Verification status

The lockfile targets Apple Silicon and Castner ARM Linux.
The `mac-dev` environment installed successfully. Runtime checks passed for the
editable `appa2` import, a PyTorch CPU tensor operation, importing graph attention
without xformers, a small graph-attention fallback forward/backward operation,
and creating a small 42-node icosphere. Lockfile consistency and conda-forge-only
Conda package sources were also checked. These are local Mac and dependency
resolution checks, not Castner runtime validation. GPU installation and
training still need verification on Castner; the ARM wheel cannot run on the Mac.
