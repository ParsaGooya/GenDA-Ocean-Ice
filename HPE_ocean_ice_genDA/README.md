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
  includes development and plotting tools; `gpu` adds dawgz and W&B.
  The legacy xformers pin lives in a separate `legacy-xformers` extra that none
  of the supported environments select. Both HPC environments use the PyTorch graph-attention fallback.

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
  Protage x86-64 Linux uses PyTorch 2.9.1 for its ROCm wheel.
  Other upstream version constraints are retained.

## Environments and configuration

Use Pixi >=0.83 on all machines. One `pixi.toml` defines three independent
environments that install the same local `appa2` source in editable mode:

| Environment | Platform | Package extras | Purpose |
|---|---|---|---|
| `mac-dev` | Apple Silicon macOS (`osx-arm64`) | `dev` | CPU development and checks |
| `gpu-dev-castner` | Castner ARM Linux with GH200 GPUs | `dev`, `gpu` | GPU development and HPC training |
| `gpu-dev-protage` | Protage x86-64 Linux with MI300A GPUs | `dev`, `gpu` | ROCm GPU development and HPC training |

- `pyproject.toml` defines package metadata, common dependencies, and optional
  dependency groups. The `gpu` group adds dawgz and W&B. The separate
  `legacy-xformers` extra retains the old CUDA dependency for historical use;
  it requires a separate compatible PyTorch configuration.
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
CPU (`aarch64`) and NVIDIA GH200 GPUs. The GPU targets using
`linux-64` (Intel/AMD CPUs) cannot run on this machine even though both
systems use NVIDIA GPUs. This target therefore selects ARM Linux packages
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

#### Protage: AMD MI300A environment

`gpu-dev-protage` targets Protage's x86-64 CPUs (`linux-64`) and AMD Instinct
MI300A GPUs (`gfx942`). The inspected compute node runs RHEL 9.6 and exposes
four GPU agents. MI300A combines CPU and GPU compute with shared HBM memory;
CPU agents in `rocminfo` are expected alongside GPU agents. This environment
uses standard upstream packages, not an HPE-customized or performance-tuned build.

The environment pins the official Python 3.11 PyTorch **2.9.1+rocm6.4** wheel.
This is a ROCm 6.4 build with bundled runtime libraries; it does not require
loading a matching site module just to install it. The wheel requires glibc
2.28 or newer. Pixi does not install the host AMD driver, Slurm, or the site's
ROCm modules. GPU execution passed on Protage with the default `rocm/7.2.1`
module loaded; no module switch is needed for this setup. Loading a module
does not change the PyTorch build installed in the Pixi environment.

Install from the project directory on the login node; no GPU allocation or
ROCm module is required for this prebuilt-package installation:

```sh
pixi install --locked -e gpu-dev-protage
pixi shell -e gpu-dev-protage
```

The installed environment was accessible from both the compute and login
nodes after leaving the allocation, consistent with shared project storage.
Module availability can differ between nodes; that does not prevent installation.

GPUs are not available on log-in nodes. For an interactive compute-node shell, the site-provided request is:

```sh
srun -N 1 -n 1 --cpus-per-task 96 -p parry --pty bash
```

This requests one node, one task, and 96 Slurm CPUs in `parry`.  Four GPU agents were
visible in the reported allocation, which alone does not establish exclusive
ownership. In the allocated shell, enter the same project directory and keep
the default `rocm/7.2.1` module. To display the loaded modules:

```sh
module list
```

The following GPU smoke test passed within the allocation and can be rerun
after environment or system changes:

```sh
pixi run -e gpu-dev-protage python - <<'PYTORCH'
import torch
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip, "CUDA:", torch.version.cuda)
assert torch.version.hip is not None, "Expected a ROCm PyTorch build"
assert torch.cuda.is_available(), "No accessible AMD GPU"
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
x = torch.randn(256, 256, device="cuda", requires_grad=True)
(x @ x.T).square().mean().backward()
torch.cuda.synchronize()
print("GPU forward/backward passed")
PYTORCH
```

The successful run reported:

```text
PyTorch: 2.9.1+rocm6.4
HIP: 6.4.43484-123eb5128 CUDA: None
GPU count: 4
0 AMD Instinct MI300A Accelerator
1 AMD Instinct MI300A Accelerator
2 AMD Instinct MI300A Accelerator
3 AMD Instinct MI300A Accelerator
GPU forward/backward passed
```

`CUDA: None` is expected for this ROCm build. The run also printed
`/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory` four times;
these messages did not prevent GPU detection or the forward/backward operation.
The test enumerated all four GPUs and performed the computation on the default
GPU. It establishes basic GPU execution with the loaded `rocm/7.2.1` module.

ROCm intentionally retains PyTorch's `cuda` device names and `torch.cuda`
interfaces. The distributed `nccl` backend uses RCCL in ROCm builds. These names
therefore remain in the training code. The legacy xformers extra is not installed;
graph attention uses the existing PyTorch fallback.

Configure dataset/statistics paths, `slurm_account`, `wandb_entity`, and W&B
credentials as described for Castner. Use the activated `gpu-dev-protage`
environment in the submission shell with the default `rocm/7.2.1` module:

```sh
cd experiments/autoencoder
python train.py --gpus 4 --partition parry ae=ico7to5
```

This submits a Slurm job rather than directly running interactive training.
Use a separate submission shell after releasing an interactive allocation unless
your site's workflow specifies otherwise. Ensure the submitted job inherits the
`gpu-dev-protage` environment and the module configuration above. Set account,
partition, and resource arguments according to site policy. Dataset paths and
ERA5 assumptions require adaptation for RIOPS.

Official build reference: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

### Git synchronization and dependency changes

Commit source code, `pixi.toml`, `pyproject.toml`, and `pixi.lock` when generated.
Keep `.pixi/` environments local; do not commit datasets, credentials, or large
training outputs. Git synchronizes configuration and code, not installed packages.

After pulling changes, run `pixi install --locked -e mac-dev` on the Mac or
`pixi install --locked -e gpu-dev-castner` on Castner, or
`pixi install --locked -e gpu-dev-protage` on Protage to synchronize with the committed lockfile.
Use environment names explicitly with `pixi install`, `pixi run`, and `pixi shell`.
The manifest also has Pixi's implicit default environment, which does not install
`appa2`; select one of the named environments above for project work.

Keep shared Python requirements in `pyproject.toml` and GPU-only requirements
in its `gpu` group. Use Pixi for environment/platform settings and avoid ad hoc
`pip install` changes that are not recorded in configuration.


## Cross-platform locking

Pixi resolves all environment/platform combinations when updating `pixi.lock`,
even when installing only one named environment. All three named environments
include host-compatible variants for Apple Silicon, Castner ARM Linux, and
Protage x86-64 Linux so PyPI metadata builds can run on any of those hosts.
macOS variants use `dev` extras and PyTorch 2.5.1; ARM Linux variants use the
CUDA 2.6.0 wheel; x86-64 Linux variants use the ROCm 2.9.1 wheel. GPU environments
select `dev` plus `gpu` extras on Linux. Use the environment named for your
machine; the extra variants support locking, not hardware interchangeability.

## Verification status

The manifest and regenerated lockfile target Apple Silicon, Castner ARM Linux,
and Protage x86-64 Linux. `pixi lock --check` passed after adding Protage.
The ROCm build and its matching Triton wheel are explicitly pinned to official
PyTorch download URLs.
The Protage environment installed successfully on a compute node and remained
accessible from the login node. With `rocm/7.2.1` loaded, PyTorch reported
2.9.1+rocm6.4, HIP 6.4.43484-123eb5128, and four MI300A GPUs; the default-GPU
forward/backward smoke test passed. No module switch was needed. These results
cover basic GPU execution; they do not establish graph-attention performance,
mixed-precision behavior, or multi-GPU distributed training results.
The `mac-dev` environment installed successfully. Runtime checks passed for the
editable `appa2` import, a PyTorch CPU tensor operation, importing graph attention
without xformers, a small graph-attention fallback forward/backward operation,
and creating a small 42-node icosphere. Lockfile consistency and conda-forge-only
Conda package sources were also checked. These are local Mac and dependency
resolution checks, not Castner runtime validation. GPU installation and
training still need verification on Castner; the ARM wheel cannot run on the Mac.
