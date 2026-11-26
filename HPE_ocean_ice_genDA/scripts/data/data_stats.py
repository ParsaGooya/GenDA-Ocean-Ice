r"""Script to compute pixel-space statistics over a dataset."""

import numpy as np
import shutil
import sys
import time
import torch
import wandb
import xarray as xr

from dawgz import after, job, schedule
from omegaconf import OmegaConf
from pathlib import Path

from appa2.config.hydra import compose
from appa2.data.const import (
    CONTEXT_VARIABLES,
    ATM_FORCING_VARIABLES,
    SURFACE_VARIABLES,
    OCEAN_VARIABLES,
)

from appa2.data.dataloaders import get_dataloader
from appa2.data.data_config import get_dataset_specs
from appa2.date import split_daily_interval


def channel_moments(values):
    """Return finite counts, sums, and squared sums for each channel."""
    values = values.cuda()
    finite = torch.isfinite(values)
    values = torch.where(finite, values, 0).double()
    sample_dims = (0, 1, 3, 4)  # Batch, trajectory, latitude, longitude
    return (
        finite.sum(dim=sample_dims),
        values.sum(dim=sample_dims),
        values.square().sum(dim=sample_dims),
    )


def compute_statistics(config):
    num_chunks = config.num_chunks
    data_configs = get_dataset_specs(config.state_grid)
    output_path = Path(config.get("output_path") or data_configs.path_stat_state)
    output_suffix = output_path.suffix.lower()
    if output_suffix not in {"", ".nc", ".zarr"}:
        raise ValueError("output_path must end in .nc or .zarr, or have no suffix for Zarr.")
    tmp_path = output_path.parent / "tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)

    start_date = config.get("start_date")
    end_date = config.get("end_date")
    train_start, train_end = data_configs.time_splits[0]
    if start_date is None:
        start_date = train_start
    if end_date is None:
        end_date = train_end

    time_intervals = split_daily_interval(num_chunks, start_date, end_date)
    num_chunks = len(time_intervals)

    @job(
        name="appa2 train stats (chunk)",
        array=num_chunks,
        **config.hardware.chunk,
    )
    def compute_stats_chunk(rank: int):
        chunk_start, chunk_end = time_intervals[rank]

        dataset = data_configs.dataset(
            state_variables_path=data_configs.state_variables_path,
            start_date=chunk_start,
            end_date=chunk_end,
            trajectory_size=1,
            surface_variables= SURFACE_VARIABLES,
            depth_variables= OCEAN_VARIABLES,
            forcing_variables= ATM_FORCING_VARIABLES,
            context_variables= CONTEXT_VARIABLES,
            forcing_path = data_configs.forcing_path,
            depths=data_configs.depth_levels,
            fill_nans=False,
        )

        dataloader = get_dataloader(
            dataset,
            batch_size=config.batch_size,
            num_workers=8,
            prefetch_factor=1,
            shuffle=False,
        )

        len_dl = len(dataloader)

        num_channels = (
            len(SURFACE_VARIABLES)
            + len(OCEAN_VARIABLES) * len(data_configs.depth_levels)
            + len(ATM_FORCING_VARIABLES)
        )
        valid_count = torch.zeros(num_channels, dtype=torch.int64, device="cuda")
        sums = torch.zeros(num_channels, dtype=torch.float64, device="cuda")
        sum_of_squares = torch.zeros_like(sums)
        start_time = time.time()

        for idx, (data, _, _, forcing) in enumerate(dataloader):
            state_moments = channel_moments(data)
            forcing_moments = channel_moments(forcing)
            counts, totals, squared_totals = (
                torch.cat((state, atmosphere))
                for state, atmosphere in zip(state_moments, forcing_moments)
            )
            valid_count += counts
            sums += totals
            sum_of_squares += squared_totals

            if idx % config.log_every == 0:
                elapsed = time.time() - start_time
                print(f"Chunk {rank}, Step {idx}/{len_dl}, Elapsed: {elapsed:.2f}s")
                start_time = time.time()

        torch.save(
            {
                "valid_count": valid_count.cpu(),
                "sum": sums.cpu(),
                "sum_of_squares": sum_of_squares.cpu(),
            },
            tmp_path / f"{rank}.pt",
        )

    @after(compute_stats_chunk)
    @job(
        name="appa2 train stats (merge)",
        **config.hardware.aggregate,
    )
    def aggregate():
        chunks = [
            torch.load(tmp_path / f"{i}.pt", weights_only=True) for i in range(num_chunks)
        ]
        valid_count, sums, sum_of_squares = (
            torch.stack([chunk[key] for chunk in chunks]).sum(dim=0)
            for key in ("valid_count", "sum", "sum_of_squares")
        )
        if (valid_count == 0).any():
            empty_channels = torch.where(valid_count == 0)[0].tolist()
            raise ValueError(f"No finite values for stats channel indices {empty_channels}.")

        means = sums / valid_count
        variance = (sum_of_squares / valid_count - means.square()).clamp_min(0)
        stds = variance.sqrt().clamp_min(1e-32)

        means = means.float().numpy()
        stds = stds.float().numpy()

        num_surface_vars = len(SURFACE_VARIABLES)
        num_ocean_vars = len(OCEAN_VARIABLES)
        num_levels = len(data_configs.depth_levels)
        num_forcing_vars = len(ATM_FORCING_VARIABLES)
        depth_dim = data_configs.depth_dim_name

        ds = xr.Dataset(
            coords={
                "statistic": ["mean", "std"],
                depth_dim: data_configs.depth_levels,
            },
        )

        # Surface variables
        for i in range(num_surface_vars):
            ds[SURFACE_VARIABLES[i]] = xr.DataArray(
                data=np.array([means[i], stds[i]]),
                coords={
                    "statistic": ["mean", "std"],
                },
            )

        # Ocean variables
        for i in range(num_ocean_vars):
            offset_idx = num_surface_vars + i * num_levels
            var_name = OCEAN_VARIABLES[i]
            ds[var_name] = xr.DataArray(
                data=np.stack(
                    [
                        means[offset_idx : offset_idx + num_levels],
                        stds[offset_idx : offset_idx + num_levels],
                    ],
                ),
                coords={
                    "statistic": ["mean", "std"],
                    depth_dim: data_configs.depth_levels,
                },
            )

        # Forcing variables
        for i in range(num_forcing_vars):
            offset_idx = num_surface_vars + num_ocean_vars * num_levels + i
            ds[ATM_FORCING_VARIABLES[i]] = xr.DataArray(
                data=np.array([means[offset_idx], stds[offset_idx]]),
                coords={
                    "statistic": ["mean", "std"],
                },
            )

        if output_suffix == ".nc":
            ds.to_netcdf(output_path, mode="w")
        else:
            ds.to_zarr(output_path, mode="w", consolidated=True)

        shutil.rmtree(tmp_path, ignore_errors=True)

    options = {"export": "ALL"}
    if config.hardware.get("account") is not None:
        options["account"] = config.hardware["account"]

    schedule(
        aggregate,
        name="appa2 train stats",
        backend="slurm",
        **options
    )


if __name__ == "__main__":
    config = compose("configs/data_stats.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" not in config:
        config["id"] = wandb.util.generate_id()

    compute_statistics(config)
