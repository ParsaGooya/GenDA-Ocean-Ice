r"""Tools to download ERA5 data from WeatherBench2."""

import numpy as np
import os
import pandas as pd
import xarray as xr

from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence, Tuple

from .const import (
    ERA5_AVAILABLE_CONFIGURATIONS,
    ERA5_NUM_TOTAL_LEVELS,
    ERA5_RESOLUTION,
    ERA5_TIME_INTERVAL,
)


def display_available_datasets() -> None:
    r"""Displays available ERA5 datasets on WeatherBench2."""
    print("Available ERA5 datasets configurations:")
    for config in ERA5_AVAILABLE_CONFIGURATIONS:
        time_interval, levels, res_long, res_lat, poles = config
        print(f"  {time_interval}h, {levels} levels, {res_long}x{res_lat}, poles={poles}")


def get_bucket_url(
    time_interval: int,
    total_levels: int,
    resolution: Tuple[int, int],
    include_poles: bool,
) -> str:
    r"""Returns URL for the ERA5 dataset based on configuration.

    Arguments:
        time_interval: Time interval between data points (1 or 6 hours).
        total_levels: Number of pressure levels.
        resolution: Spatial resolution (longitude, latitude).
        include_poles: Whether pole data is included.
    """

    config = (time_interval, total_levels, *resolution, include_poles)

    if config not in ERA5_AVAILABLE_CONFIGURATIONS:
        if (*config[:-1], not include_poles) in ERA5_AVAILABLE_CONFIGURATIONS:
            raise ValueError(
                f"Configuration {config} not found. Try with include_poles={not include_poles}."
            )
        display_available_datasets()
        raise ValueError(f"Configuration {config} not found in datasets.")

    return f"gs://weatherbench2/datasets/era5/1959-{ERA5_AVAILABLE_CONFIGURATIONS[config]}"


def download(
    output_path: Path,
    start_date: str,
    end_date: str,
    variables: Optional[Sequence[str]] = None,
    pressure_levels: Optional[Sequence[int]] = None,
    time_interval: int = 1,
    total_levels: int = 37,
    resolution: Tuple[int, int] = (1440, 721),
    include_poles: bool = True,
    chunk_size: int = 1,
) -> None:
    r"""Downloads and saves ERA5 data from WeatherBench2 to a local disk.

    Arguments:
        output_path: Path to save Zarr dataset
        start_date: Start date of the data split (format: 'YYYY-MM-DD').
        end_date: End date of the data split (format: 'YYYY-MM-DD').
        variables: Variable names to retain from the dataset.
        pressure_levels: List of pressure levels to include in the dataset.
        time_interval: Time interval between data points.
        total_levels: Total number of vertical levels in the dataset.
        resolution: Spatial resolution of dataset (longitude, latitude).
        include_poles: Whether to include pole data in the dataset.
        chunk_hours: Size of time chunks (in hours) for processing.
    """

    url = get_bucket_url(time_interval, total_levels, resolution, include_poles)
    print(f"Downloading ERA5 data from {url}...", flush=True)

    data = xr.open_zarr(
        url, chunks="auto", storage_options={"token": "anon"}, zarr_format=2, consolidated=False
    )

    if variables:
        data = data[variables]

    if "time" in data.dims:
        time_slice = slice(start_date, end_date)
        if "level" in data.dims and pressure_levels:
            try:
                data = data.sel(time=time_slice, level=pressure_levels)
            except KeyError:
                available_levels = data.level.values
                raise KeyError(
                    f"Invalid pressure levels. Available levels: {available_levels}"
                ) from None
        else:
            data = data.sel(time=time_slice)

    dataset_size_gb = data.nbytes / 1e9
    print(f"Size of the dataset: {dataset_size_gb:.2f} GB", flush=True)

    times = None
    if "time" in data.coords:
        times = data.coords["time"]
    elif os.path.exists(output_path):
        print("Loading existing data to get time coordinate...", flush=True)
        existing_data = xr.open_zarr(output_path, chunks="auto", zarr_format=2, consolidated=False)
        if "time" in existing_data.coords:
            times = existing_data.coords["time"]
    else:
        if "T" not in start_date:
            start_date += "T00"
        if "T" not in end_date:
            end_date += "T23"

        times = pd.date_range(start=start_date, end=end_date, freq=f"{time_interval}H")
        times = xr.DataArray(times, dims=["time"], name="time")

    if times is not None:
        times = times.sel(time=slice(start_date, end_date))

        # Expand variables without time dimension (e.g., orography)
        for var in data:
            if "time" not in data[var].dims:
                print("Expanding variable without time dimension:", var, flush=True)
                data[var] = data[var].expand_dims(time=times)
    else:
        print("No time coordinate found in the dataset.", flush=True)
        print("Variables without time dimension cannot be expanded!", flush=True)

    # Chunk the data to limit memory usage during saving
    data = data.chunk({"time": chunk_size})
    for var in data:
        # Ensure correct Zarr encoding
        del data[var].encoding["chunks"]

    data.to_zarr(output_path, mode="a-", zarr_format=2)

    print(f"Download complete. Zarr store saved to: {output_path}", flush=True)


DAILY_FORCING_FIELDS = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
    "total_precipitation": "tp",
}
DAILY_FORCING_UNITS = {
    "t2m": "K",
    "q2m": "kg kg-1",
    "u10": "m s-1",
    "v10": "m s-1",
    "tp": "m",
}


def _daily_forcing(hourly: xr.Dataset, days: pd.DatetimeIndex) -> xr.Dataset:
    # The final midnight belongs to the preceding day's precipitation total.
    state = hourly.isel(time=slice(0, -1))
    vapour_pressure = 611.21 * np.exp(17.502 * (state.d2m - 273.16) / (state.d2m - 32.19))
    q2m = 0.622 * vapour_pressure / (state.sp - 0.378 * vapour_pressure)

    means = state[["t2m", "u10", "v10"]].assign(q2m=q2m)
    means = means.chunk({"time": 24}).coarsen(time=24).mean(skipna=False)
    means = means.assign_coords(time=days)
    tp = hourly.tp.isel(time=slice(1, None))
    tp = tp.chunk({"time": 24}).coarsen(time=24).sum(skipna=False).assign_coords(time=days)
    daily = means.assign(tp=tp).drop_encoding()
    daily.attrs = {}
    for name, units in DAILY_FORCING_UNITS.items():
        daily[name].attrs = {
            "units": units,
            "cell_methods": "time: sum" if name == "tp" else "time: mean",
        }
    return daily[list(DAILY_FORCING_UNITS)]


def _validate_daily_file(path, days, source, metadata):
    with xr.open_dataset(path, engine="netcdf4") as existing:
        if (
            set(existing.data_vars) != set(DAILY_FORCING_UNITS)
            or any(existing.attrs.get(key) != value for key, value in metadata.items())
            or not existing.indexes["time"].equals(days)
            or any(
                not existing.indexes[dim].equals(source.indexes[dim])
                for dim in ("latitude", "longitude")
            )
            or any(
                existing[name].dims != ("time", "latitude", "longitude")
                or existing[name].attrs.get("units") != units
                or existing[name].attrs.get("cell_methods")
                != ("time: sum" if name == "tp" else "time: mean")
                for name, units in DAILY_FORCING_UNITS.items()
            )
        ):
            raise ValueError(f"Existing ERA5 output does not match this request: {path}")


def download_daily_forcing(
    output_dir: str,
    start_date: str,
    end_date: str,
    south: float = 24.5,
    north: float = 90.0,
) -> None:
    """Write regional daily forcing from hourly WeatherBench2, one file per month."""
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    if not -90 <= south <= north <= 90:
        raise ValueError("Latitude bounds must satisfy -90 <= south <= north <= 90")

    url = get_bucket_url(ERA5_TIME_INTERVAL, ERA5_NUM_TOTAL_LEVELS, ERA5_RESOLUTION, True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_midnight = end + timedelta(days=1)
    expected_hours = pd.date_range(start, final_midnight, freq="h")
    metadata = {
        "source": url,
        "time_zone": "UTC",
        "precipitation_interval": "24 one-hour accumulations ending at next-day 00:00 UTC",
    }
    print(f"Opening {url}", flush=True)

    with xr.open_zarr(url, chunks={}, storage_options={"token": "anon"}, zarr_format=2) as source:
        missing = DAILY_FORCING_FIELDS.keys() - source.data_vars.keys()
        if missing:
            raise ValueError(f"Missing WeatherBench2 fields: {', '.join(sorted(missing))}")
        if any(
            set(source[name].dims) != {"time", "latitude", "longitude"}
            for name in DAILY_FORCING_FIELDS
        ):
            raise ValueError("Expected all forcing fields on the hourly surface grid")
        latitude = source.indexes["latitude"]
        region = slice(north, south) if latitude.is_monotonic_decreasing else slice(south, north)
        hourly = (
            source[list(DAILY_FORCING_FIELDS)]
            .sel(latitude=region, time=slice(expected_hours[0], expected_hours[-1]))
            .rename(DAILY_FORCING_FIELDS)
            .transpose("time", "latitude", "longitude")
        )
        if not hourly.sizes["latitude"]:
            raise ValueError("No source latitude points in the requested region")
        if not hourly.indexes["time"].equals(expected_hours):
            available = source.indexes["time"]
            raise ValueError(
                f"Expected complete hourly coverage from {expected_hours[0]} through "
                f"{expected_hours[-1]}, including the final precipitation midnight. "
                f"Source spans {available.min()} through {available.max()}."
            )

        while start <= end:
            last_day = monthrange(start.year, start.month)[1]
            stop = min(end, start.replace(day=last_day))
            days = pd.date_range(start, stop, freq="D")
            output = output_dir / f"era5_daily_{start}_{stop}.nc"
            if output.exists():
                _validate_daily_file(output, days, hourly, metadata)
                print(f"Skipping validated {output}", flush=True)
            else:
                print(f"Processing {start} through {stop}", flush=True)
                month = hourly.sel(time=slice(days[0], days[-1] + pd.Timedelta(days=1)))
                daily = _daily_forcing(month, days).assign_attrs(metadata)
                encoding = {
                    name: {"dtype": "float32", "zlib": True, "complevel": 1}
                    for name in DAILY_FORCING_UNITS
                }
                with TemporaryDirectory(dir=output_dir) as work:
                    temporary = Path(work) / output.name
                    daily.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
                    temporary.replace(output)
                print(f"Wrote {output}", flush=True)
            start = stop + timedelta(days=1)
