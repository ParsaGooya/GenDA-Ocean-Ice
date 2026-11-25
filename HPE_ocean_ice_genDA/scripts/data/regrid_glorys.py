"""Smooth and subsample regular 1/12° GLORYS NetCDF files."""

from pathlib import Path
from tempfile import TemporaryDirectory

from dawgz import after, job, schedule
from collections import defaultdict
import dask
import gcm_filters
import numpy as np
import xarray as xr
from functools import partial

NATIVE_SPACING = 1 / 12
EARTH_RADIUS_METERS = 6_371_000.0
NETCDF_SUFFIXES = {".nc", ".nc4", ".netcdf"}


def horizontal_dimensions(ds: xr.Dataset) -> tuple[str, str]:
    dims = []
    for names in (("latitude", "lat"), ("longitude", "lon")):
        dim = next((name for name in names if name in ds.coords), None)
        if dim is None or ds[dim].dims != (dim,):
            raise ValueError(f"Expected a one-dimensional {names[0]} coordinate.")
        coord = ds[dim].values
        if (
            coord.size < 2
            or not np.isfinite(coord).all()
            or not np.allclose(np.diff(coord), NATIVE_SPACING, atol=5e-5, rtol=0)
        ):
            raise ValueError(f"{dim} must be increasing at 1/12 degree spacing.")
        dims.append(dim)
    return dims[0], dims[1]


def physical_grid_metrics(
    ds: xr.Dataset, latitude: str, longitude: str
) -> tuple[dict[str, xr.DataArray], float]:
    """Build spherical cell geometry for a regular, cell-centered lat/lon grid."""
    lat = np.deg2rad(ds[latitude].astype("float64"))
    if bool((abs(lat) > np.pi / 2).any()):
        raise ValueError("Latitude must lie within [-90, 90] degrees.")
    step = np.deg2rad(NATIVE_SPACING)
    south = (lat - step / 2).clip(min=-np.pi / 2)
    north = (lat + step / 2).clip(max=np.pi / 2)
    # Use the midpoint of a clipped polar cell to keep its zonal spacing positive.
    metric_latitude = (south + north) / 2
    zonal_cell_width = EARTH_RADIUS_METERS * step * np.cos(metric_latitude)
    latitude_spacing = xr.full_like(lat, EARTH_RADIUS_METERS * step)
    metrics = {
        "dxw": zonal_cell_width,
        "dyw": EARTH_RADIUS_METERS * (north - south),
        "dxs": EARTH_RADIUS_METERS * step * np.cos(south),
        "dys": latitude_spacing,
        "area": EARTH_RADIUS_METERS**2 * step * (np.sin(north) - np.sin(south)),
        "kappa_w": xr.ones_like(lat),
        "kappa_s": xr.ones_like(lat),
    }
    # The spacing across a face controls the shortest grid interval. An edge
    # itself can have zero length at a pole.
    dx_min = min(float(zonal_cell_width.min()), float(latitude_spacing.min()))
    return {
        name: metric.expand_dims({longitude: ds[longitude]}).transpose(latitude, longitude)
        for name, metric in metrics.items()
    }, dx_min


def smooth_glorys(
    ds: xr.Dataset,
    filtering_method: str,
    *,
    spatial_filter_scale: float | None = None,
    num_filter_grids: float | None = None,
    local_length_scales_file: Path | str | None = None,
    local_length_scale_variable: str = "length_scale",
    filter_type: str = "gaussian",
    mask_reference: xr.Dataset | None = None,
) -> xr.Dataset:
    """Smooth horizontally using a scale in meters or in native grid intervals."""
    try:
        filter_shape = gcm_filters.FilterShape[filter_type.upper()]
    except KeyError as error:
        raise ValueError("filter_type must be 'gaussian' or 'taper'.") from error

    latitude, longitude = horizontal_dimensions(ds)
    dims = (latitude, longitude)
    # The kernels wrap both axes. A dry ghost cell blocks wraparound at
    # regional boundaries while leaving the global longitude seam periodic.
    padding = {latitude: (0, 1)}
    if not np.isclose(ds.sizes[longitude] * NATIVE_SPACING, 360):
        padding[longitude] = (0, 1)

    if filtering_method == "fixed_grid_scale_filtering":
        scale, scale_name = num_filter_grids, "num_filter_grids"
        grid_vars, dx_min = {}, 1.0
        grid_type = gcm_filters.GridType.REGULAR_WITH_LAND
    elif filtering_method in ("fixed_length_scale_filtering", "local_length_scale_filtering"):
        grid_vars, dx_min = physical_grid_metrics(ds, latitude, longitude)
        grid_type = gcm_filters.GridType.IRREGULAR_WITH_LAND
        scale, scale_name = spatial_filter_scale, "spatial_filter_scale"
        if filtering_method == "local_length_scale_filtering":
            if local_length_scales_file is None:
                raise ValueError("Provide local_length_scales_file for local-length filtering.")
            with xr.open_dataset(local_length_scales_file) as scales:
                length = scales[local_length_scale_variable]
                if set(length.dims) != set(dims) or any(
                    dim not in length.coords or not length[dim].equals(ds[dim])
                    for dim in dims
                ):
                    raise ValueError("Local lengths must be 2D and match the GLORYS grid exactly.")
                length = length.transpose(*dims).astype("float64").load()
            if not bool((np.isfinite(length) & (length > 0)).all()):
                raise ValueError("Local lengths must be finite and positive in meters.")

            # At closed edges the neighboring value is unused by the wet-mask
            # stencil. Reuse the boundary cell there instead of wrapping it.
            west = (
                length.shift({longitude: 1}).fillna(length)
                if longitude in padding
                else length.roll({longitude: 1}, roll_coords=False)
            )
            south = length.shift({latitude: 1}).fillna(length)
            west_length = 0.5 * length + 0.5 * west
            south_length = 0.5 * length + 0.5 * south
            scale = max(float(west_length.max()), float(south_length.max()))
            scale_name = "maximum local face length"
            grid_vars["kappa_w"] = (west_length / scale) ** 2
            grid_vars["kappa_s"] = (south_length / scale) ** 2
    else:
        raise ValueError(f"Unsupported filtering_method: {filtering_method!r}.")

    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{scale_name} must be finite and positive for the selected mode.")

    crop = {dim: slice(0, ds.sizes[dim]) for dim in padding}
    padded_metrics = {name: metric.pad(padding, mode="edge") for name, metric in grid_vars.items()}

    smoothed = ds.copy(deep=False)
    for name, data in ds.data_vars.items():
        if not set(dims).issubset(data.dims) or not np.issubdtype(data.dtype, np.floating):
            continue

        first = data
        if "time" in data.dims:
            first = (
                data.isel(time=0, drop=True) if mask_reference is None else mask_reference[name]
            )
        field = data.pad(padding, mode="edge")
        wet_mask = first.notnull().astype("int8").pad(padding, mode="constant", constant_values=0)
        wet_mask = wet_mask.assign_coords({dim: field[dim] for dim in padding})
        wet_mask = wet_mask.chunk({dim: -1 if dim in dims else 1 for dim in wet_mask.dims})
        smoother = gcm_filters.Filter(
            filter_scale=scale,
            dx_min=dx_min,
            filter_shape=filter_shape,
            grid_type=grid_type,
            grid_vars={**padded_metrics, "wet_mask": wet_mask},
        )

        # Keep each horizontal slice intact for the stencil, and process
        # one time/depth slice at a time when writing the output.
        field = field.chunk({dim: -1 if dim in dims else 1 for dim in field.dims})
        filtered = smoother.apply(field, dims=dims).isel(crop)
        smoothed[name] = (
            filtered.where(first.notnull() & data.notnull())
            .astype(data.dtype)
            .transpose(*data.dims)
        )
        smoothed[name].attrs = data.attrs.copy()

    return smoothed


def main(
        sources: list[Path], 
         filtering_method: str,
         filter_type: str,
         spatial_filter_scale: float,
         num_filter_grids: int,
         subsample_stride: int,
         local_length_scale_variable: str,
         local_length_scales_file: Path,
         output_dir: Path, 
         input_dir: Path,
         MIN_LAT: float | None,
         time_batch_size: int = 7,
):
    if time_batch_size < 1:
        raise ValueError("time_batch_size must be positive.")

    for source in sources:
        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.is_file():
            print(f"Skipped {source.relative_to(input_dir)} ...", flush=True)
            continue

        print(f"Start regridding {source.relative_to(input_dir)} ...", flush=True)

        with (
            xr.open_dataset(source, chunks={}) as ds,
            TemporaryDirectory(dir=destination.parent) as work,
            dask.config.set(scheduler="single-threaded"),
        ):
            latitude, longitude = horizontal_dimensions(ds)
            has_time = "time" in ds.dims
            count = ds.sizes.get("time", 1)
            if count == 0:
                raise ValueError(f"No time records in {source}")
            # Every batch uses the original first record for the filter's wet mask.
            reference = ds.isel(time=0, drop=True) if has_time else None
            batch_paths = []
            for start in range(0, count, time_batch_size):
                stop = min(start + time_batch_size, count)
                print(f"  Processing records {start + 1}–{stop} of {count}", flush=True)
                batch = ds.isel(time=slice(start, stop)) if has_time else ds
                smoothed = smooth_glorys(
                    batch,
                    filtering_method,
                    spatial_filter_scale=spatial_filter_scale,
                    num_filter_grids=num_filter_grids,
                    local_length_scales_file=local_length_scales_file,
                    local_length_scale_variable=local_length_scale_variable,
                    filter_type=filter_type,
                    mask_reference=reference,
                )
                coarse = smoothed.isel({
                    dim: slice(None, None, subsample_stride) for dim in (latitude, longitude)
                })
                if MIN_LAT is not None:
                    coarse = coarse.sel({latitude: slice(MIN_LAT, None)})

                batch_path = Path(work) / f"batch-{start}.part"
                coarse.to_netcdf(batch_path, engine="netcdf4")
                batch_paths.append(batch_path)

            temporary_path = batch_paths[0]
            if len(batch_paths) > 1:
                # The final write reads the already-filtered coarse batches.
                temporary_path = Path(work) / "combined.part"
                with xr.open_mfdataset(
                    batch_paths,
                    engine="netcdf4",
                    combine="nested",
                    concat_dim="time",
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    join="exact",
                ) as combined:
                    combined.to_netcdf(temporary_path, engine="netcdf4")
            temporary_path.replace(destination)
        print(f"{source} -> {destination}", flush=True)


if __name__ == "__main__":
    input_dir = Path("/lus/scratch/PGooya/data/glorys_daily")
    output_dir = Path("/lus/scratch/PGooya/data/glorys_daily_25")
    filter_type = "gaussian"
    filtering_method = "fixed_grid_scale_filtering"
    num_filter_grids = 8
    spatial_filter_scale = 100_000  # meters
    local_length_scales_file = Path("/path/to/local_length_scales.nc")
    local_length_scale_variable = "length_scale"  # 2D, in meters
    subsample_stride = 3  # Native grid intervals per output grid interval.
    time_batch_size = 7  # Daily records per filtering batch.
    MIN_LAT: float | None = 25.0


    hardware = {
        "backend": "slurm",  # or async
        "job":
            {"cpus": 4,
            "ram": "64GB",
            "time": "12:00:00",
            "partition": "parry",}
    }

    if type(subsample_stride) is not int or subsample_stride < 1:
        raise ValueError("subsample_stride must be a positive integer.")

    input_dir = input_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Expected a GLORYS input directory: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Choose an output directory outside the input directory.")

    sources = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in NETCDF_SUFFIXES
    )

    if not sources:
        raise FileNotFoundError(f"No NetCDF files found in {input_dir}")

    sources_by_year = defaultdict(list)
    for source in sources:
        year = source.name.split("_")[3][:4]
        sources_by_year[year].append(source)

    jobs = []

    for year, year_sources in sorted(sources_by_year.items()):
        
        dawgz_download = partial(
            main,
            year_sources,
            filtering_method,
            filter_type,
            spatial_filter_scale,
            num_filter_grids,
            subsample_stride,
            local_length_scale_variable,
            local_length_scales_file,
            output_dir,
            input_dir,
            MIN_LAT,
            time_batch_size,
        )

        year_job = job(
            dawgz_download,
            name=f"glorys_regrid_{year}",
            **hardware["job"],
        )
        if jobs:
            year_job = after(jobs[-1], status="any")(year_job)
        jobs.append(year_job)

    options = {"export": "ALL"}
    if hardware.get("account") is not None:
        options["account"] = hardware["account"]

    schedule(*jobs, backend=hardware["backend"], **options)


