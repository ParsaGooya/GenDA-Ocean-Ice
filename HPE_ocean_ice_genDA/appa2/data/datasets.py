r"""Datasets."""

import dask
import h5py
import numpy as np
import torch
import xarray as xr

from omegaconf import ListConfig
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable, Optional, Sequence, Tuple, Union

from .const import (
    DATA_NAN_VALUE,
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    TOY_DATASET_DATES_TEST,
    TOY_DATASET_DATES_TRAINING,
    TOY_DATASET_DATES_VALIDATION,
    TOY_DATASET_VARIABLES,
)
from ..config import PATH_ERA5
from ..date import assert_date_format, get_date_features
from ..save import safe_load


class ERA5Dataset(Dataset):
    r"""Represents an Zarr dataset formatted as ERA5.

    Reference format of ERA5 is WeatherBench2 (ref: https://weatherbench2.readthedocs.io/en/latest/data-guide.html).

    For ERA5, the path, variables, and levels can be found in the `appa2.data.const` module.

    Arguments:
        path: List of path to the Zarr datasets.
        start_date: Start date of the data split (format: 'YYYY-MM-DD').
        end_date: End date of the data split (format: 'YYYY-MM-DD').
        start_hour: Start hour of the data split (0-23).
        end_hour: End hour of the data split (0-23).
        trajectory_size: Number of time steps in each sample.
        trajectory_dt: Time step between samples.
        num_samples: Number of equally spaced samples to use. If None, all samples are used.
        state_variables: State variable names to retain from the dataset. If None, all variables are used.
        context_variables: Context variable names to retain from the dataset. If None, context=None is returned.
        levels: Atmospheric levels to retain from the dataset.
        transform: A callable for data transformation (e.g. standardization).
        fill_nans: Whether to set NaN values to 0 or not.
    """

    def __init__(
        self,
        path: Union[Path, Sequence[Path]],
        start_date: str,
        end_date: str,
        start_hour: int = 0,
        end_hour: int = 23,
        trajectory_size: int = 1,
        trajectory_dt: int = 1,
        num_samples: Optional[int] = None,
        state_variables: Optional[Sequence[str]] = None,
        context_variables: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        fill_nans: Optional[bool] = True,
    ):
        super().__init__()

        assert_date_format(start_date)
        assert_date_format(end_date)

        start_datetime = np.datetime64(f"{start_date}T{start_hour:02d}")
        end_datetime = np.datetime64(f"{end_date}T{end_hour:02d}")

        if isinstance(path, (list, ListConfig)):
            data_ls = []
            for pth in path:
                ds = xr.open_zarr(pth, chunks={"time": 1})
                if start_datetime <= ds.time.max() and end_datetime >= ds.time.min():
                    data_ls.append(ds)

            if len(data_ls) > 1:
                data = xr.concat(data_ls, dim="time")
            else:
                data = data_ls[0]
        else:
            data = xr.open_zarr(path)

        data = data.sortby("time")

        self.dataset = data.sel(time=slice(start_datetime, end_datetime))

        if levels is not None:
            self.dataset = self.dataset.sel(level=levels)

        if num_samples is not None:
            indices = np.linspace(0, len(self.dataset.time) - 1, num_samples, dtype=int)
            self.dataset = self.dataset.isel(time=indices)

        self.state_variables = state_variables
        self.context_variables = context_variables
        self.trajectory_size = trajectory_size
        self.trajectory_dt = trajectory_dt
        self.fill_nans = fill_nans
        self.transform = transform
        self.start_hour = start_hour

        if state_variables or context_variables:
            state_variables = state_variables or []
            context_variables = context_variables or []
            all_variables = state_variables + context_variables
            self.dataset = self.dataset[all_variables]

    def __len__(self) -> int:
        r"""Returns the total number of samples in the dataset."""

        return self.dataset.time.size - self.trajectory_size + 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        r"""Gets and preprocesses a sample from the dataset.

        Argument:
            idx: Index of the first time step in the sample. This offset is in HOURS!
                 Only the next steps in the trajectory are affected by trajectory_dt.

        Returns:
            sample: Preprocessed data tensor of shape (trajectory_size, z_total, x, y).
            time: Date features corresponding to the start of the sample.
        """

        return self._load_sub_trajectory(idx, idx + self.trajectory_size * self.trajectory_dt)

    def _zarr_to_tensor(
        self, sample: xr.DataArray, variables: Optional[Sequence[str]] = None
    ) -> Tensor:
        if variables is not None:
            sample = sample[variables]

        sample = sample.to_stacked_array(
            new_dim="z_total", sample_dims=("time", "longitude", "latitude")
        ).transpose("time", "z_total", ...)

        sample = torch.tensor(sample.load().data)

        return sample

    def _load_sub_trajectory(self, step_start: int, step_end: int) -> Tuple[Tensor, Tensor]:
        r"""Extracts and reshapes a sample from the dataset.

        Arguments:
            step_start: Start index of the sample.
            step_end: End index of the sample.

        Returns:
            state: A tensor containing the preprocessed state, of shape :math`(trajectory_size, s_vars, lat, lon)`.
            context: A tensor containing the preprocessed context, of shape :math`(trajectory_size, c_vars, lat, lon)`.
            time: Date features corresponding to the start of the sample.
        """

        # Handle large data by splitting into smaller chunks
        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            sub_trajectory = self.dataset.isel(
                time=slice(step_start, step_end, self.trajectory_dt)
            )

            if len(sub_trajectory.time) < self.trajectory_size:
                # Do not return incomplete trajectories. Can't be batched.
                raise ValueError(
                    "Incomplete trajectory. Probably wrong dates or trajectory dt value."
                )

            time = [
                get_date_features(sub_trajectory.time[i].values)
                for i in range(sub_trajectory.time.size)
            ]
            time = torch.stack(time, dim=0)

            # Process separately to support different levels for the context (<=> nb. channels ≠ nb. variables)
            if self.state_variables is None and self.context_variables is None:
                state = self._zarr_to_tensor(sub_trajectory)
                context_shape = list(state.shape)
                context_shape[1] = 0  # Does not break DataLoader batching
                context = torch.empty(context_shape)
            elif self.state_variables is None or len(self.state_variables) == 0:
                context = self._zarr_to_tensor(sub_trajectory, self.context_variables)
                state_shape = list(context.shape)
                state_shape[1] = 0  # Does not break DataLoader batching
                state = torch.empty(state_shape)
            elif self.context_variables is None or len(self.context_variables) == 0:
                state = self._zarr_to_tensor(sub_trajectory, self.state_variables)
                context_shape = list(state.shape)
                context_shape[1] = 0  # Does not break DataLoader batching
                context = torch.empty(context_shape)
            else:
                state = self._zarr_to_tensor(sub_trajectory, self.state_variables)
                context = self._zarr_to_tensor(sub_trajectory, self.context_variables)

            if self.transform is not None:
                state, context = self.transform(state, context)

            if self.fill_nans:  # Fill NaNs after transformation.
                state = torch.nan_to_num(state, nan=DATA_NAN_VALUE)
                context = torch.nan_to_num(context, nan=DATA_NAN_VALUE)

        return state, context, time


class GlorysDataset(Dataset):
    """Daily GLORYS NetCDF data, returned as ``(state, context, time)``.

    Arguments:
        state_variables_path: Required default source: a NetCDF file, directory
            of .nc files, or sequence of files/directories.
        start_date: Inclusive first day, YYYY-MM-DD.
        end_date: Inclusive last day, YYYY-MM-DD.
        surface_variables: Required ordered surface state names; [] disables group.
        depth_variables: Required ordered depth state names; [] disables group.
        forcing_variables: Required ordered forcing state names; [] disables group.
        context_variables: Required ordered context names; [] disables context.
        surface_path: Optional separate surface source.
        depth_path: Optional separate depth source.
        forcing_path: Optional separate forcing source.
        context_path: Optional separate context source.
            Each omitted group path falls back to state_variables_path. These
            arguments are complete paths, not suffixes appended to the default.
        trajectory_size: Number of daily fields in a sample.
        trajectory_dt: Spacing between fields within a sample, in days.
        num_samples: Number of evenly spaced valid trajectory starts, or None
            for all starts. Does not subsample the daily time axis.
        depths: Exact depth coordinates to select from depth-dependent fields.
        transform: Callable taking and returning (state, context).
        fill_nans: Replace NaNs after transformation.

    Groups without separate paths are selected together using variables_to_load.
    State channel order is surface, depth, then forcing; context is separate.
    Variable names must be unique across all four groups. All sources must already
    share a latitude/longitude grid and daily dates; aggregate subdaily forcing
    before using this loader.

    Tensors have shape (time, channels, longitude, latitude), preserving APPA's
    spatial order. Depth is flattened into channels, in variable order. Date
    features retain (year, month, day, hour), with hour set to zero.
    """

    def __init__(
        self,
        state_variables_path: Union[str, Path, Sequence[Union[str, Path]]],
        start_date: str,
        end_date: str,
        trajectory_size: int = 1,
        trajectory_dt: int = 1,
        num_samples: Optional[int] = None,
        depths: Optional[Sequence[float]] = None,
        transform: Optional[Callable] = None,
        fill_nans: bool = True,
        *,
        surface_variables: Sequence[str],
        depth_variables: Sequence[str],
        forcing_variables: Sequence[str],
        context_variables: Sequence[str],
        surface_path=None,
        depth_path=None,
        forcing_path=None,
        context_path=None,
    ):
        super().__init__()
        self._validate_arguments(start_date, end_date, trajectory_size, trajectory_dt, num_samples)

        self.trajectory_size = trajectory_size
        self.trajectory_dt = trajectory_dt
        self.transform = transform
        self.fill_nans = fill_nans
        self._sources = {}
        try:
            groups = (
                ("surface", surface_variables, surface_path),
                ("depth", depth_variables, depth_path),
                ("forcing", forcing_variables, forcing_path),
                ("context", context_variables, context_path),
            )
            separate_groups = self._configure_variable_groups(groups)

            # Select all fallback variables together and open each distinct path once.
            selected_sources = []
            if self.variables_to_load:
                selected_sources.append(
                    self._select_source(self.variables_to_load, state_variables_path)
                )
            selected_sources.extend(
                self._select_source(names, source_path) for names, source_path in separate_groups
            )
            fields = []
            for name, field in (
                (name, selected[name])
                for selected in selected_sources
                for name in selected.data_vars
            ):
                field = self._prepare_field(name, field, start_date, end_date, depths)
                fields.append(field.to_dataset())

            # Exact joins prevent silently padding mismatched sources with NaNs.
            self.dataset = xr.merge(fields, join="exact", compat="equals")
            if "time" not in self.dataset.dims:
                raise ValueError("At least one selected variable must have a daily time axis.")
            self._starts = self._build_trajectory_starts(num_samples)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_arguments(start_date, end_date, trajectory_size, trajectory_dt, num_samples):
        """Check daily date bounds and positive integer trajectory/sampling settings."""
        for date in (start_date, end_date):
            assert_date_format(date)
            if len(date) != 10:
                raise ValueError("Daily dates must have YYYY-MM-DD format.")
        if np.datetime64(start_date) > np.datetime64(end_date):
            raise ValueError("start_date must be on or before end_date.")
        for name, value in (
            ("trajectory_size", trajectory_size),
            ("trajectory_dt", trajectory_dt),
            ("num_samples", num_samples),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _preprocess_source(ds: xr.Dataset) -> xr.Dataset:
        """Normalize datetime64 timestamps to days without averaging field values.

        Static sources pass through unchanged. Subdaily timestamps become duplicate
        days and are rejected later by field validation.
        """
        if "time" in ds.dims:
            if not np.issubdtype(ds.time.dtype, np.datetime64):
                raise ValueError("GLORYS time must decode to numpy datetime64.")
            # Daily means may be labelled at noon rather than midnight.
            ds = ds.assign_coords(time=ds.time.values.astype("datetime64[D]"))
        return ds

    def _open_source(self, source) -> xr.Dataset:
        """Open NetCDF files lazily and cache by their resolved, unique file paths.

        A source can be a file, a directory of .nc files, or a sequence of either.
        Reusing a source shares its open handles; callers release them with close().
        """
        entries = [source] if isinstance(source, (str, Path)) else list(source)
        files = []
        for entry in entries:
            entry = Path(entry)
            files.extend(sorted(entry.glob("*.nc")) if entry.is_dir() else [entry])
        key = tuple(sorted({str(p.resolve()) for p in files}))
        if not key:
            raise ValueError(f"No NetCDF files found in {source!r}.")
        if key not in self._sources:
            ds = xr.open_mfdataset(
                list(key),
                combine="by_coords",
                preprocess=self._preprocess_source,
                chunks={},
                data_vars="minimal",
                coords="minimal",
                join="exact",
            )
            self._sources[key] = ds
        return self._sources[key]

    def _configure_variable_groups(self, groups) -> list:
        """Validate named groups and partition their variables by source location.

        Store each group's ordered names and the combined state names. Variables
        without separate paths populate variables_to_load; return nonempty groups
        with separate paths. Reject duplicate names, including across groups.
        """
        seen = set()
        self.variables_to_load = []
        separate_groups = []
        for group, names, source_path in groups:
            if names is None or isinstance(names, str):
                raise ValueError(f"{group}_variables must be a sequence; use [] to disable it.")
            names = list(names)
            for name in names:
                if not isinstance(name, str) or name in seen:
                    raise ValueError("Variable names must be strings unique across all groups.")
                seen.add(name)
            setattr(self, f"{group}_variables", names)
            if source_path is None:
                self.variables_to_load.extend(names)
            elif names:
                separate_groups.append((names, source_path))
        self.state_variables = (
            self.surface_variables + self.depth_variables + self.forcing_variables
        )
        if not seen:
            raise ValueError("Select at least one state or context variable.")
        return separate_groups

    def _select_source(self, names: Sequence[str], source_path) -> xr.Dataset:
        """Select requested variables lazily, raising if any are absent from the source."""
        source = self._open_source(source_path)
        missing = set(names) - set(source.data_vars)
        if missing:
            raise ValueError(f"Variables {sorted(missing)} missing from its source.")
        return source[names]

    @staticmethod
    def _prepare_field(
        name: str,
        field: xr.DataArray,
        start_date: str,
        end_date: str,
        depths: Optional[Sequence[float]],
    ) -> xr.DataArray:
        """Subset dates/depths and validate spatial dimensions and consecutive days.

        Date bounds are inclusive and depth selection uses exact coordinates.
        Static fields need no time axis. Shared grids and dates across fields are
        checked separately when the prepared fields are merged with an exact join.
        """
        if "time" in field.dims:
            field = field.sortby("time").sel(time=slice(start_date, end_date))
        if depths is not None and "depth" in field.dims:
            field = field.sel(depth=list(depths))
        required = {"longitude", "latitude"}
        if not required.issubset(field.dims):
            raise ValueError(f"{name!r} must have longitude and latitude dimensions.")
        extra = set(field.dims) - required - {"time", "depth"}
        if extra:
            raise ValueError(f"Unexpected dimensions for {name!r}: {sorted(extra)}.")
        if "time" in field.dims:
            field = field.sortby("time")
            times = field.time.values.astype("datetime64[D]")
            if (
                not len(times)
                or np.isnat(times).any()
                or (len(times) > 1 and not np.all(np.diff(times) == np.timedelta64(1, "D")))
            ):
                raise ValueError(f"{name!r} must have one field per consecutive day.")
        return field

    def _build_trajectory_starts(self, num_samples: Optional[int]) -> np.ndarray:
        """Return all or evenly spaced complete trajectory starts on the daily axis.

        Account for the within-trajectory stride without thinning the source dates.
        Raise if no complete trajectory fits or too many starts are requested.
        """
        available = self.dataset.sizes["time"] - (self.trajectory_size - 1) * self.trajectory_dt
        if available < 1:
            raise ValueError("Date range is too short for the requested trajectory.")
        if num_samples is not None and num_samples > available:
            raise ValueError(f"num_samples exceeds the {available} valid trajectory starts.")
        return (
            np.arange(available)
            if num_samples is None
            else np.linspace(0, available - 1, num_samples, dtype=int)
        )

    def __len__(self) -> int:
        """Number of complete trajectories, including their daily stride."""
        return len(self._starts)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return a trajectory; adjacent unsampled start indices are one day apart."""
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        start = int(self._starts[idx])
        return self._load_sub_trajectory(start, start + self.trajectory_size * self.trajectory_dt)

    def _dataset_to_tensor(self, sample: xr.Dataset, variables: Sequence[str]) -> Tensor:
        """Load a selected trajectory as float32 (time, channels, longitude, latitude).

        Broadcast static maps across sample dates and flatten variables/depths
        into channels. Empty selections return batchable zero-channel tensors.
        """
        if not variables:
            return torch.empty((
                sample.sizes["time"],
                0,
                sample.sizes["longitude"],
                sample.sizes["latitude"],
            ))
        # Broadcast only each selected trajectory, not static maps over the full archive.
        fields = {}
        for name in variables:
            field = sample[name]
            if "time" not in field.dims:
                field = field.expand_dims(time=sample.time)
            fields[name] = field
        stacked = (
            xr.Dataset(fields)
            .to_stacked_array(new_dim="z_total", sample_dims=("time", "longitude", "latitude"))
            .transpose("time", "z_total", "longitude", "latitude")
        )
        return torch.tensor(stacked.load().data, dtype=torch.float32)

    def _load_sub_trajectory(
        self, step_start: int, step_end: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Read a strided trajectory, transform state/context, then optionally fill NaNs.

        Return state, context, and per-day (year, month, day, zero-hour) features.
        step_start and step_end are positional bounds, with the end excluded.
        """
        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            sample = self.dataset.isel(time=slice(step_start, step_end, self.trajectory_dt))
            if sample.sizes["time"] != self.trajectory_size:
                raise ValueError("Incomplete daily trajectory.")
            time = torch.stack([get_date_features(date) for date in sample.time.values])
            state = self._dataset_to_tensor(sample, self.state_variables)
            context = self._dataset_to_tensor(sample, self.context_variables)
            if self.transform is not None:
                state, context = self.transform(state, context)
            if self.fill_nans:
                state = torch.nan_to_num(state, nan=DATA_NAN_VALUE)
                context = torch.nan_to_num(context, nan=DATA_NAN_VALUE)
        return state, context, time

    def close(self) -> None:
        """Close the backing NetCDF sources when the dataset is no longer in use."""
        for source in self._sources.values():
            source.close()


class LatentBlanketDataset(Dataset):
    r"""Dataset for latent blankets.
    This serves as a wrapper around latent encoded data in h5 format.
    Arguments:
        path: Path to the h5 dataset file.
        start_date: Start date of the data split (format: 'YYYY-MM-DD').
        end_date: End date of the data split (format: 'YYYY-MM-DD').
        blanket_size: Size of the blankets.
        start_hour: Start hour of the data split (0-23).
        end_hour: End hour of the data split (0-23).
        standardize: Whether to standardize the data or not.
        stride: Spacing between blanket elements.
        noise_level: Level of noise to add to the data (before standardization).
    """

    def __init__(
        self,
        path: Path,
        start_date: str,
        end_date: str,
        blanket_size: int,
        start_hour: int = 0,
        end_hour: int = 23,
        standardize: bool = True,
        stride: int = 1,
        noise_level: float = 0.0,
    ):
        super().__init__()

        self.h5_file = h5py.File(path, "r")
        self.latents = self.h5_file["latents"]
        self.dates = torch.as_tensor(self.h5_file["dates"][...])

        self.latents_folder = path.parent
        self.blanket_size = blanket_size
        self.standardize = standardize
        self.stride = stride
        self.noise_level = noise_level

        start_date = [int(x) for x in start_date.split("-")] + [start_hour]
        end_date = [int(x) for x in end_date.split("-")] + [end_hour]
        start_date, end_date = torch.as_tensor(start_date), torch.as_tensor(end_date)
        try:
            self.start_idx = (self.dates == start_date).all(dim=-1).nonzero().item()
            self.end_idx = (self.dates == end_date).all(dim=-1).nonzero().item() + 1
        except RuntimeError as e:
            raise ValueError("Start or end date not found in the dataset.") from e

        stats_path = self.latents_folder / "stats.pth"
        if standardize:
            if stats_path.exists():
                stats = safe_load(stats_path)
                self.mean = stats["mean"]
                self.std = stats["std"]

                if self.noise_level > 0:
                    self.std = torch.sqrt(torch.as_tensor(self.noise_level) ** 2 + self.std**2)
            else:
                raise ValueError("Statistics are not computed.")

    def get_stats(self):
        if not self.standardize:
            return None, None

        assert self.mean is not None, "Statistics are not computed."

        return self.mean, self.std

    def get_dim(self):
        return self.latents[0].shape[-1]

    def __len__(self):
        return self.end_idx - self.start_idx - (self.blanket_size - 1) * self.stride

    def __getitem__(self, idx):
        idx = self.start_idx + idx
        select = slice(idx, idx + self.blanket_size * self.stride, self.stride)
        blanket = torch.as_tensor(self.latents[select])
        date = self.dates[select]

        if self.noise_level > 0.0:
            blanket = blanket + torch.randn_like(blanket) * self.noise_level

        if self.standardize:
            blanket = (blanket - self.mean) / self.std

        return blanket, date


def get_datasets(**kwargs) -> Tuple[ERA5Dataset, ERA5Dataset, ERA5Dataset]:
    r"""Returns the training, validation, and test datasets.

    Splits:
        Training: 1999-01-01 to 2017-12-31
        Validation: 2018-01-01 to 2018-12-31
        Test: 2019-01-01 to 2019-12-31

    Arguments:
        kwargs: Keyword arguments passed to :class:`ERA5Dataset`.
    """

    splits = [
        DATASET_DATES_TRAINING,
        DATASET_DATES_VALIDATION,
        DATASET_DATES_TEST,
    ]

    datasets = [
        ERA5Dataset(
            path=PATH_ERA5,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        for start_date, end_date in splits
    ]

    return tuple(datasets)


def get_toy_datasets(
    variables: Optional[Sequence[str]] = None,
    **kwargs,
) -> Tuple[ERA5Dataset, ERA5Dataset, ERA5Dataset]:
    r"""Returns the toy training, validation, and test datasets.

    Variables:
        By default, only the surface and atmosphere temperature fields.

    Splits:
        Training: 2017-07-01 to 2017-09-30
        Validation: 2018-07-01 to 2018-09-30
        Test: 2019-07-01 to 2019-09-30

    Arguments:
        variables: Variable names to retain from the dataset.
        kwargs: Keyword arguments passed to :class:`ERA5Dataset`.
    """

    if variables is None:
        variables = TOY_DATASET_VARIABLES

    splits = [
        TOY_DATASET_DATES_TRAINING,
        TOY_DATASET_DATES_VALIDATION,
        TOY_DATASET_DATES_TEST,
    ]

    datasets = [
        ERA5Dataset(
            path=PATH_ERA5,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            **kwargs,
        )
        for start_date, end_date in splits
    ]

    return tuple(datasets)
