r"""Datasets."""

import os
from collections import OrderedDict

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


NetCDFSource = Union[str, Path, Sequence[Union[str, Path]]]
GlorysSample = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]


class GlorysDataset(Dataset):
    """Read daily variable/year NetCDF archives using a metadata-only time index.

    Files may be supplied individually, as sequences, or as directories searched
    recursively. Each group path defaults to state_variables_path. Coordinates
    must be named time, longitude, latitude and optionally depth. Dates are
    inclusive; every dynamic variable must cover every requested day exactly once.
    Noon-labelled daily means are normalized to days; subdaily duplicates fail.

    Return (state, context, time), appending forcing when forcing_variables is
    nonempty. Field tensors are float32 (time, channels, latitude, longitude).
    State channels contain surface variables then depth variables, flattening
    depth in variable order. Forcing retains its own native spatial grid. Context
    shares the ocean grid. Time contains (year, month, day, hour=0).

    transform accepts/returns (state, context); forcing_transform independently
    transforms forcing. NaNs are filled after transforms when fill_nans=True.
    trajectory_size and trajectory_dt specify sample length and daily stride;
    num_samples selects evenly spaced valid starts. depths selects exact levels.
    max_open_files bounds the lazy file-handle cache separately in each worker.
    Only selected slices are materialized, with no Dask archive graph.
    """

    surface_variables: list[str]
    depth_variables: list[str]
    forcing_variables: list[str]
    context_variables: list[str]

    def __init__(
        self, 
        state_variables_path: NetCDFSource, 
        start_date: str, 
        end_date: str,
        trajectory_size: int = 1, 
        trajectory_dt: int = 1,
        num_samples: Optional[int] = None, 
        depths: Optional[Sequence[float]] = None,
        transform: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
        fill_nans: bool = True, 
        *, 
        surface_variables: Sequence[str],
        depth_variables: Sequence[str], 
        forcing_variables: Sequence[str],
        context_variables: Sequence[str], 
        surface_path: Optional[NetCDFSource] = None,
        depth_path: Optional[NetCDFSource] = None, 
        forcing_path: Optional[NetCDFSource] = None,
        context_path: Optional[NetCDFSource] = None,
        forcing_transform: Optional[Callable[[Tensor], Tensor]] = None,
        max_open_files: int = 16,
    ) -> None:
        """Scan metadata, validate coverage/grids, and construct trajectory starts."""

        super().__init__()

        for date in (start_date, end_date):
            assert_date_format(date)
            if len(date) != 10:
                raise ValueError("Daily dates must have YYYY-MM-DD format.")
        if np.datetime64(start_date) > np.datetime64(end_date):
            raise ValueError("start_date must be on or before end_date.")
        for name, value in (("trajectory_size", trajectory_size), ("trajectory_dt", trajectory_dt),
                            ("num_samples", num_samples), ("max_open_files", max_open_files)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f"{name} must be a positive integer.")
            
        self.trajectory_size = trajectory_size
        self.trajectory_dt = trajectory_dt
        self.depths = None if depths is None else list(depths)
        self.transform = transform
        self.forcing_transform = forcing_transform
        self.fill_nans = fill_nans
        self.max_open_files = max_open_files
        self._files: OrderedDict[str, xr.Dataset] = OrderedDict()
        self._pid = os.getpid()
        self.dates = np.arange(np.datetime64(start_date, 'D'),
                               np.datetime64(end_date, 'D') + np.timedelta64(1, 'D'))
        self.temporal_index: dict[str, dict[np.datetime64, tuple[str, int]]] = {} #VAR: Date -> (path, index)
        self._static: dict[str, str] = {}
        self._coords: dict[str, dict[str, np.ndarray]] = {}
        self.variables_to_load_from_shared_path: list[str] = []
        routes: dict[str, list[str]] = {}
        seen: set[str] = set()
        for group, names, source in (
            ('surface', surface_variables, surface_path), 
            ('depth', depth_variables, depth_path),
            ('forcing', forcing_variables, forcing_path), 
            ('context', context_variables, context_path),
        ):
            if names is None or isinstance(names, str):
                raise ValueError(f"{group}_variables must be a sequence; use [] to disable it.")
            ordered = list(names)
            for name in ordered:
                if not isinstance(name, str) or name in seen:
                    raise ValueError("Variable names must be strings unique across all groups.")
                seen.add(name)
            setattr(self, f'{group}_variables', ordered)
            if not ordered:
                continue
            if source is None:
                self.variables_to_load_from_shared_path.extend(ordered)
            for path in self._discover_files(state_variables_path if source is None else source):
                routes.setdefault(path, []).extend(ordered)
                
        self.state_variables = list(surface_variables) + list(depth_variables)
        if not seen:
            raise ValueError("Select at least one state or context variable.")
        for path, names in routes.items():
            self._index_file(path, list(dict.fromkeys(names)))
        for name in seen:
            if name not in self._coords:
                raise ValueError(f"Variable {name!r} missing from its source.")
            if name not in self._static:
                missing = set(self.dates) - self.temporal_index[name].keys()
                if missing:
                    raise ValueError(f"{name!r} missing {len(missing)} requested daily dates; first: {min(missing)}.")
        ocean = self.state_variables + self.context_variables
        self._ocean_coords = self._validate_grid(ocean) if ocean else self._coords[self.forcing_variables[0]]
        self._forcing_coords = self._validate_grid(self.forcing_variables) if self.forcing_variables else self._ocean_coords
        available = len(self.dates) - (trajectory_size - 1) * trajectory_dt
        if available < 1:
            raise ValueError("Date range is too short for the requested trajectory.")
        if num_samples is not None and num_samples > available:
            raise ValueError(f"num_samples exceeds the {available} valid trajectory starts.")
        self._starts = np.arange(available) if num_samples is None else np.linspace(0, available - 1, num_samples, dtype=int)

    @staticmethod
    def _discover_files(source: NetCDFSource) -> list[str]:
        """Resolve a file/sequence or recursive variable/year directory to files."""
        entries = [source] if isinstance(source, (str, Path)) else source
        files: set[str] = set()
        for entry in entries:
            path = Path(entry)
            files.update(str(p.resolve()) for p in (path.rglob('*.nc') if path.is_dir() else [path]))
        if not files:
            raise ValueError(f"No NetCDF files found in {source!r}.")
        return sorted(files)

    def _index_file(self, path: str, names: Sequence[str]) -> None:
        """Read coordinates only; record each date's file and positional offset."""
        with xr.open_dataset(path, chunks=None, cache=False) as ds:
            for name in names:
                if name not in ds.data_vars:
                    continue
                field = ds[name]
                if not {'longitude', 'latitude'}.issubset(field.dims):
                    raise ValueError(f"{name!r} must have longitude and latitude dimensions.")
                if set(field.dims) - {'time', 'depth', 'longitude', 'latitude'}:
                    raise ValueError(f"Unexpected dimensions for {name!r}.")
                if self.depths is not None and 'depth' in field.dims:
                    field = field.sel(depth=self.depths)
                coords = {d: field[d].values.copy() for d in ('longitude', 'latitude', 'depth') if d in field.dims}
                if name in self._coords and (
                    coords.keys() != self._coords[name].keys() or any(
                        not np.array_equal(v, self._coords[name][d])
                        for d, v in coords.items()
                    )
                ):
                    raise ValueError(f"Grid/depth coordinates change across files for {name!r}.")
                self._coords[name] = coords
                if 'time' not in field.dims:
                    if name in self.temporal_index or name in self._static:
                        raise ValueError(f"Ambiguous repeated static source for {name!r}.")
                    self._static[name] = path
                    continue
                if name in self._static:
                    raise ValueError(f"Mixed static and dynamic sources for {name!r}.")
                if not np.issubdtype(field.time.dtype, np.datetime64):
                    raise ValueError("GLORYS time must decode to numpy datetime64.")
                dates = field.time.values.astype('datetime64[D]')
                if np.isnat(dates).any() or len(np.unique(dates)) != len(dates):
                    raise ValueError(f"{name!r} must have at most one field per day.")
                index = self.temporal_index.setdefault(name, {})
                for position, date in enumerate(dates):
                    if self.dates[0] <= date <= self.dates[-1]:
                        if date in index:
                            raise ValueError(f"Duplicate date {date} for {name!r}.")
                        index[date] = (path, position)

    def _validate_grid(self, names: Sequence[str]) -> dict[str, np.ndarray]:
        """Require identical horizontal coordinates within one output group."""
        coords = self._coords[names[0]]
        for name in names[1:]:
            if any(not np.array_equal(coords[d], self._coords[name][d]) for d in ('longitude', 'latitude')):
                raise ValueError(f"Horizontal grid mismatch for {name!r}.")
        return coords

    def _open_file(self, path: str) -> xr.Dataset:
        """Get a process-local lazy handle, evicting least recently used files."""

        ##each worker reuses its own open files, avoiding opening the file for every sample while keeping file handles separate between processes.
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if path in self._files:
            self._files.move_to_end(path)
        else:
            self._files[path] = xr.open_dataset(path, chunks=None, cache=False)
            while len(self._files) > self.max_open_files:
                _, old = self._files.popitem(last=False)
                old.close()
        return self._files[path]

    def _load_variables(self, 
                        names: Sequence[str], dates: np.ndarray,
                        coords: dict[str, np.ndarray]) -> Tensor:
        
        """Read only indexed sample slices; preserve variable/depth channel order."""
        channels: list[np.ndarray] = []
        for name in names:
            requests: dict[str, list[tuple[int, int]]] = {} ##PATH -> List of (output_position, position_in_file)
            if name in self._static:
                requests[self._static[name]] = [(i, 0) for i in range(len(dates))]
            else:
                for output_position, date in enumerate(dates):
                    path, position = self.temporal_index[name][date]
                    requests.setdefault(path, []).append((output_position, position))
            values: Optional[np.ndarray] = None
            for path, positions in requests.items():
                field = self._open_file(path)[name]
                if self.depths is not None and 'depth' in field.dims:
                    field = field.sel(depth=self.depths)
                if 'time' in field.dims:
                    field = field.isel(time=[position for _, position in positions])
                else:
                    field = field.expand_dims(time=len(positions))
                order = ['time'] + (['depth'] if 'depth' in field.dims else []) + ['latitude', 'longitude']
                array = np.asarray(field.transpose(*order).load().values, dtype=np.float32)
                array = array.reshape(len(positions), -1, len(coords['latitude']), len(coords['longitude']))
                if values is None:
                    values = np.empty((len(dates), *array.shape[1:]), dtype=np.float32)
                values[[i for i, _ in positions]] = array
            if values is None:
                raise RuntimeError(f"No indexed values for {name!r}.")
            channels.append(values)
        if not channels:
            return torch.empty((len(dates), 0, len(coords['latitude']), len(coords['longitude'])), dtype=torch.float32)
        return torch.from_numpy(np.concatenate(channels, axis=1))

    def __len__(self) -> int:
        """Return the number of complete trajectory starts."""
        return len(self._starts)

    def __getitem__(self, idx: int) -> GlorysSample:
        """Load one trajectory, with forcing appended only when configured."""
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        start = int(self._starts[idx])
        dates = self.dates[start:start + self.trajectory_size * self.trajectory_dt:self.trajectory_dt]
        state = self._load_variables(self.state_variables, dates, self._ocean_coords)
        context = self._load_variables(self.context_variables, dates, self._ocean_coords)
        time = torch.stack([get_date_features(date) for date in dates])
        if self.transform is not None:
            state, context = self.transform(state, context)
        if self.fill_nans:
            state = torch.nan_to_num(state, nan=DATA_NAN_VALUE)
            context = torch.nan_to_num(context, nan=DATA_NAN_VALUE)
        if not self.forcing_variables:
            return state, context, time
        forcing = self._load_variables(self.forcing_variables, dates, self._forcing_coords)
        if self.forcing_transform is not None:
            forcing = self.forcing_transform(forcing)
        if self.fill_nans:
            forcing = torch.nan_to_num(forcing, nan=DATA_NAN_VALUE)
        return state, context, time, forcing

    def close(self) -> None:
        """Release cached handles in this process; subsequent reads reopen files."""
        for source in self._files.values():
            source.close()
        self._files.clear()

    def __getstate__(self) -> dict[str, object]:
        """Exclude open handles when spawning DataLoader workers."""
        state = self.__dict__.copy()
        state['_files'] = OrderedDict()
        state['_pid'] = None
        return state

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
