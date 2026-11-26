# GlorysDataset internal attributes

Reference: `appa2/data/datasets.py`, class `GlorysDataset`.

The dataset retains metadata and file locations during initialization. Physical
field values are loaded only when a sample is requested. State, context, and
forcing tensors use `(time, channels, latitude, longitude)`; the DataLoader adds
a leading batch dimension. Flattening latitude/longitude makes longitude vary
fastest, matching `create_glorys()` node order when coordinate values/order agree.

| Attribute | Type | Purpose |
|---|---|---|
| `dates` | `np.ndarray` of `datetime64[D]` | Complete daily axis between the inclusive requested start and end dates. |
| `temporal_index` | `dict[str, dict[np.datetime64, tuple[str, int]]]` | Maps each dynamic variable and requested date to an absolute file path and its zero-based time position inside that file. |
| `_static` | `dict[str, str]` | Maps variables without a time dimension to their file paths; these fields are repeated across sample dates. |
| `_coords` | `dict[str, dict[str, np.ndarray]]` | Stores each variable's longitude, latitude, and optional selected depth coordinates, used for validation and output shapes. |
| `_ocean_coords` | `dict[str, np.ndarray]` | Reference coordinates for state/context outputs. Their horizontal grids must match. For forcing-only datasets, falls back to the first forcing variable's coordinates. |
| `_forcing_coords` | `dict[str, np.ndarray]` | Reference coordinates for forcing outputs, which may differ from the ocean grid. Falls back to `_ocean_coords` when forcing is disabled. |
| `_starts` | `np.ndarray` of integers | Positions in `dates` where selected complete trajectories begin. Dataset indices address this array, not necessarily every day when `num_samples` is used. |
| `trajectory_size` | `int` | Number of time steps returned in each sample. |
| `trajectory_dt` | `int` | Separation in days between successive fields within a trajectory. |
| `surface_variables` | `list[str]` | Ordered surface-state variable names. |
| `depth_variables` | `list[str]` | Ordered depth-dependent state names; selected depths are flattened into channels. |
| `state_variables` | `list[str]` | Surface names followed by depth names; determines state channel order and excludes forcing/context. |
| `forcing_variables` | `list[str]` | Ordered forcing names. A nonempty list enables a separate fourth output tensor. |
| `context_variables` | `list[str]` | Ordered ocean-context names, returned separately from state. |
| `variables_to_load_from_shared_path` | `list[str]` | Records variables whose group path was omitted and therefore uses `state_variables_path`. Routing bookkeeping, not a second temporal index. |
| `depths` | `list[float]` or `None` | Exact depth coordinates to select; `None` retains all available depths. |
| `transform` | callable or `None` | Accepts and returns `(state, forcing)` before missing-value filling. When forcing is disabled, the callable receives a zero-channel forcing tensor. |
| `fill_nans` | `bool` | Enables `torch.nan_to_num` on output fields after transforms, replacing NaNs with `DATA_NAN_VALUE`; infinities also follow PyTorch's default replacement behavior. |
| `_files` | `OrderedDict[str, xr.Dataset]` | Process-local cache of open lazy datasets, ordered from least to most recently used. Caches handles rather than full annual arrays. |
| `max_open_files` | `int` | Maximum number of cached file entries per process; older entries are closed and evicted when exceeded. |
| `_pid` | `int` or `None` | Identifies the process owning the cache. A worker process change clears inherited handles; serialized copies reset this to `None`. |

`routes`, `seen`, and `available` are temporary constructor variables, not stored
attributes. The original source-path arguments are represented by resolved paths
in the indexes rather than retained as separate attributes. The temporal index is
in memory and is rebuilt for each new dataset instance.

This documentation was produced with the assistance of AI.

## Comparing with GlorysDataset2

Import `GlorysDataset2` from `appa2.data.datasets` and pass exactly the same
arguments as `GlorysDataset`. Both return identical sample structures, daily
trajectory starts, float32 channel ordering, depth selection, transforms, and
native forcing grids.

| Loader | Archive representation | Sample loading |
|---|---|---|
| `GlorysDataset` | Metadata-only date-to-file index and per-worker LRU file handles | Groups requested dates by file and reads selected slices without Dask |
| `GlorysDataset2` | Lazy xarray DataArrays backed by Dask, with one-day time chunks | Selects the trajectory from concatenated/sorted arrays and computes the selected graph |

The second loader mirrors ERA5's lazy slicing/materialization approach while
reading the GLORYS NetCDF layout. Its `dataset` attribute is a dictionary keyed
by variable, so forcing coordinates never align or expand the ocean grid.
`max_open_files` is accepted but does not bound xarray's file manager cache.
Neither loader preloads the full archive. `GlorysDataset2` does retain an archive
Dask graph; compare initialization time as well as repeated DataLoader throughput
and memory with identical worker, batch, and shuffle settings. No performance
winner is assumed without measurements on the actual archive.
