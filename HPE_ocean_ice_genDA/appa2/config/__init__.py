r"""Global paths and configuration helpers."""

__all__ = [
    "compose",
]

from pathlib import Path

from .hydra import compose

PROJECT = Path(__file__).resolve().parents[2]

PATH_AE = PROJECT / "autoencoders"

PATH_GLORYS = PROJECT / "data" / "era5_1993-2021-1h-1440x721.zarr"
PATH_GLORYS_FORING = PROJECT / "data" / "era5_1993-2021-1h-1440x721.zarr"

PATH_GLORYS_STAT_FORCING = PROJECT / "data" / "stats_era5_1993-2021-1h-1440x721.zarr"
PATH_GLORYS_STAT_STATE = PROJECT / "data" / "stats_era5_1993-2021-1h-1440x721.zarr"

PATH_GLORYS_BATHYMETRY = PROJECT / "data" / "masks_era5_1993-2021-1h-1440x721.zarr"
