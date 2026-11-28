r"""Global paths and configuration helpers."""

__all__ = [
    "compose",
]

from pathlib import Path

from .hydra import compose

PROJECT = Path(__file__).resolve().parents[2]
DATA = Path("/lus/scratch/PGooya/data")

PATH_AE = PROJECT / "autoencoders"

PATH_GLORYS = DATA / "glorys/glorys_daily_25"
PATH_GLORYS_FORING = DATA / "era5/era5_daily_weatherbench2"
PATH_GLORYS_CONTEXT = DATA / "glorys/static_25"

PATH_GLORYS_STAT_FORCING = DATA / "glorys_daily" / "stats"
PATH_GLORYS_STAT_STATE = DATA / "glorys_daily" / "stats"

PATH_GLORYS_BATHYMETRY = DATA / "glorys_daily" / "bathymetry_glorys.nc"
