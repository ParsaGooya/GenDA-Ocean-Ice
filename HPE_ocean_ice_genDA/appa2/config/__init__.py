r"""Global paths and configuration helpers."""

__all__ = [
    "compose",
]

from pathlib import Path

from .hydra import compose

PROJECT = Path(__file__).resolve().parents[2]
DATA = Path("/lus/scratch/PGooya/data")

PATH_AE = PROJECT / "autoencoders"

PATH_GLORYS = DATA / "glorys_daily"
PATH_GLORYS_FORING = DATA / "glorys_daily"

PATH_GLORYS_STAT_FORCING = DATA / "glorys_daily" / "stats"
PATH_GLORYS_STAT_STATE = DATA / "glorys_daily" / "stats"

PATH_GLORYS_BATHYMETRY = DATA / "glorys_daily" / "bathymetry_glorys.nc"
