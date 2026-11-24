
from appa2.data.const import (
    GLORYS_DATASET_DATES_TRAINING,
    GLORYS_DATASET_DATES_VALIDATION,
    GLORYS_DEPTH_LEVELS,
    GLORYS_DEPTH_DIM
)
from appa2.config import (
    PATH_GLORYS,
    PATH_GLORYS_FORING,
    PATH_GLORYS_STAT_FORCING,
    PATH_GLORYS_STAT_STATE,
    PATH_GLORYS_BATHYMETRY,
)
from appa2.grid import SUPPORTED_GRIDS
from appa2.data.datasets import GlorysDataset, NetCDFSource
from typing import Optional
import dataclasses

from torch import Tensor

@dataclasses.dataclass
class dataset_config:

    grid: SUPPORTED_GRIDS
    state_variables_path: Optional[NetCDFSource] | None
    time_splits: list[tuple[str]]
    depth_levels: list[float]
    rearange_protocol: str = None
    rearange_protocol_inverse: str = None
    surface_path: Optional[NetCDFSource] = None
    depth_path: Optional[NetCDFSource] = None
    forcing_path: Optional[NetCDFSource] = None
    bathymetry_path: Optional[NetCDFSource] = None
    path_stat_state:  Optional[NetCDFSource] = None
    path_stat_forcing:  Optional[NetCDFSource] = None
    depth_dim_name: Optional[str] = 'depth'

    def __post_init__(self):
        self.grid = self.grid.lower()
        if self.state_variables_path is None:
            if all([
                self.surface_path is None,
                self.depth_path is None,
            ]):
                raise ValueError(
                    "When no state_variables_path is provided " 
                    "at least one of surface_path or depth_path "
                    "must be supplied."
                )
        if self.bathymetry_path is not None:
            raise NotImplementedError
            # self.bathymetry_mask = torch.load()
        else:
            self.bathymetry_mask = None

        if self.grid in ["glorys", "N_320"]:
            self.dataset = GlorysDataset
        else:
            self.dataset = None

def get_dataset_specs(starting_grid: SUPPORTED_GRIDS):

    if starting_grid.lower() == "glorys":
        
        return dataset_config(
            grid = starting_grid,
            state_variables_path = PATH_GLORYS,
            time_splits = [
                GLORYS_DATASET_DATES_TRAINING,
                GLORYS_DATASET_DATES_VALIDATION,
                GLORYS_DATASET_DATES_VALIDATION,
            ],
            depth_levels= GLORYS_DEPTH_LEVELS,
            rearange_protocol= "B T Z Lat Lon -> (B T) (Lat Lon) Z",
            rearange_protocol_inverse= "(B T) (Lat Lon) Z -> B T Z Lat Lon",
            # bathymetry_path=PATH_GLORYS_BATHYMETRY,
            forcing_path = PATH_GLORYS_FORING,
            path_stat_state = PATH_GLORYS_STAT_STATE,
            path_stat_forcing= PATH_GLORYS_STAT_FORCING,
            depth_dim_name=GLORYS_DEPTH_DIM,
        )
    elif starting_grid == "riops":
        raise NotImplementedError("riops dataset is not implemented yet.")
    else:
        raise ValueError(f"Unsupported starting grid: {starting_grid}")