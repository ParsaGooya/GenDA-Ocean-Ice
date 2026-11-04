r"""Custom Data transformations."""

import torch
import xarray as xr

from pathlib import Path
from torch import Tensor
from typing import Optional, Sequence


class StandardizeTransform:
    r"""Creates a standardization tranform with pre-computed statistics.

    Arguments:
        path: Path to the Zarr dataset containing mean and standard deviation statistics.
        state_variables: Variables present in the non-standardized dataset for the state. if None, all variables are used.
        context_variables: Variables present in the non-standardized dataset for the context. If None, no context is used.
        levels: Atmospheric levels to select from the dataset. If None, all levels are used.
    """

    def __init__(
        self,
        path: Path,
        state_variables: Optional[Sequence[str]] = None,
        context_variables: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[str]] = None,
    ):
        stats = xr.open_zarr(path)

        if levels is not None:
            stats = stats.sel(level=levels)

        self.state_mean, self.state_std = self._stats_zarr_to_tensor(stats, state_variables)
        if context_variables is not None:
            self.context_mean, self.context_std = self._stats_zarr_to_tensor(
                stats, context_variables
            )
        else:
            self.context_mean = None
            self.context_std = None

    def _stats_zarr_to_tensor(
        self, stats: xr.DataArray, variables: Optional[Sequence[str]] = None
    ) -> Tensor:
        r"""Converts Zarr statistics to a tensor.

        Args:
            stats: Zarr dataset containing mean and standard deviation statistics.
            variables: Variables present in the non-standardized dataset.

        Returns:
            A tuple of tensors containing mean and standard deviation statistics.
        """
        if variables is not None:
            stats = stats[variables]

        era5_stats_array = stats.to_stacked_array(
            new_dim="z_total",
            sample_dims=("statistic",),
        ).transpose("z_total", ...)
        era5_stats_tensor = torch.tensor(era5_stats_array.load().data)
        era5_stats_tensor = era5_stats_tensor[None, :, :, None, None]  # [1, n_vars, 2, 1, 1]

        return era5_stats_tensor[:, :, 0], era5_stats_tensor[:, :, 1]

    def __call__(self, state: Tensor, context: Tensor) -> Tensor:
        r"""Applies standardization to the input tensor.

        Args:
            state: Input tensor representing the state, of shape :math:`(T, Z, Lat, Lon)`.
            context: Input tensor representing the context, of shape :math:`(T, Z, Lat, Lon)`.

        Returns:
            A tuple of tensors representing the standardized state and context, both of shape :math:`(T, Z, Lat, Lon)`.
        """
        state = (state - self.state_mean) / self.state_std
        if self.context_mean is not None:
            context = (context - self.context_mean) / self.context_std

        return state, context

    def unstandardize(self, state: Tensor, context: Tensor = None) -> Tensor:
        r"""Applies unstandardization to the input tensor.

        Args:
            state: Input tensor representing the state, of shape :math:`(T, Z, Lat, Lon)`.
            context: Input tensor representing the context, of shape :math:`(T, Z, Lat, Lon)`.

        Returns:
            A tuple of tensors representing the unstandardized state and context, both of shape :math:`(T, Z, Lat, Lon)`.
        """
        state = (state * self.state_std) + self.state_mean
        if context is not None and self.context_mean is not None:
            context = (context * self.context_std) + self.context_mean

        return state, context
