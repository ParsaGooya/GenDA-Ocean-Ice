"""Physical/channel weights and masked losses for graph autoencoder training."""

import math
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor, nn

from appa2.data.const import GLORYS_DEPTH_LEVELS
from appa2.grid import SUPPORTED_GRIDS, create_original_grid

def _coordinates(grid_coords: Tensor) -> Tensor:
    """Validate latitude/longitude coordinates without changing node order."""
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 2 or not len(grid_coords):
        raise ValueError("grid_coords must have nonempty shape (N, 2): latitude, longitude.")
    coords = grid_coords.to(dtype=torch.float32)
    if not torch.isfinite(coords).all() or (coords[:, 0].abs() > 90).any():
        raise ValueError("Coordinates must be finite, with latitude in [-90, 90].")
    return coords


def get_area_weights(grid: SUPPORTED_GRIDS) -> Tensor:
    """Return cosine-latitude relative areas, shape (N,), in input node order.
    Expects input coords of shape (N,2) being latitude and longitude per node.

    Appropriate for regular, equally spaced latitude/longitude grids such as
    create_N320() and create_glorys(). This helper does not infer grid type.
    For ORG, curvilinear riops or other unequal grids, supply actual cell areas
    to get_weights instead. No normalization or artificial pole offset is used.
    """
    
    coords, _ =  create_original_grid(grid)
    # Double precision avoids negative float32 cosine values at the poles.
    areas = torch.deg2rad(coords[:, 0].double()).cos().clamp_min(0)
    areas = torch.where(coords[:, 0].abs() == 90, 0., areas)
    return areas.float()


def get_weights(
    grid: SUPPORTED_GRIDS,
    surface_variables: Sequence[str],
    depth_variables: Sequence[str],
    area_per_grid: Optional[Tensor] = None,
    level_weighting: bool = False,
    bathymetry_mask: Optional[Tensor] = None,
    surface_variable_weights: Optional[Sequence[float]] = None,
    depth_variable_weights: Optional[Sequence[float]] = None,
    levels: Optional[Sequence[float]] = GLORYS_DEPTH_LEVELS,
    level_weighting_method: Literal['grid_size', 'exponential_decay'] = 'grid_size',
    decay_scale: Optional[float] = None,
) -> Tensor:
    """Build unnormalized float32 weights shaped (channels, N).

    Channel order matches datasets in XXXX: surface variables, then all selected
    levels of each depth variable. Variable weights default to one. Horizontal
    area_per_grid has shape (N,), defaults to one, and may use any consistent
    area units. Output stays on grid_coords.device and retains node order.

    levels are depth centres in metres, positive downward, in dataset channel
    order. grid_size methods estimates layer thickness from midpoint boundaries, with
    a zero sea-surface boundary and half-spacing extrapolation at the bottom;
    it requires at least two strictly increasing depths. Thickness weights are
    divided by their maximum, so the thickest layer has weight one.
    exponential_decay uses exp(-depth / decay_scale), requiring a finite positive scale in metres.
    Disabled level weighting uses ones. Surface level factors are always one.

    bathymetry_mask is Boolean (len(levels), N), True for valid wet cells.
    It is repeated for each depth variable; the shallowest-level mask is used
    for surface variables. Without depth centres, a surface-only mask is (1, N).
    Areas, variable weights and vertical factors are multiplied without global
    normalization; AELoss normalizes over included weights at runtime.
    """
    coords, _ = create_original_grid(grid)
    device, n = coords.device, len(coords)

    ns, nd = len(surface_variables), len(depth_variables)
    if not ns + nd:
        raise ValueError("Select at least one surface or depth variable.")
    z = torch.as_tensor([] if levels is None else levels, dtype=torch.float32, device=device)
    if z.ndim != 1 or not torch.isfinite(z).all() or (z < 0).any():
        raise ValueError("levels must be a finite, nonnegative one-dimensional sequence.")
    if nd and not len(z):
        raise ValueError("Depth variables require nonempty levels.")

    def variable_weights(values: Optional[Sequence[float]], count: int) -> Tensor:
        """Validate one nonnegative multiplier per variable."""
        result = torch.ones(count, device=device) if values is None else torch.as_tensor(values, dtype=torch.float32, device=device)
        if result.shape != (count,) or not torch.isfinite(result).all() or (result < 0).any():
            raise ValueError("Variable weights must be finite, nonnegative and match their variable count.")
        return result

    surface = variable_weights(surface_variable_weights, ns)
    depth = variable_weights(depth_variable_weights, nd)
    vertical = torch.ones_like(z)
    if level_weighting and nd:
        if level_weighting_method == 'grid_size':
            if len(z) < 2 or not (z.diff() > 0).all():
                raise ValueError("grid_size requires at least two strictly increasing depth centres.")
            boundaries = torch.cat([z.new_zeros(1), (z[:-1] + z[1:]) / 2,
                                    (z[-1] + (z[-1] - z[-2]) / 2).reshape(1)])
            vertical = boundaries.diff()
            vertical = vertical / vertical.max()
        elif level_weighting_method == 'exponential_decay':
            if decay_scale is None or not math.isfinite(decay_scale) or decay_scale <= 0:
                raise ValueError("exponential_decay requires finite positive decay_scale in metres.")
            vertical = torch.exp(-z / decay_scale)
        else:
            raise ValueError("Unknown level_weighting_method.")
    areas = torch.ones(n, device=device) if area_per_grid is None else area_per_grid.to(device=device, dtype=torch.float32)
    if areas.shape != (n,) or not torch.isfinite(areas).all() or (areas < 0).any():
        raise ValueError("area_per_grid must be finite, nonnegative and shaped (N,).")
    channel_weights = torch.cat([surface, (depth[:, None] * vertical[None, :]).flatten()])
    weights = channel_weights[:, None] * areas[None, :]
    if bathymetry_mask is not None:
        expected = (len(z) if len(z) else 1, n)
        if bathymetry_mask.dtype != torch.bool or bathymetry_mask.shape != expected:
            raise ValueError(f"bathymetry_mask must be Boolean with shape {expected}; True means wet.")
        wet = bathymetry_mask.to(device)
        surface_wet = wet[z.argmin() if len(z) else 0].expand(ns, n)
        depth_wet = wet.repeat(nd, 1) if nd else wet.new_empty((0, n))
        weights = weights * torch.cat([surface_wet, depth_wet])
    return weights


class AELoss(nn.Module):
    """Weighted MSE/MAE for predictions and targets shaped (B, N, channels).

    weights: Optional (channels, N) nonnegative tensor, typically get_weights().
    mask at call time: Boolean (N,), (N, channels), or (B, N, channels),
    True for included entries. Masks and zero weights exclude values before
    error evaluation, so masked NaNs/infinities contribute no loss or gradient.
    Reduction divides by the sum of included weights across the batch. Empty
    support raises ValueError; unmasked nonfinite values are not silently hidden.

    """

    def __init__(self, criterion: str = 'mse', weights: Optional[Tensor] = None) -> None:
        super().__init__()
        if criterion not in ('mse', 'mae'):
            raise ValueError(f"Unknown criterion: {criterion}")
        self.criterion = criterion
        if weights is not None:
            if weights.ndim != 2 or not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("weights must be finite, nonnegative and shaped (channels, N).")
            weights = weights.detach().T.unsqueeze(0).float().clone()
        self.register_buffer('weights', weights)

    def forward(self, prediction: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Reduce error over the intersection of positive weights and mask."""
        if prediction.ndim != 3 or prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shape (B, N, channels).")
        if self.weights is not None and self.weights.shape[1:] != prediction.shape[1:]:
            raise ValueError("weights do not match prediction nodes/channels.")
        # Accumulate half-precision errors/weights in float32 to avoid large-grid overflow.
        dtype = torch.promote_types(prediction.dtype, torch.float32)
        weights = torch.ones((), device=prediction.device, dtype=dtype) if self.weights is None else self.weights.to(device=prediction.device, dtype=dtype)
        weights = weights.expand(prediction.shape)
        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("mask must be Boolean; True means included.")
            b, n, c = prediction.shape
            if mask.shape == (n,):
                mask = mask[None, :, None]
            elif mask.shape not in ((n, c), (b, n, c)):
                raise ValueError("mask must have shape (N,), (N, C) or (B, N, C).")
            weights = weights * mask.to(prediction.device)
        denominator = weights.sum()
        if denominator <= 0:
            raise ValueError("No positive-weight unmasked entries remain.")
        included = weights > 0
        pred = torch.where(included, prediction.to(dtype), 0.)
        truth = torch.where(included, target.to(dtype), 0.)
        error = (pred - truth).square() if self.criterion == 'mse' else (pred - truth).abs()
        return (error * weights).sum() / denominator
