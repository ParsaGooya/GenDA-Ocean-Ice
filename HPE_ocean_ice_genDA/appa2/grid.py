r"""Grid and mesh helpers."""

import math
import torch

from scipy.spatial import KDTree
from torch import LongTensor, Tensor
from typing import Final, Optional, Sequence, Literal



SUPPORTED_GRIDS = Literal["N_320", "glorys", "RIOPS"]
POLAR_GRIDS = Literal["RIOPS"]
GLORYS_MIN_LAT: Final[float] = 26

def latlon_to_xyz(grid: Tensor) -> Tensor:
    r"""Converts a latitude-longitude grid to 3D cartesian coordinates.

    Reference:
        | GraphCast: Learning skillful medium-range global weather forecasting
        | https://github.com/google-deepmind/graphcast

    Arguments:
        grid: The latitude-longitude grid to convert, with shape :math:`(*, 2)`.

    Returns:
        The cartesian grid, with shape :math:`(*, 3)`.
    """

    lat, lon = torch.deg2rad(grid).unbind(dim=-1)
    lat = torch.pi / 2 - lat

    x = torch.sin(lat) * torch.cos(lon)
    y = torch.sin(lat) * torch.sin(lon)
    z = torch.cos(lat)

    return torch.stack((x, y, z), dim=-1)


def xyz_to_latlon(grid: Tensor) -> Tensor:
    r"""Converts a 3D cartesian grid to latitude-longitude coordinates.

    Arguments:
        grid: The 3D cartesian grid to convert, with shape :math:`(*, 3)`.

    Returns:
        The latitude-longitude grid, with shape :math:`(*, 2)`.
    """

    x, y, z = grid.unbind(dim=-1)

    lat = torch.asin(z) * 180 / torch.pi
    lon = torch.atan2(y, x) * 180 / torch.pi

    return torch.stack((lat, lon), dim=-1)


def arc_to_chord(arc: float) -> float:
    r"""Converts arc length to chord length.

    Wikipedia:
        https://wikipedia.org/wiki/Chord_(geometry)

    Arguments:
        arc: The arc length, expressed in [rad].

    Returns:
        The chord length, dimensionless.
    """

    return 2 * math.sin(arc / 2)


def chord_to_arc(chord: float) -> float:
    r"""Converts chord length to arc length.

    Wikipedia:
        https://wikipedia.org/wiki/Chord_(geometry)

    Arguments:
        chord: The chord length, dimensionless.

    Returns:
        The arc length, expressed in [rad].
    """

    return 2 * math.asin(chord / 2)




class PolarCapFilter:
    """Latitude thresholds for consecutive, cyclic longitude bands.

    min_latitudes[i] applies to the band from longitude_bands[i - 1]
    eastward to longitude_bands[i]. The first band wraps across
    the dateline.

    Longitude bands include their western boundary and exclude
    their eastern boundary. Latitude thresholds are inclusive.
    """

    def __init__(
        self,
        longitude_bands: Sequence[float],
        min_latitudes: Sequence[float],
    ):
        longitudes = tuple(float(x) for x in longitude_bands)
        latitudes = tuple(float(x) for x in min_latitudes)

        if not longitudes or len(longitudes) != len(latitudes):
            raise ValueError(
                "longitudes and latitudes must have the same nonzero length."
            )

        if not all(math.isfinite(x) for x in (*longitudes, *latitudes)):
            raise ValueError("All coordinates must be finite.")

        if not all(-180 <= x < 180 for x in longitudes):
            raise ValueError("Longitudes must be in [-180, 180).")

        if any(a >= b for a, b in zip(longitudes, longitudes[1:])):
            raise ValueError("Longitudes must be strictly increasing.")

        if not all(-90 <= x <= 90 for x in latitudes):
            raise ValueError("Latitudes must be in [-90, 90].")

        self.longitudes = longitudes
        self.latitudes = latitudes

    def __call__(self, vertices: torch.Tensor) -> torch.Tensor:
        """Return a Boolean keep mask for vertices shaped (N, 2)."""
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (N, 2): latitude, longitude.")

        lat = vertices[:, 0]
        lon = (vertices[:, 1] + 180) % 360 - 180

        boundaries = vertices.new_tensor(self.longitudes)
        thresholds = vertices.new_tensor(self.latitudes)

        # Find the band ending at the next longitude boundary.
        # Values beyond the last boundary wrap to the first band.
        band = torch.bucketize(lon.contiguous(), boundaries, right=True)
        band = band % len(self.longitudes)

        return lat >= thresholds[band]


def ORG_vertices_per_lat(latitude_idx: int) -> int:
    r"""Returns the number of longitude vertices per latitude index. The first index has to be 1.

    Arguments:
        latitude_idx: The latitude index :math:`i`.

    Returns:
        The number of longitude points :math:`M_i`.
    """

    assert latitude_idx > 0
    return latitude_idx * 4 + 16


def create_ORG(N: int) -> Tensor:
    r"""Creates an approximate octahedral reduced Gaussian (ORG) grid.

    References:
        | Introducing the octahedral reduced Gaussian grid
        | https://confluence.ecmwf.int/display/FCST/Introducing+the+octahedral+reduced+Gaussian+grid

    Arguments:
        N: The number of latitude lines per hemisphere of the grid.

    Returns:
        The ORG vertices, with shape :math:`(2 \sum_{i = 1}^N M_i, 2)`. A point is a latitude-longitude pair.
    """

    vertices = []
    latitude_lines = torch.linspace(90 / (2 * N), 90 - 90 / (2 * N), N)
    for i, lat in enumerate(latitude_lines):
        M_i = ORG_vertices_per_lat(i + 1)

        lon = torch.linspace(0, 360 - 360 / M_i, M_i)
        lon[lon > 180] = lon[lon > 180] - 360

        abs_lat = torch.abs(lat - 90)
        abs_lat = abs_lat.expand(M_i)

        vertices.append(torch.stack((abs_lat, lon), axis=-1))
        vertices.append(torch.stack((-abs_lat, lon), axis=-1))

    return torch.concatenate(vertices)


def create_N320() -> Tensor:
    r"""Creates the N320 reduced Gaussian grid.

    Returns:
        The N320 vertices, with shape :math:`(721 \times 1440, 2)`. A point is a latitude-longitude pair.
    """

    lat = torch.linspace(90, -90, 721)
    lon = torch.linspace(0, 360 - 360 / 1440, 1440)
    lon[720:] = lon[720:] - 360
    coord = torch.cartesian_prod(lat, lon)

    return coord

def create_glorys(min_lat: float = -80) -> Tensor:
    r"""Creates the GLORYS 1/12-degree grid above a minimum latitude.

    Arguments:
        min_lat: Minimum latitude to retain, between -80 and 90 degrees.

    Returns:
        Vertices with shape :math:`(N, 2)`, ordered by descending latitude.
        Each point is a latitude-longitude pair in degrees.
    """
    if not -80 <= min_lat <= 90:
        raise ValueError("min_lat must be between -80 and 90 degrees.")

    lat = torch.linspace(-80, 90, 2041)
    lat = lat[lat >= min_lat]
    lon = torch.linspace(-180, 180 - 1 / 12, 4320)

    return torch.cartesian_prod(lat, lon)

def create_original_grid(
        grid_name: SUPPORTED_GRIDS,
        polar_cap_filter: Optional[PolarCapFilter] = None,
    ) -> Tensor:

        if grid_name == "N_320":
            coord = create_N320()
        elif grid_name == "glorys":
            coord = create_glorys(GLORYS_MIN_LAT)
        elif grid_name == "RIOPS":
            coord = create_ORG(80)
        else:
            raise ValueError(
                f"Unsupported grid {grid_name}. Supported grids are 'N_320', 'glorys', 'RIOPS'."
            )

        if polar_cap_filter is not None:
            mask = polar_cap_filter(coord)
            coord = coord[mask]

        return coord, mask

def _project(v):
    r"""Projects a vertex onto the unit sphere.

    Arguments:
        v: Vertex cartesian coordinates, with shape :math:`(*, 3)`.

    Returns:
        The projected vertex :math:`v / norm(v)`.
    """

    length = sum(x**2 for x in v) ** 0.5
    return [x / length for x in v]


def _mid_point(i, j, vertices, edges):
    r"""Creates or retrieves the midpoint of an edge and adds it to the vertex list.

    Arguments:
        i: The first vertex index.
        j: The second vertex index.
        vertices: The vertex list.
        edges: The edges dictionary.

    Returns:
        The midpoint index.
    """

    edge = tuple(sorted([i, j]))
    if edge not in edges:
        mid_point = _project([
            (vertices[i][0] + vertices[j][0]) / 2,
            (vertices[i][1] + vertices[j][1]) / 2,
            (vertices[i][2] + vertices[j][2]) / 2,
        ])
        edges[edge] = len(vertices)
        vertices.append(mid_point)
    return edges[edge]


def _edges_from_faces(faces: Sequence[Sequence[int]]):
    r"""Creates directed edges from triangular faces.

    Arguments:
        faces: The sequence of faces vertices.

    Returns:
        The dictionary of directed edges.
    """

    edges = {}
    for f in faces:
        for i, j in zip(range(3), range(-1, 2)):
            edges.setdefault(f[i], set()).add(f[j])
            edges.setdefault(f[j], set()).add(f[i])

    return edges


class Icosphere:
    """Construct and retain a full-sphere multi-mesh graph.

    Attributes:
        subdivisions: Number of face subdivisions.
        vertices: Full-sphere latitude-longitude coordinates in degrees,
            with shape (N, 2).
        edges: Full-sphere directed multi-mesh edges, including
            self-connections, with shape (E, 2).
    """

    def __init__(self, subdivisions: int = 6):
        if (
            isinstance(subdivisions, bool)
            or not isinstance(subdivisions, int)
            or subdivisions < 0
        ):
            raise ValueError("subdivisions must be a nonnegative integer.")

        self.subdivisions = subdivisions

        phi = (1 + 5**0.5) / 2
        vertices = [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ]
        vertices = [_project(v) for v in vertices]

        faces = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]

        multimesh = _edges_from_faces(faces)

        for _ in range(subdivisions):
            new_faces = []
            midpoint_cache = {}

            for v1, v2, v3 in faces:
                v12 = _mid_point(v1, v2, vertices, midpoint_cache)
                v23 = _mid_point(v2, v3, vertices, midpoint_cache)
                v31 = _mid_point(v3, v1, vertices, midpoint_cache)

                new_faces.extend([
                    [v1, v12, v31],
                    [v2, v23, v12],
                    [v3, v31, v23],
                    [v12, v23, v31],
                ])

            for k, neighbours in _edges_from_faces(new_faces).items():
                multimesh.setdefault(k, set()).update(neighbours)

            faces = new_faces

        for k in multimesh:
            multimesh[k].add(k)

        self.vertices = xyz_to_latlon(
            torch.tensor(vertices, dtype=torch.float32)
        )

        target = torch.cat([
            torch.full((len(neighbours),), k, dtype=torch.long)
            for k, neighbours in multimesh.items()
        ])
        source = torch.cat([
            torch.tensor(sorted(neighbours), dtype=torch.long)
            for neighbours in multimesh.values()
        ])
        self.edges = torch.stack([target, source], dim=-1)

    @property
    def num_vertices(self) -> int:
        """Number of vertices on the full sphere."""
        return self.vertices.shape[0]

    def get_graph(
        self,
        polar_cap_filter: Optional[PolarCapFilter] = None,
        return_edges: bool = True,
    ):
        """Return the full or filtered graph.

        Returned tensors are independent copies of the stored graph.
        Filtering preserves vertex order and remaps edge indices.

        Returns:
            (vertices, edges) when return_edges=True.
            vertices otherwise.
        """
        if polar_cap_filter is None:
            if return_edges:
                return self.vertices.clone(), self.edges.clone()
            return self.vertices.clone()

        keep = polar_cap_filter(self.vertices)

        if keep.dtype != torch.bool or keep.shape != (self.num_vertices,):
            raise ValueError(
                "polar_cap_filter must return a Boolean mask of shape (N,)."
            )

        vertices = self.vertices[keep]

        if not return_edges:
            return vertices

        # Remove edges touching discarded vertices.
        edges = self.edges[keep[self.edges].all(dim=1)]

        # Translate original indices into filtered vertex indices.
        old_to_new = torch.full_like(
            keep, -1, dtype=torch.long
        )
        old_to_new[keep] = torch.arange(
            vertices.shape[0],
            dtype=torch.long,
            device=keep.device,
        )
        edges = old_to_new[edges]

        return vertices, edges

    def get_hop_radius(self, n_hops: int) -> float:
        """Estimate an angular neighbourhood radius from full-sphere geometry.

        Uses the maximum distance to the k-th nearest vertex, where k
        approximates the number of vertices within n_hops mesh steps.

        Returns:
            Angular radius in radians.
        """

        if (
            isinstance(n_hops, bool)
            or not isinstance(n_hops, int)
            or n_hops < 0
        ):
            raise ValueError("n_hops must be a nonnegative integer.")

        if n_hops == 0:
            return 0.0

        query_xyz = latlon_to_xyz(self.vertices)
        num_vertices = query_xyz.shape[0]

        if num_vertices > 12:
            centres = query_xyz[12:]
            # Remove icosahedron edges (12 first) that only have 5 direct neighbors
            # The number of neighbors in an icosphere up to n hops is 1 + 3*n*(n+1)
            k=1 + 3 * n_hops * (n_hops + 1)
        else:
            # We are on ico-0
            centres = query_xyz
            k=1 + 2.5 * n_hops * (n_hops + 1)

        if k > num_vertices:
            raise ValueError(
                f"n_hops={n_hops} requires {k} nearest vertices, "
                f"but this sphere has only {num_vertices}."
            )

        query_kd = KDTree(query_xyz)
        hop_dist, _ = query_kd.query(centres, k=[k])
        # Take the maximum will ensure that every node is at least connected to its n_hop neighbors
        hop_dist = float(hop_dist.max())
        hop_arc = float(chord_to_arc(hop_dist))
        return hop_arc


def icosphere_nhops_edges(
    ico_query: Tensor,
    key: Tensor,
    n_hops: int,
) -> LongTensor:
    r"""Creates edge indices of n hops neighbors between any key points and icosphere query points.

    Arguments:
        ico_query: Query icosphere grid from which hops distance is inferred, with shape :math:`(M, 2)`.
        key: The key grid, with shape :math:`(N, 2)`.
        n_hops: Number of hops to delimit the neighborhood.

    Returns:
        The edge indices, with shape :math:`(E, 2)`.
    """

    query_xyz = latlon_to_xyz(ico_query)
    query_kd = KDTree(query_xyz)

    if query_xyz.shape[0] > 12:
        # Remove icosahedron edges (12 first) that only have 5 direct neighbors
        # The number of neighbors in an icosphere up to n hops is 1 + 3*n*(n+1)
        hop_dist, _ = query_kd.query(query_xyz[12:], k=1 + 3 * n_hops * (n_hops + 1))
    else:
        # We are on ico-0
        hop_dist, _ = query_kd.query(query_xyz, k=1 + 2.5 * n_hops * (n_hops + 1))

    # Take the maximum will ensure that every node is at least connected to its n_hop neighbors
    hop_dist = hop_dist.max()
    hop_arc = chord_to_arc(hop_dist)

    return create_edges(ico_query, key, max_arc=hop_arc)


def create_edges(
    query: Tensor,
    key: Tensor,
    max_arc: Optional[float] = None,
    neighbors: Optional[int] = None,
) -> LongTensor:
    r"""Creates edge indices of neighborhoods between key and query points.

    Arguments:
        query: The query grid, with shape :math:`(M, 2)`.
        key: The key grid, with shape :math:`(N, 2)`.
        max_arc: The maximum arc length between neighbors, expressed in [rad].
        neighbors: The number of nearest neighbors to extract for each query point. Overrides max_arc if greater than 0.

    Returns:
        The edge indices, with shape :math:`(E, 2)`.
    """

    M, _ = query.shape

    key_xyz = latlon_to_xyz(key)
    key_kd = KDTree(key_xyz)

    query_xyz = latlon_to_xyz(query)

    assert not (
        max_arc is None and neighbors is None
    ), "either 'max_arc' or 'neighbors' must be provided."

    if max_arc is None:
        _, key_indices = key_kd.query(query_xyz, k=neighbors, workers=-1)
        key_indices = torch.from_numpy(key_indices.flatten())
    else:
        key_indices = key_kd.query_ball_point(query_xyz, arc_to_chord(max_arc), workers=-1)
        neighbors = torch.as_tensor([len(ki) for ki in key_indices])
        key_indices = torch.cat([torch.as_tensor(ki) for ki in key_indices])

    query_indices = torch.arange(M).repeat_interleave(neighbors)

    return torch.stack((query_indices, key_indices), dim=-1).long()


def num_icosphere_vertices(icosphere_divisions: int) -> int:
    r"""Returns the number of vertices in an icosphere.

    Arguments:
        icosphere_divisions: The number of divisions of the icosphere.

    Returns:
        The number of vertices :math:`N`.
    """

    return 10 * 4**icosphere_divisions + 2
