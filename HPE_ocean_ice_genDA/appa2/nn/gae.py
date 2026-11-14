r"""Graph autoencoder implementation."""

import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Sequence, Tuple, get_args

from appa2.date import get_local_time_encoding, get_year_progress_encoding
from appa2.grid import (
    create_edges,
    Icosphere,
    create_original_grid,
    latlon_to_xyz,
    PolarCapFilter,
    SUPPORTED_GRIDS,
    POLAR_GRIDS
)
from appa2.nn.graph import GraphPoolAttention, GraphSelfAttention, GraphDistancePool
from appa2.nn.layers import Linear, SirenEmbedding



class ForcingEncoder(nn.Module):
    """Hierarchical forcing encoder, returning finest-to-coarsest features.

    Attention queries use nearest features from the preceding grid plus
    positional embeddings. Calendar embeddings are added before
    initial pooling and after each pooling, before mesh self-attention.
    Calendar features are sampled from the native forcing grid at each level.
    Ocean geographic context is handled exclusively by GraphAE.
    """

    def __init__(self, grid, mask, meshes, full_mesh_sizes, in_channels,
                 hid_channels, heads, pooling_area, method, sa_edges, self_attention_blocks,
                 dropout=None, checkpointing=False, use_xfa=True, identity_init=False):
        super().__init__()
        if method not in ("distance", "attention"):
            raise ValueError("forcing_pooling_method must be 'distance' or 'attention'.")
        if len(pooling_area) != len(meshes):
            raise ValueError("forcing_pooling_area must have one entry per mesh level.")
        self.in_channels = in_channels
        self.register_buffer("grid", grid)
        self.register_buffer("mask", mask)
        self.pools = nn.ModuleList()
        self.processors = nn.ModuleList()
        self.calendar_embeddings = nn.ModuleList([
            SirenEmbedding(4, width, n_layers=1,
                           checkpointing=checkpointing)
            for width in [in_channels, *hid_channels]
        ])
        self.calendar_sizes = [len(g) for g in [grid, *meshes]]
        self.register_buffer("calendar_indices", torch.cat([
            create_edges(g, grid, neighbors=1)[:, 1]
            for g in [grid, *meshes]
        ]))
        # Calendar features follow the same native-grid sampling protocol as GAE.
        self.register_buffer("calendar_longitudes", grid[:, 1])
        source = grid
        source_width = in_channels
        for level, (vertices, size, width, n_heads, area) in enumerate(zip(
            meshes, full_mesh_sizes, hid_channels, heads, pooling_area
        )):
            radius = float(area) * 2 * math.asin(1 / math.sqrt(size))
            if not math.isfinite(radius) or not 0 < radius <= math.pi:
                raise ValueError("Forcing pooling radii must be in (0, pi].")
            edges = create_edges(vertices, source, max_arc=radius)
            if (torch.bincount(edges[:, 0], minlength=len(vertices)) == 0).any():
                raise ValueError("Some forcing destinations have no neighbors; increase forcing_pooling_area.")
            q, k = latlon_to_xyz(vertices), latlon_to_xyz(source)
            if method == "distance":
                pool = nn.Sequential(GraphDistancePool(q, k, edges, radius),
                                     Linear(source_width, width, identity_init=identity_init))
            else:
                pool = GraphPoolAttention(
                    q_coords=q, k_coords=k, edges=edges,
                    in_channels=source_width, hid_channels=width,
                    out_channels=width, heads=n_heads,
                    k_to_q_idx=create_edges(vertices, source, neighbors=1)[:, 1],
                    dropout=dropout, checkpointing=checkpointing, use_xfa=use_xfa,
                    identity_init=identity_init,
                )
            self.pools.append(pool)
            self.processors.append(nn.Sequential(*[
                GraphSelfAttention(channels=width, edges=sa_edges[level],
                                   heads=n_heads, dropout=dropout,
                                   checkpointing=checkpointing, use_xfa=use_xfa)
                for _ in range(self_attention_blocks[level])
            ]))
            source, source_width = vertices, width

    def forward(self, forcing: Tensor, t: Tensor) -> list[Tensor]:
        """Encode forcing with calendar conditioning, independent of ocean context.

        forcing: (B, native_forcing_nodes, in_channels), before filtering.
        t: (B, 4), using the same date convention as GraphAE.
        """
        if forcing.ndim != 3 or forcing.shape[1:] != (self.mask.numel(), self.in_channels):
            raise ValueError(
                f"forcing must have shape (B, {self.mask.numel()}, {self.in_channels})."
            )

        local_hour = get_local_time_encoding(t, self.calendar_longitudes)
        year = get_year_progress_encoding(t).unsqueeze(-2).expand(-1, local_hour.shape[1], -1)
        context = torch.cat([year, local_hour], dim=-1)
        sampled = torch.split(context[:, self.calendar_indices, :], self.calendar_sizes, dim=1)
        x = forcing[:, self.mask, :] + self.calendar_embeddings[0](sampled[0])
        outputs = []
        for level, (pool, processor) in enumerate(zip(self.pools, self.processors)):
            x = pool(x)
            x = x + self.calendar_embeddings[level + 1](sampled[level + 1])
            x = processor(x)
            outputs.append(x)
        return outputs


class GraphAE(nn.Module):
    r"""Creates a graph auto-encoder inspired from Graphcast's architecture.

    Illustration:
        The graph auto-encoder is decomposed in modules, each working on a specific query grid.
        An encoder module is structured as

            KEY_GRID => PROJ_KEY --> QUERY_GRID <-S-> QUERY_GRID

        with '=>' denoting a channel projection, '-->' an attention pooling block queried by query grid features
        and <-S-> are `S` self-attention blocks on the query grid.
        The auto-encoder is structured as

        N_320 => N_320                                                -- N_320 => N_320
                    |                                                 |
                    -- ICO-x                                      -- ICO-x
                            |                                     |
                            -- ...                              ...
                                    ICO-l => ICO_latent => ICO-l

        where intermediate grids are explicited as ICO-?.
        The latent icosphere is projected onto desired channels after self-attention as well as output grid which
        is projected back to input grid channels. In the ascending part, upper grids acts as keys and lower ones as queries.
        For the decoder, the symmetric applies from ICO_latent to the original grid.

    Arguments:
        in_channels: Number of input state channels :math:`C_i` corresponding to physical variables and their levels.
        latent_channels: Number of latent channels.
        context_channels: Number of context channels :math:`C_c`.
        hid_channels: Number of channels at each intermediate level.
        heads: Number of attention heads at each intermediate level.
        ico_divisions: Number of icosahedron subdivisions for each intermediate resolution.
        self_attention_blocks: Number of self-attention blocks at each intermediate resolution.
        pooling_area: Number of surface per point used for pooling at each intermediate resolution.
        use_hop_pooling: Whether to use hop for pooling area.
        self_attention_hops: Number of hops for self-attention at each intermediate level. If not specified, use multi-mesh attention.
        saturation: Saturation function used for latent regularization. Either softclip, softclip2, tanh or asinh.
        saturation_bound: Maximal absolute value of the saturation output. Does not apply for asinh.
        dropout: Dropout rate.
        identity_init: Whether to use identity like initialisation for pooling and projection blocks.
        forcing_grid: Optional supported native forcing grid; enables conditioning on both paths.
        forcing_in_channels: Number of raw forcing variables, before hidden-width encoding.
        forcing_pooling_area: Angular radius multipliers, one per mesh level; uses
            full-sphere equal-area cap reference radii, independently of ocean pooling.
        forcing_pooling_method: "distance" (Gaussian average then linear) or
            "attention" (nearest-state and positional queries).
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        context_channels: int,
        hid_channels: Sequence[int],
        heads: Sequence[int],
        ico_divisions: Sequence[int],
        self_attention_blocks: Sequence[int],
        pooling_area: Sequence[float],
        use_hop_pooling: bool = False,
        self_attention_hops: Optional[Sequence[int]] = None,
        polar_cap_filter_longitude_bands: Optional[Sequence[float]] = None,
        polar_cap_filter_min_latitudes: Optional[Sequence[float]] = None,
        starting_grid: SUPPORTED_GRIDS = "N_320",
        saturation: str = "softclip2",
        saturation_bound: float = 5.0,
        noise_level: float = 0.0,
        dropout: Optional[float] = None,
        identity_init: bool = False,
        checkpointing: bool = False,
        use_xfa: bool = True,
        forcing_grid: Optional[SUPPORTED_GRIDS] = None,
        forcing_in_channels: Optional[int] = None,
        forcing_pooling_area: Optional[Sequence[float]] = None,
        forcing_pooling_method: str = "distance",
        **kwargs,
    ):
        super().__init__()
        if starting_grid not in get_args(SUPPORTED_GRIDS):
            raise ValueError(
                f"Unsupported starting grid {starting_grid}. Supported grids are {SUPPORTED_GRIDS}."
            )
        if (polar_cap_filter_longitude_bands is None) != (polar_cap_filter_min_latitudes is None):
            raise ValueError(
                "Both polar_cap_filter_longitude_bands and polar_cap_filter_min_latitudes must be provided together."
            )
        if (starting_grid in get_args(POLAR_GRIDS) and 
         (polar_cap_filter_longitude_bands is None or 
          polar_cap_filter_min_latitudes is None)):
            raise ValueError(
                f"Starting grid {starting_grid} requires polar cap filter parameters."
            )

        if polar_cap_filter_longitude_bands is not None and polar_cap_filter_min_latitudes is not None:
            self.polar_cap_filter = PolarCapFilter(
                longitude_bands=polar_cap_filter_longitude_bands,
                min_latitudes=polar_cap_filter_min_latitudes,
            )
        else:
            self.polar_cap_filter = None

        # CREATE PROBLEM-RELATED FEATURES
        # Full-sphere reference meshes.
        icospheres = [Icosphere(N) for N in ico_divisions]

        # Regional meshes used by attention.
        self.icosphere_graphs = [
            sphere.get_graph(polar_cap_filter=self.polar_cap_filter)
            for sphere in icospheres
        ]

        grid, mask = create_original_grid(starting_grid, polar_cap_filter=self.polar_cap_filter)

        self.register_buffer("starting_grid", grid)
        self.register_buffer("polar_mask", mask)

        enc_grids = [self.starting_grid, *[vertices for vertices, edges in self.icosphere_graphs]]
        self.register_buffer("in_longitudes", enc_grids[0][:, -1])

        # Create pixel to ico static edges for context
        static_pool_indices = []
        for grid in enc_grids[1:]:
            static_indices = create_edges(grid, enc_grids[0], neighbors=1)[:, -1]
            static_pool_indices.append(static_indices)
        self.static_pool_indices_cat = nn.Buffer(torch.cat(static_pool_indices))
        self.static_pool_sizes = list(map(len, static_pool_indices))
        if not use_hop_pooling:
            # The whole earth surface should be covered
            min_area = 2 / math.sqrt(3)
            for area in pooling_area:
                assert area > min_area, f"pooling area must be at least {min_area}, found {area}."

            # As the distance is almost constant between points on the icosphere,
            # pooling is made using multiple of arc corresponding to area per point of the query grid.
            pool_args = [
                {"max_arc": n_area * 2 * math.asin(math.sqrt(1 / sphere.num_vertices))}
                for n_area, sphere in zip(pooling_area, icospheres)
            ]
        else:
            for n_hops in pooling_area:
                assert (
                    n_hops % 1 == 0 and n_hops >= 1
                ), f"must use integer area (>= 1) when using hops pooling, found {n_hops}."
            pool_args =  [
                {"max_arc": sphere.get_hop_radius(n_hops=int(n_area))}
                for n_area, sphere in zip(pooling_area, icospheres)
            ]

        pool_edges = []
        pool_q_features = []
        pool_k_features = []
        encoder_q_idx = []
        decoder_q_idx = []

        for key, query, pool_arg in zip(enc_grids[:-1], enc_grids[1:], pool_args):

            ed = create_edges(query, key, **pool_arg)

            ## Check that all query nodes have at least one pooling edge given the filtering of the polar cap. This is important to avoid empty pooling for some query nodes.
            encoder_degree = torch.bincount(ed[:, 0], minlength=len(query))
            decoder_degree = torch.bincount(ed[:, 1], minlength=len(key))

            if (encoder_degree == 0).any():
                raise ValueError("Some encoder query nodes have no pooling edges.")

            if (decoder_degree == 0).any():
                raise ValueError("Some decoder query nodes have no pooling edges.")


            pool_edges.append(ed)

            # Source indices for query creation
            encoder_q_idx.append(create_edges(query, key, neighbors=1)[:, -1])
            decoder_q_idx.append(create_edges(key, query, neighbors=1)[:, -1])

            feat_q = latlon_to_xyz(query)
            feat_k = latlon_to_xyz(key)

            pool_q_features.append(feat_q)
            pool_k_features.append(feat_k)

        if self_attention_hops is None:
            sa_edges = [edges for vertices, edges in self.icosphere_graphs]
        else:
            sa_edges = [
                create_edges(
                    vertices,
                    vertices,
                    max_arc=sphere.get_hop_radius(n_hops=n_hops),
                )
                for sphere, (vertices, _), n_hops in zip(
                    icospheres, self.icosphere_graphs, self_attention_hops
                )
            ]

        self.forcing_encoder = None
        if forcing_grid is not None:
            if forcing_grid not in get_args(SUPPORTED_GRIDS):
                raise ValueError(f"Unsupported forcing grid {forcing_grid}.")
            if starting_grid in get_args(POLAR_GRIDS) and forcing_grid not in get_args(POLAR_GRIDS):
                raise ValueError("A polar starting grid requires a polar forcing grid.")
            if forcing_pooling_area is None:
                raise ValueError("forcing_pooling_area is required when forcing_grid is set.")
            forcing_coords, forcing_mask = create_original_grid(
                forcing_grid, polar_cap_filter=self.polar_cap_filter
            )
            self.forcing_encoder = ForcingEncoder(
                forcing_coords, forcing_mask,
                [vertices for vertices, _ in self.icosphere_graphs],
                [sphere.num_vertices for sphere in icospheres],
                forcing_in_channels, hid_channels, heads, forcing_pooling_area,
                forcing_pooling_method, sa_edges, self_attention_blocks,
                dropout=dropout, checkpointing=checkpointing, use_xfa=use_xfa,
                identity_init=identity_init,
            )
            self.encoder_forcing_embedding = nn.ModuleList([nn.Linear(w, w) for w in hid_channels])
            self.decoder_forcing_embedding = nn.ModuleList([nn.Linear(w, w) for w in reversed(hid_channels)])
        elif forcing_in_channels is not None or forcing_pooling_area is not None:
            raise ValueError("Forcing configuration requires forcing_grid.")

        # CREATE AE ARCHITECTURE

        self.encoder = []
        self.decoder = []
        self.encoder_context_embedding = []
        self.decoder_context_embedding = []

        for i in range(len(hid_channels)):
            if i == 0:
                in_dim = in_channels
                out_dim = in_channels
                self.encoder_context_embedding.append(
                    SirenEmbedding(
                        4 + context_channels,
                        channels=in_dim,
                        n_layers=1,
                        checkpointing=checkpointing,
                    )
                )
            else:
                in_dim = out_dim = hid_channels[i - 1]

            self.encoder.append(
                GraphPoolAttention(
                    q_coords=pool_q_features[i],
                    k_coords=pool_k_features[i],
                    edges=pool_edges[i],
                    in_channels=in_dim,
                    hid_channels=hid_channels[i],
                    out_channels=hid_channels[i],
                    heads=heads[i],
                    k_to_q_idx=encoder_q_idx[i],
                    dropout=dropout,
                    identity_init=identity_init,
                    checkpointing=checkpointing,
                    use_xfa=use_xfa,
                )
            )

            self.decoder.insert(
                0,
                GraphPoolAttention(
                    q_coords=pool_k_features[i],
                    k_coords=pool_q_features[i],
                    edges=pool_edges[i].flip(-1),
                    in_channels=hid_channels[i],
                    hid_channels=hid_channels[i],
                    out_channels=out_dim,
                    heads=heads[i],
                    k_to_q_idx=decoder_q_idx[i],
                    dropout=dropout,
                    identity_init=identity_init,
                    checkpointing=checkpointing,
                    use_xfa=use_xfa,
                ),
            )

            self.encoder_context_embedding.append(
                SirenEmbedding(
                    4 + context_channels,
                    channels=hid_channels[i],
                    n_layers=1,
                    checkpointing=checkpointing,
                )
            )

            self.decoder_context_embedding.insert(
                0,
                SirenEmbedding(
                    4 + context_channels,
                    channels=hid_channels[i],
                    n_layers=1,
                    checkpointing=checkpointing,
                ),
            )

            for _ in range(self_attention_blocks[i]):
                self.encoder.append(
                    GraphSelfAttention(
                        channels=hid_channels[i],
                        edges=sa_edges[i],
                        heads=heads[i],
                        dropout=dropout,
                        checkpointing=checkpointing,
                        use_xfa=use_xfa,
                    )
                )
                self.decoder.insert(
                    0,
                    GraphSelfAttention(
                        channels=hid_channels[i],
                        edges=sa_edges[i],
                        heads=heads[i],
                        dropout=dropout,
                        checkpointing=checkpointing,
                        use_xfa=use_xfa,
                    ),
                )

            if i + 1 == len(hid_channels):
                self.encoder.append(
                    Linear(
                        hid_channels[i],
                        latent_channels,
                        identity_init=identity_init,
                    )
                )
                self.decoder.insert(
                    0,
                    Linear(
                        latent_channels,
                        hid_channels[i],
                        identity_init=identity_init,
                    ),
                )

        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.encoder_context_embedding = nn.ModuleList(self.encoder_context_embedding)
        self.decoder_context_embedding = nn.ModuleList(self.decoder_context_embedding)

        self.saturation_bound = saturation_bound
        self.saturation = saturation
        self.noise_level = noise_level

    @property
    def latent_shape(self) -> Tuple[int, int]:
        num_nodes = self.icosphere_graphs[-1][0].shape[0]
        latent_channels = self.encoder[-1].out_features

        return num_nodes, latent_channels

    def description(self):
        # TODO: Improve description
        return "Graph autoencoder"

    def static_pool(self, context: Tensor) -> Sequence[Tensor]:
        r"""Pools the context to each ICO level.

        Arguments:
            context: The auto-encoding context, with shape :math:`(*, N, K)`.

        Returns:
            A list of pooled context, each with shape :math:`(*, N_i, K)`.
        """

        static_pool_indices = torch.split(self.static_pool_indices_cat, self.static_pool_sizes)

        return [context[..., i, :] for i in static_pool_indices]

    def saturate(self, z: Tensor) -> Tensor:
        if self.saturation is None:
            return z
        elif self.saturation == "softclip":
            return z / (1 + torch.abs(z) / self.saturation_bound)
        elif self.saturation == "softclip2":
            return z * torch.rsqrt(1 + torch.square(z / self.saturation_bound))
        elif self.saturation == "tanh":
            return self.saturation_bound * torch.tanh(z / self.saturation_bound)
        elif self.saturation == "asinh":
            return torch.asinh(z)
        else:
            raise ValueError(f"Unknown saturation function: {self.saturation}")

    def encode_date(self, date: Tensor):
        year_progress = get_year_progress_encoding(date).unsqueeze(-2)
        local_hour = get_local_time_encoding(date, self.in_longitudes)
        year_progress = year_progress.repeat(1, local_hour.shape[1], 1)
        date = torch.cat([year_progress, local_hour], dim=-1)

        return date

    @staticmethod
    def _check_forcing_batch(forcing, state):
        if forcing is not None and forcing.ndim == 3 and forcing.shape[0] != state.shape[0]:
            raise ValueError("Forcing and ocean batch sizes must match.")

    def encode_forcing(self, forcing: Optional[Tensor] = None,
                       t: Optional[Tensor] = None) -> Optional[list[Tensor]]:
        """Return finest-to-coarsest forcing features for reuse or caching."""
        if self.forcing_encoder is None:
            if forcing is not None:
                raise ValueError("Forcing supplied but forcing_grid is not configured.")
            return None
        if forcing is None:
            raise ValueError("forcing is required when forcing_grid is configured.")

        return self.forcing_encoder(forcing, t)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None,
                forcing: Optional[Tensor] = None):
        r"""
        Arguments:
            x: State tensor containing variables of interest, with shape :math:`(B, N, Z)`.
            t: Timestamp tensor, with shape :math:`(B, 4)`.
            c: Context tensor, with shape :math:`(B, N, K)`.
            forcing: Native forcing tensor (B, N_forcing, forcing_in_channels).
                Required exactly when forcing_grid is configured.
        """
        self._check_forcing_batch(forcing, x)
        forcing_features = self.encode_forcing(forcing, t)
        z = self._encode(x, t, c, forcing_features)
        x = self._decode(z, t, c, forcing_features)
        if forcing_features is not None:
            return z, x, forcing_features
        return z, x

    def encode(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None,
               forcing: Optional[Tensor] = None):
        """Encode ocean state; configured forcing is required on its native grid."""
        self._check_forcing_batch(forcing, x)
        return self._encode(x, t, c, self.encode_forcing(forcing, t))

    def _encode(self, x, t, c, forcing_features):
        # Compose context as timestamp features and given physical context
        x = x[..., self.polar_mask, :] ##PG: If a polar cap fileter is used, the input shape will be different from the output shape. This is to ensure that the input shape is consistent with the output shape.
        t = self.encode_date(t)
        if c is not None:
            c = c[..., self.polar_mask, :]
            hr_context = torch.cat([t, c], dim=-1)
        else:
            hr_context = t
        pooled_context = self.static_pool(hr_context)

        context_idx = 0
        x = x + self.encoder_context_embedding[context_idx](hr_context)

        for e_block in self.encoder:
            x = e_block(x)
            if isinstance(e_block, GraphPoolAttention):
                context_idx += 1
                x = x + self.encoder_context_embedding[context_idx](
                    pooled_context[context_idx - 1]
                )
                if forcing_features is not None:
                    x = x + self.encoder_forcing_embedding[context_idx - 1](
                        forcing_features[context_idx - 1]
                    )

        x = self.saturate(x)

        return x

    def decode(self, z: Tensor, t: Tensor, c: Optional[Tensor] = None,
               forcing: Optional[Tensor] = None):
        """Decode ocean latent; configured forcing is required on its native grid."""
        self._check_forcing_batch(forcing, z)
        return self._decode(z, t, c, self.encode_forcing(forcing, t))

    def _decode(self, z, t, c, forcing_features):
        # Compose context as timestamp features and given physical context
        t = self.encode_date(t)
        if c is not None:
            c = c[..., self.polar_mask, :]
            hr_context = torch.cat([t, c], dim=-1)
        else:
            hr_context = t
        pooled_context = self.static_pool(hr_context)
        pooled_context.reverse()

        z = z + torch.randn_like(z) * self.noise_level

        context_idx = 0
        for d_block in self.decoder:
            if isinstance(d_block, GraphPoolAttention):
                z = z + self.decoder_context_embedding[context_idx](pooled_context[context_idx])
                if forcing_features is not None:
                    z = z + self.decoder_forcing_embedding[context_idx](
                        forcing_features[-1 - context_idx]
                    )
                context_idx += 1
            z = d_block(z)

        output = z.new_zeros( ##PG: If a polar cap fileter is used, the output shape will be different from the input shape. This is to ensure that the output shape is consistent with the input shape. 
            (*z.shape[:-2], self.polar_mask.numel(), z.shape[-1])
        )
        output[..., self.polar_mask, :] = z

        return output ##PG: this used to be z. Becareful with other scripts.
