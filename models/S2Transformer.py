"""Structured spatial Transformer for French global-station forecasting.

This is an M3-native implementation of the public S2Transformer architecture:
each block performs attention within geographic subgraphs, pools subgraph
tokens for global inter-subgraph attention, and broadcasts the global message
back to stations.  Fine-to-coarse partitions progressively expand the spatial
scale.  Variables are folded into the batch dimension, matching the paper's
multi-station single-variable forecasting setup.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils.station_coordinates import load_station_coordinates


def _coordinate_features(coordinates: np.ndarray) -> np.ndarray:
    latitude = np.deg2rad(coordinates[:, 0])
    longitude = np.deg2rad(coordinates[:, 1])
    altitude = coordinates[:, 2]
    altitude = (altitude - altitude.mean()) / (altitude.std() + 1e-6)
    return np.stack(
        (np.sin(latitude), np.cos(latitude), np.sin(longitude),
         np.cos(longitude), altitude), axis=-1,
    ).astype(np.float32)


def _geographic_partitions(coordinates: np.ndarray, group_counts: list[int]):
    from sklearn.cluster import KMeans

    features = _coordinate_features(coordinates)[:, :4]
    partitions = []
    for groups in group_counts:
        groups = min(max(1, int(groups)), len(coordinates))
        labels = KMeans(
            n_clusters=groups, random_state=2024, n_init=20
        ).fit_predict(features)
        members = [np.flatnonzero(labels == group) for group in range(groups)]
        width = max(map(len, members))
        indices = np.full((groups, width), len(coordinates), dtype=np.int64)
        valid = np.zeros((groups, width), dtype=bool)
        for group, stations in enumerate(members):
            indices[group, :len(stations)] = stations
            valid[group, :len(stations)] = True
        partitions.append((indices, valid))
    return partitions


class StructuredSpatialBlock(nn.Module):
    def __init__(self, dimension, heads, feedforward, dropout, indices, valid):
        super().__init__()
        self.register_buffer("indices", torch.as_tensor(indices, dtype=torch.long))
        self.register_buffer("valid", torch.as_tensor(valid, dtype=torch.bool))
        self.local_norm = nn.LayerNorm(dimension)
        self.local_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.local_ffn_norm = nn.LayerNorm(dimension)
        self.local_ffn = nn.Sequential(
            nn.Linear(dimension, feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward, dimension), nn.Dropout(dropout),
        )
        self.global_norm = nn.LayerNorm(dimension)
        self.global_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.global_ffn_norm = nn.LayerNorm(dimension)
        self.global_ffn = nn.Sequential(
            nn.Linear(dimension, feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward, dimension), nn.Dropout(dropout),
        )
        self.fusion = nn.Linear(2 * dimension, dimension)

    def forward(self, station_tokens):
        samples, stations, dimension = station_tokens.shape
        padding = station_tokens.new_zeros(samples, 1, dimension)
        padded = torch.cat((station_tokens, padding), dim=1)
        grouped = padded[:, self.indices]
        groups, width = self.indices.shape
        local = grouped.reshape(samples * groups, width, dimension)
        key_padding = (~self.valid).unsqueeze(0).expand(samples, -1, -1)
        key_padding = key_padding.reshape(samples * groups, width)
        normalized = self.local_norm(local)
        attended, _ = self.local_attention(
            normalized, normalized, normalized, key_padding_mask=key_padding,
            need_weights=False,
        )
        local = local + attended
        local = local + self.local_ffn(self.local_ffn_norm(local))
        local = local.reshape(samples, groups, width, dimension)
        local = local * self.valid[None, :, :, None]

        denominator = self.valid.sum(dim=1).clamp_min(1)[None, :, None]
        subgraphs = local.sum(dim=2) / denominator
        normalized = self.global_norm(subgraphs)
        global_message, _ = self.global_attention(
            normalized, normalized, normalized, need_weights=False
        )
        subgraphs = subgraphs + global_message
        subgraphs = subgraphs + self.global_ffn(self.global_ffn_norm(subgraphs))
        broadcast = subgraphs[:, :, None, :].expand(-1, -1, width, -1)
        fused = local + torch.relu(self.fusion(torch.cat((local, broadcast), dim=-1)))
        fused = fused.reshape(samples, groups * width, dimension)
        flat_valid = self.valid.flatten()
        flat_indices = self.indices.flatten()[flat_valid]
        return station_tokens.index_copy(1, flat_indices, fused[:, flat_valid])


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "long_term_forecast":
            raise ValueError("S2Transformer supports long_term_forecast only.")
        self.pred_len = configs.pred_len
        self.variables = configs.enc_in
        self.stations = int(getattr(configs, "num_nodes", 1))
        dimension = configs.d_model
        if dimension % configs.n_heads:
            raise ValueError("S2Transformer d_model must be divisible by n_heads.")
        coordinates = load_station_coordinates(
            configs.root_path, getattr(configs, "station_coords_path", None),
            expected_nodes=self.stations,
        )
        group_counts = list(getattr(configs, "s2_group_counts", [16, 4]))
        if not group_counts:
            raise ValueError("S2Transformer needs at least one subgraph scale.")
        if len(group_counts) < configs.e_layers:
            group_counts.extend([group_counts[-1]] * (configs.e_layers - len(group_counts)))
        partitions = _geographic_partitions(coordinates, group_counts[:configs.e_layers])

        self.history_projection = nn.Linear(configs.seq_len, dimension)
        self.coordinate_projection = nn.Linear(5, dimension, bias=False)
        self.time_projection = nn.Linear(4, dimension, bias=False)
        self.register_buffer(
            "coordinate_features", torch.from_numpy(_coordinate_features(coordinates))
        )
        self.blocks = nn.ModuleList([
            StructuredSpatialBlock(
                dimension, configs.n_heads, configs.d_ff, configs.dropout,
                indices, valid,
            )
            for indices, valid in partitions
        ])
        self.output_norm = nn.LayerNorm(dimension)
        self.head = nn.Linear(dimension, configs.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(
                f"S2Transformer expects [B,L,N,V], got {tuple(x_enc.shape)}."
            )
        batch, length, stations, variables = x_enc.shape
        if stations != self.stations or variables != self.variables:
            raise ValueError(
                f"S2Transformer configured for N={self.stations}, V={self.variables}; "
                f"received N={stations}, V={variables}."
            )
        series = x_enc.permute(0, 3, 2, 1).reshape(
            batch * variables, stations, length
        )
        tokens = self.history_projection(series)
        tokens = tokens + self.coordinate_projection(self.coordinate_features)[None]
        if x_mark_enc is not None and x_mark_enc.shape[-1] >= 4:
            time_token = self.time_projection(x_mark_enc[:, -1, :4])
            time_token = time_token.repeat_interleave(variables, dim=0)
            tokens = tokens + time_token[:, None]
        for block in self.blocks:
            tokens = block(tokens)
        prediction = self.head(self.output_norm(tokens))
        return prediction.reshape(
            batch, variables, stations, self.pred_len
        ).permute(0, 3, 2, 1).contiguous()
