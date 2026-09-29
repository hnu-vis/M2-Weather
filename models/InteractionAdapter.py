"""Frozen-backbone prediction interaction adapter.

The backbone produces the canonical M3 tensor ``[batch, horizon, station,
variable]``.  This module keeps that prediction as an identity residual and
learns only a small post-processing correction:

* ``variable``: mixes variables independently at every station (for MS models);
* ``station``: diffuses predictions over a geographic kNN graph (for SM models);
* ``both``: combines the two branches as a lightweight MM adapter.

Every residual output is zero-initialized, so an untrained adapter is exactly
the supplied checkpoint rather than a randomly perturbed forecast.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils.station_coordinates import load_station_coordinates


SELF_LOADING_BACKBONES = {"Moirai", "Sundial", "TimeMoE", "Timer"}


def _checkpoint_state(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Backbone checkpoint {path!r} does not contain a state dict.")
    if payload and all(key.startswith("module.") for key in payload):
        payload = {key[len("module."):]: value for key, value in payload.items()}
    return payload


def _geographic_adjacency(coordinates: np.ndarray, neighbors: int) -> torch.Tensor:
    """Return a row-normalized geographic kNN adjacency without self loops."""
    if coordinates.shape[0] < 2:
        raise ValueError("Station interaction requires at least two stations.")
    neighbors = min(max(int(neighbors), 1), coordinates.shape[0] - 1)
    lat_lon = torch.as_tensor(coordinates[:, :2], dtype=torch.float64)
    radians = torch.deg2rad(lat_lon)
    lat, lon = radians[:, 0], radians[:, 1]
    delta_lat = lat[:, None] - lat[None, :]
    delta_lon = lon[:, None] - lon[None, :]
    haversine = (
        torch.sin(delta_lat / 2).square()
        + torch.cos(lat[:, None]) * torch.cos(lat[None, :])
        * torch.sin(delta_lon / 2).square()
    ).clamp(0.0, 1.0)
    distance = 2.0 * torch.asin(torch.sqrt(haversine))
    distance.fill_diagonal_(float("inf"))
    knn_distance, knn_index = torch.topk(
        distance, k=neighbors, dim=1, largest=False
    )
    # A per-station length scale avoids making sparse rural stations vanish.
    scale = knn_distance.median(dim=1, keepdim=True).values.clamp_min(1e-8)
    weights = torch.softmax(-knn_distance / scale, dim=1)
    adjacency = torch.zeros_like(distance)
    adjacency.scatter_(1, knn_index, weights)
    return adjacency.to(dtype=torch.float32)


class PredictionInteractionAdapter(nn.Module):
    """Small identity-initialized residual corrector for forecast tensors."""

    def __init__(
        self,
        pred_len: int,
        num_variables: int,
        mode: str,
        adjacency: torch.Tensor | None = None,
        use_calibration: bool = True,
    ):
        super().__init__()
        if mode not in {"variable", "station", "both"}:
            raise ValueError(f"Unknown adapter mode {mode!r}.")
        if mode in {"station", "both"} and adjacency is None:
            raise ValueError("Station adapter mode requires a station adjacency.")
        self.pred_len = int(pred_len)
        self.num_variables = int(num_variables)
        self.mode = mode
        self.use_calibration = bool(use_calibration)

        if self.use_calibration:
            self.calibration_scale = nn.Parameter(
                torch.zeros(self.pred_len, self.num_variables)
            )
            self.calibration_bias = nn.Parameter(
                torch.zeros(self.pred_len, self.num_variables)
            )

        if mode in {"variable", "both"}:
            self.variable_norm = nn.LayerNorm(
                self.num_variables, elementwise_affine=False
            )
            # Lead-specific off-diagonal maps are both interpretable and much
            # cheaper than materializing a hidden tensor over B*H*N.
            self.variable_weight = nn.Parameter(
                torch.zeros(
                    self.pred_len, self.num_variables, self.num_variables
                )
            )
            self.register_buffer(
                "off_diagonal",
                1.0 - torch.eye(self.num_variables),
                persistent=False,
            )

        if mode in {"station", "both"}:
            self.register_buffer("adjacency", adjacency)
            # Keep the dense matrix for checkpoint/test compatibility, but use
            # its O(N*k) sparse representation for Europe/Global propagation.
            # The global graph has 2,504 nodes and only k non-zeros per row;
            # dense O(N^2) einsums would dominate every adapter-fit batch.
            self.register_buffer(
                "adjacency_sparse", adjacency.to_sparse().coalesce(),
                persistent=False,
            )
            # Separate one-hop and two-hop graph-diffusion strengths for every
            # lead and variable. This branch mixes stations but never variables.
            self.station_weight = nn.Parameter(
                torch.zeros(self.pred_len, self.num_variables, 2)
            )

    def _variable_correction(self, prediction: torch.Tensor) -> torch.Tensor:
        weight = self.variable_weight * self.off_diagonal[None, :, :]
        return torch.einsum(
            "bhnv,hvw->bhnw", self.variable_norm(prediction), weight
        )

    def _station_mix(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the geographic graph along N using sparse matrix multiply."""
        batch, horizon, stations, variables = values.shape
        if stations != self.adjacency.shape[0]:
            raise ValueError(
                f"Station adapter graph has {self.adjacency.shape[0]} nodes, "
                f"but prediction has {stations}."
            )
        # CUDA sparse-mm support is most portable in float32. The adapter's
        # public dtype is restored so this also remains safe under AMP.
        output_dtype = values.dtype
        flattened = values.float().permute(2, 0, 1, 3).reshape(stations, -1)
        mixed = torch.sparse.mm(self.adjacency_sparse.float(), flattened)
        return mixed.reshape(stations, batch, horizon, variables).permute(
            1, 2, 0, 3
        ).to(dtype=output_dtype)

    def station_differences(self, prediction: torch.Tensor) -> torch.Tensor:
        """Return standardized one-hop and two-hop graph residual features."""
        mean = prediction.mean(dim=2, keepdim=True)
        std = prediction.var(dim=2, keepdim=True, unbiased=False).add(1e-5).sqrt()
        normalized = (prediction - mean) / std
        neighbor_1 = self._station_mix(normalized)
        neighbor_2 = self._station_mix(neighbor_1)
        return torch.stack(
            (neighbor_1 - normalized, neighbor_2 - normalized), dim=-1
        )

    def _station_correction(self, prediction: torch.Tensor) -> torch.Tensor:
        graph_difference = self.station_differences(prediction)
        return torch.einsum(
            "bhnvk,hvk->bhnv", graph_difference, self.station_weight
        )

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 4:
            raise ValueError(
                "InteractionAdapter expects [B,H,N,V], got "
                f"{tuple(prediction.shape)}."
            )
        if prediction.shape[1] != self.pred_len:
            raise ValueError(
                f"Expected horizon {self.pred_len}, got {prediction.shape[1]}."
            )
        if prediction.shape[-1] != self.num_variables:
            raise ValueError(
                f"Expected {self.num_variables} variables, got {prediction.shape[-1]}."
            )

        correction = torch.zeros_like(prediction)
        if self.use_calibration:
            correction = correction + (
                prediction * self.calibration_scale[None, :, None, :]
                + self.calibration_bias[None, :, None, :]
            )
        if self.mode in {"variable", "both"}:
            correction = correction + self._variable_correction(prediction)
        if self.mode in {"station", "both"}:
            correction = correction + self._station_correction(prediction)
        return prediction + correction


class Model(nn.Module):
    """TSLib model wrapper that loads and permanently freezes a backbone."""

    def __init__(self, configs):
        super().__init__()
        backbone_name = getattr(configs, "backbone_model", None)
        checkpoint = getattr(configs, "backbone_checkpoint", None)
        if not backbone_name:
            raise ValueError(
                "InteractionAdapter requires --backbone_model."
            )
        if backbone_name == "InteractionAdapter":
            raise ValueError("InteractionAdapter cannot wrap itself.")
        if checkpoint:
            checkpoint = str(Path(checkpoint).expanduser().resolve())
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint}")
        elif backbone_name not in SELF_LOADING_BACKBONES:
            raise ValueError(
                f"{backbone_name} requires --backbone_checkpoint; only "
                f"{sorted(SELF_LOADING_BACKBONES)} self-load pretrained weights."
            )

        backbone_configs = copy.copy(configs)
        backbone_configs.model = backbone_name
        if backbone_name == "TimerXL":
            # run.py normally derives this default from args.model; the outer
            # wrapper name must not silently change the pretrained activation.
            backbone_configs.activation = "relu"
        module = importlib.import_module(f"models.{backbone_name}")
        backbone_class = getattr(module, "Model", getattr(module, backbone_name, None))
        if backbone_class is None:
            raise AttributeError(f"models.{backbone_name} has no model class.")
        self.backbone = backbone_class(backbone_configs)
        if checkpoint:
            incompatible = self.backbone.load_state_dict(
                _checkpoint_state(checkpoint), strict=True
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}."
                )
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.backbone_name = backbone_name

        mode = getattr(configs, "adapter_mode", "both")
        adjacency = None
        if mode in {"station", "both"}:
            num_stations = int(
                getattr(configs, "num_nodes", None)
                or getattr(configs, "num_node", None)
                or 1
            )
            coordinates = load_station_coordinates(
                configs.root_path,
                getattr(configs, "station_coords_path", None),
                expected_nodes=num_stations,
            )
            adjacency = _geographic_adjacency(
                coordinates, getattr(configs, "adapter_k_neighbors", 8)
            )
        self.adapter = PredictionInteractionAdapter(
            pred_len=configs.pred_len,
            num_variables=configs.enc_in,
            mode=mode,
            adjacency=adjacency,
            use_calibration=getattr(configs, "adapter_calibration", True),
        )
        trainable = sum(p.numel() for p in self.adapter.parameters())
        frozen = sum(p.numel() for p in self.backbone.parameters())
        print(
            f"InteractionAdapter: backbone={backbone_name} frozen={frozen:,}; "
            f"mode={mode}; trainable={trainable:,}."
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # ``super().train`` also toggles children, so re-freeze stochastic and
        # normalization behavior of the pretrained model every time.
        self.backbone.eval()
        return self

    def forecast_backbone(
        self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None
    ):
        self.backbone.eval()
        with torch.no_grad():
            if self.backbone_name in {"TQNet", "EasyST"}:
                prediction = self.backbone(
                    x_enc, x_mark_enc, x_dec, x_mark_dec, mask
                )
            else:
                prediction = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            prediction = prediction[:, -self.adapter.pred_len:, ...]
            if prediction.ndim == 3:
                expected = self.adapter.num_variables
                if prediction.shape[-1] % expected != 0:
                    raise ValueError(
                        f"Cannot restore station axis from backbone output "
                        f"{tuple(prediction.shape)} with {expected} variables."
                    )
                prediction = prediction.reshape(
                    prediction.shape[0], prediction.shape[1],
                    prediction.shape[-1] // expected, expected,
                )
            if prediction.ndim != 4:
                raise ValueError(
                    f"{self.backbone_name} returned unsupported shape "
                    f"{tuple(prediction.shape)}."
                )
        return prediction

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        prediction = self.forecast_backbone(
            x_enc, x_mark_enc, x_dec, x_mark_dec, mask
        )
        return self.adapter(prediction)
