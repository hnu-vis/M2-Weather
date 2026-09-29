"""xPatch adapted to the canonical M3 station tensor.

The temporal architecture follows the Apache-2.0 reference implementation of
"xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend
Decomposition" (Stitsyuk and Choi, AAAI 2025).  Stations are folded into the
batch dimension and restored after forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, variables: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(variables))
        self.bias = nn.Parameter(torch.zeros(variables))

    def normalize(self, values: torch.Tensor):
        mean = values.mean(dim=1, keepdim=True).detach()
        scale = values.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt().detach()
        normalized = (values - mean) / scale
        return normalized * self.weight + self.bias, mean, scale

    def denormalize(self, values: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor):
        values = (values - self.bias) / (self.weight + self.eps * self.eps)
        return values * scale + mean


class ExponentialDecomposition(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"xPatch alpha must be in (0, 1], got {alpha}.")
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, values: torch.Tensor):
        trend = [values[:, :1]]
        previous = values[:, 0]
        alpha = self.alpha.to(dtype=values.dtype)
        for step in range(1, values.shape[1]):
            previous = alpha * values[:, step] + (1.0 - alpha) * previous
            trend.append(previous[:, None])
        trend = torch.cat(trend, dim=1)
        return values - trend, trend


class DualStreamNetwork(nn.Module):
    def __init__(
        self, seq_len: int, pred_len: int, patch_len: int, stride: int,
        padding_at_end: bool,
    ):
        super().__init__()
        if pred_len % 2:
            raise ValueError("xPatch currently requires an even prediction length.")
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_at_end = padding_at_end
        patch_count = (seq_len - patch_len) // stride + 1
        if padding_at_end:
            self.padding = nn.ReplicationPad1d((0, stride))
            patch_count += 1
        self.patch_count = patch_count
        hidden = patch_len * patch_len

        self.patch_projection = nn.Linear(patch_len, hidden)
        self.patch_norm = nn.BatchNorm1d(patch_count)
        self.depthwise = nn.Conv1d(
            patch_count, patch_count, patch_len, patch_len, groups=patch_count
        )
        self.depthwise_norm = nn.BatchNorm1d(patch_count)
        self.residual_projection = nn.Linear(hidden, patch_len)
        self.pointwise = nn.Conv1d(patch_count, patch_count, 1)
        self.pointwise_norm = nn.BatchNorm1d(patch_count)
        self.seasonal_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(patch_count * patch_len, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len),
        )

        self.trend_1 = nn.Linear(seq_len, pred_len * 4)
        self.trend_norm_1 = nn.LayerNorm(pred_len * 2)
        self.trend_2 = nn.Linear(pred_len * 2, pred_len)
        self.trend_norm_2 = nn.LayerNorm(pred_len // 2)
        self.trend_3 = nn.Linear(pred_len // 2, pred_len)
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.fusion = nn.Linear(pred_len * 2, pred_len)
        self.activation = nn.GELU()

    def forward(self, seasonal: torch.Tensor, trend: torch.Tensor):
        batch, _, variables = seasonal.shape
        seasonal = seasonal.permute(0, 2, 1).reshape(batch * variables, -1)
        trend = trend.permute(0, 2, 1).reshape(batch * variables, -1)

        if self.padding_at_end:
            seasonal = self.padding(seasonal)
        seasonal = seasonal.unfold(-1, self.patch_len, self.stride)
        seasonal = self.patch_norm(self.activation(self.patch_projection(seasonal)))
        residual = self.residual_projection(seasonal)
        seasonal = self.depthwise_norm(self.activation(self.depthwise(seasonal)))
        seasonal = seasonal + residual
        seasonal = self.pointwise_norm(self.activation(self.pointwise(seasonal)))
        seasonal = self.seasonal_head(seasonal)

        trend = self.trend_norm_1(self.pool(self.trend_1(trend)))
        trend = self.trend_norm_2(self.pool(self.trend_2(trend)))
        trend = self.trend_3(trend)
        output = self.fusion(torch.cat((seasonal, trend), dim=-1))
        return output.reshape(batch, variables, self.pred_len).permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "long_term_forecast":
            raise ValueError("xPatch supports long_term_forecast only.")
        self.pred_len = configs.pred_len
        self.variables = configs.enc_in
        self.use_revin = bool(getattr(configs, "xpatch_revin", True))
        self.revin = RevIN(self.variables)
        self.decomposition = ExponentialDecomposition(
            getattr(configs, "xpatch_alpha", 0.3)
        )
        self.network = DualStreamNetwork(
            configs.seq_len,
            configs.pred_len,
            getattr(configs, "patch_len", 16),
            getattr(configs, "xpatch_stride", 8),
            getattr(configs, "xpatch_padding", "end") == "end",
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"xPatch expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch, length, stations, variables = x_enc.shape
        if variables != self.variables:
            raise ValueError(f"xPatch was configured for {self.variables} variables, got {variables}.")
        values = x_enc.permute(0, 2, 1, 3).reshape(batch * stations, length, variables)
        if self.use_revin:
            values, mean, scale = self.revin.normalize(values)
        seasonal, trend = self.decomposition(values)
        output = self.network(seasonal, trend)
        if self.use_revin:
            output = self.revin.denormalize(output, mean, scale)
        return output.reshape(batch, stations, self.pred_len, variables).permute(0, 2, 1, 3).contiguous()
