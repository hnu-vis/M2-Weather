"""EasyST for multi-station single-variable forecasting.

This implementation follows the CIKM 2024 EasyST design: a frozen spatial
teacher supervises a lightweight MLP student through a teacher-bounded
regression objective, while a variational information bottleneck regularizes
the student representation.  Variables are folded into the batch dimension,
so the model shares one station forecaster across variables without mixing
their values.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.network = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(self.norm(inputs))


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Teacher checkpoint {path} does not contain a state dict.")
    if payload and all(key.startswith("module.") for key in payload):
        payload = {key[len("module."):]: value for key, value in payload.items()}
    return payload


class Model(nn.Module):
    """Frozen-teacher EasyST student with identity-free variable processing."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "long_term_forecast":
            raise ValueError("EasyST supports long_term_forecast only.")
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.num_nodes = int(configs.num_nodes)
        self.num_variables = int(configs.enc_in)
        embed_dim = int(getattr(configs, "easyst_embed_dim", 64))
        node_dim = int(getattr(configs, "easyst_node_dim", 64))
        time_dim = int(getattr(configs, "easyst_time_dim", 64))
        transition_dim = int(getattr(configs, "easyst_transition_dim", 64))
        layers = int(getattr(configs, "easyst_num_layers", 3))
        dropout = float(configs.dropout)

        self.distill_weight = float(getattr(configs, "easyst_distill_weight", 0.3))
        self.teacher_delta = float(getattr(configs, "easyst_teacher_delta", 0.1))
        self.ib_weight = float(getattr(configs, "easyst_ib_weight", 1e-3))

        self.history_projection = nn.Linear(self.seq_len, embed_dim)
        self.station_prompt = nn.Parameter(torch.empty(self.num_nodes, node_dim))
        self.hour_prompt = nn.Parameter(torch.empty(24, time_dim))
        self.weekday_prompt = nn.Parameter(torch.empty(7, time_dim))
        self.transition_time = nn.Parameter(torch.empty(24, transition_dim))
        self.transition_station = nn.Parameter(
            torch.empty(self.num_nodes, transition_dim)
        )
        self.transition_core = nn.Parameter(
            torch.empty(transition_dim, transition_dim, transition_dim)
        )
        self.history_weights = nn.Parameter(torch.zeros(self.seq_len))

        hidden_dim = embed_dim + node_dim + 3 * time_dim
        if transition_dim != time_dim:
            self.transition_projection = nn.Linear(transition_dim, time_dim)
        else:
            self.transition_projection = nn.Identity()
        self.encoder = nn.Sequential(
            *[ResidualMLP(hidden_dim, dropout) for _ in range(layers)]
        )
        if hidden_dim % 2:
            raise ValueError("EasyST fused hidden dimension must be even.")
        self.latent_dim = hidden_dim // 2
        self.regression = nn.Linear(self.latent_dim, self.pred_len)
        self._reset_parameters()

        self.teacher = self._build_teacher(configs)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self._teacher_prediction: torch.Tensor | None = None
        self._latent_statistics: tuple[torch.Tensor, torch.Tensor] | None = None

        student_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("teacher.")
        )
        teacher_parameters = sum(parameter.numel() for parameter in self.teacher.parameters())
        print(
            f"EasyST: student={student_parameters:,} parameters; "
            f"frozen teacher=STELLA ({teacher_parameters:,} parameters)."
        )

    def _reset_parameters(self):
        for parameter in (
            self.station_prompt,
            self.hour_prompt,
            self.weekday_prompt,
            self.transition_time,
            self.transition_station,
        ):
            nn.init.xavier_uniform_(parameter)
        nn.init.xavier_uniform_(self.transition_core.reshape(
            self.transition_core.shape[0], -1
        ))

    @staticmethod
    def _teacher_checkpoint(configs) -> Path | None:
        # When EasyST is constructed inside InteractionAdapter, the complete
        # EasyST checkpoint (including frozen teacher weights) is loaded by the
        # wrapper, so no external teacher checkpoint is needed here.
        if getattr(configs, "backbone_checkpoint", None):
            return None
        configured = getattr(configs, "easyst_teacher_checkpoint", "auto")
        if configured and configured != "auto":
            path = Path(configured).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"EasyST teacher checkpoint not found: {path}")
            return path

        seed = int(getattr(configs, "seed", 2024))
        project_root = Path(__file__).resolve().parents[1]
        roots = [
            Path(configs.checkpoints).expanduser().resolve(),
            project_root / "experiment_records" / "forecasting_uvtrh" / "checkpoints",
        ]
        dataset_name = str(getattr(configs, "data", "French"))
        pattern = f"*_STELLA_{dataset_name}_*seed{seed}_0/checkpoint.pth"
        for root in roots:
            matches = sorted(root.glob(pattern)) if root.is_dir() else []
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple EasyST teacher checkpoints match {root / pattern}: {matches}"
                )
        raise FileNotFoundError(
            f"EasyST needs the completed STELLA seed {seed} checkpoint; searched "
            + ", ".join(str(root / pattern) for root in roots)
        )

    def _build_teacher(self, configs):
        from models.STELLA import Model as STELLAModel

        teacher_configs = copy.copy(configs)
        teacher_configs.model = "STELLA"
        teacher_configs.d_model = 32
        teacher_configs.e_layers = 2
        teacher_configs.dropout = 0.2
        teacher_configs.if_rel = False
        teacher_configs.res_conn = True
        teacher = STELLAModel(teacher_configs)
        checkpoint = self._teacher_checkpoint(configs)
        if checkpoint is not None:
            incompatible = teacher.load_state_dict(_state_dict(checkpoint), strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"EasyST teacher mismatch: missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}."
                )
            print(f"EasyST: loaded frozen STELLA teacher from {checkpoint}")
        return teacher

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _time_indices(
        self, batch: int, device: torch.device, forecast_start: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if forecast_start is None:
            raise ValueError(
                "EasyST requires forecast_start metadata for hour/day prompts."
            )
        forecast_start = forecast_start.reshape(batch).to(device=device, dtype=torch.long)
        offsets = torch.arange(-self.seq_len, 0, device=device)
        absolute_hours = forecast_start[:, None] + offsets[None, :]
        hour = absolute_hours.remainder(24)
        # 2016-01-01 was Friday (Monday=0 -> 4).
        weekday = (absolute_hours.div(24, rounding_mode="floor") + 4).remainder(7)
        return hour, weekday

    def _student(
        self, x_enc: torch.Tensor, forecast_start: torch.Tensor | None
    ) -> torch.Tensor:
        batch, length, stations, variables = x_enc.shape
        if length != self.seq_len or stations != self.num_nodes:
            raise ValueError(
                f"EasyST configured for L={self.seq_len}, N={self.num_nodes}; "
                f"received {tuple(x_enc.shape)}."
            )
        if variables != self.num_variables:
            raise ValueError(
                f"EasyST configured for {self.num_variables} variables, got {variables}."
            )

        hour, weekday = self._time_indices(batch, x_enc.device, forecast_start)
        history_weight = torch.softmax(self.history_weights, dim=0)
        hour_context = torch.einsum(
            "l,bld->bd", history_weight, self.hour_prompt[hour]
        )
        weekday_context = torch.einsum(
            "l,bld->bd", history_weight, self.weekday_prompt[weekday]
        )
        transition_context = torch.einsum(
            "l,bld->bd", history_weight, self.transition_time[hour]
        )
        # Exact Tucker interaction after associating the station factor with
        # the core first: E[b,n,k] = sum_ij T[b,i] S[n,j] G[i,j,k].
        station_core = torch.einsum(
            "nj,ijk->nik", self.transition_station, self.transition_core
        )
        transition = torch.einsum(
            "bi,nik->bnk", transition_context, station_core
        )
        transition = torch.softmax(transition, dim=1)
        transition = self.transition_projection(transition)

        series = x_enc.permute(0, 3, 2, 1).reshape(
            batch * variables, stations, length
        )
        history = self.history_projection(series)
        station = self.station_prompt[None].expand(batch * variables, -1, -1)
        hour_context = hour_context.repeat_interleave(variables, dim=0)
        weekday_context = weekday_context.repeat_interleave(variables, dim=0)
        transition = transition.repeat_interleave(variables, dim=0)
        fused = torch.cat(
            (
                history,
                station,
                hour_context[:, None].expand(-1, stations, -1),
                weekday_context[:, None].expand(-1, stations, -1),
                transition,
            ),
            dim=-1,
        )
        statistics = self.encoder(fused)
        mu, raw_std = statistics.split(self.latent_dim, dim=-1)
        std = F.softplus(raw_std).clamp_min(1e-5)
        self._latent_statistics = (mu, std)
        latent = mu + torch.randn_like(std) * std if self.training else mu
        prediction = self.regression(latent)
        return prediction.reshape(
            batch, variables, stations, self.pred_len
        ).permute(0, 3, 2, 1).contiguous()

    def forward(
        self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
        forecast_start=None,
    ):
        prediction = self._student(x_enc, forecast_start)
        if self.training:
            self.teacher.eval()
            with torch.no_grad():
                self._teacher_prediction = self.teacher(
                    x_enc, x_mark_enc, x_dec, x_mark_dec
                )
        else:
            self._teacher_prediction = None
        return prediction

    def training_objective(
        self, prediction_loss: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Equation 12/13: predictive + teacher-bound + dual-path IB."""
        if self._teacher_prediction is None or self._latent_statistics is None:
            return prediction_loss
        teacher_prediction = self._teacher_prediction.reshape_as(prediction)
        teacher_loss = F.mse_loss(teacher_prediction, target)
        bounded = torch.where(
            prediction_loss.detach() + self.teacher_delta >= teacher_loss.detach(),
            prediction_loss,
            prediction_loss.new_zeros(()),
        )
        mu, std = self._latent_statistics
        kl = 0.5 * (
            -2.0 * torch.log(std) + std.square() + mu.square() - 1.0
        ).mean()
        return prediction_loss + self.distill_weight * bounded + self.ib_weight * kl
