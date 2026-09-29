"""TSLib adapter for the TQNet implementation.

TQNet requires the phase index of every sample.  The adapter accepts it as a
one-dimensional second argument.  With normal TSLib time marks it derives the
phase from the configured mark column and optional offset.
"""

import torch
import torch.nn as nn

from layers.TQNet import Model as TQNetModel


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # ``num_features`` belongs to the generic spatial runner and is one
        # feature per station. TQNet's channel axis is instead the weather
        # variables configured by enc_in (u, v, T and RH here).
        native_channels = configs.enc_in
        for name in ("enc_in", "dec_in", "c_out"):
            setattr(configs, name, native_channels)
        self.num_variables = native_channels
        for name, value in {"cycle": 24, "model_type": "mlp", "use_revin": True}.items():
            if not hasattr(configs, name):
                setattr(configs, name, value)
        self.cycle = configs.cycle
        self.cycle_mark_column = getattr(configs, "cycle_mark_column", -1)
        self.cycle_offset = getattr(configs, "cycle_offset", 0)
        self.model = TQNetModel(configs)

    def _cycle_index(self, x_mark_enc, batch_size, device):
        if x_mark_enc is None:
            raise ValueError(
                "TQNet requires forecast_start or fallback calendar marks."
            )
        if x_mark_enc.ndim == 1:
            return x_mark_enc.to(device=device, dtype=torch.long) % self.cycle
        if x_mark_enc.ndim == 2 and x_mark_enc.shape[1] == 1:
            return x_mark_enc[:, 0].to(device=device, dtype=torch.long) % self.cycle

        phase = x_mark_enc[:, 0, self.cycle_mark_column]
        if phase.is_floating_point():
            if phase.min() >= 0 and phase.max() <= 1.01:
                phase = torch.round(phase * self.cycle)
            elif phase.min() >= -0.51 and phase.max() <= 0.51:
                phase = torch.round((phase + 0.5) * self.cycle)
        return (phase.to(dtype=torch.long) + self.cycle_offset) % self.cycle

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"TQNet expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch_size, length, num_stations, num_variables = x_enc.shape
        if num_variables != self.num_variables:
            raise ValueError(
                f"TQNet expected {self.num_variables} input variables from enc_in, "
                f"but received {num_variables}."
            )
        x_enc = x_enc.permute(0, 2, 1, 3).reshape(
            batch_size * num_stations, length, num_variables
        )
        if x_mark_enc is not None:
            if x_mark_enc.shape[-1] == 0:
                x_mark_enc = None
            else:
                x_mark_enc = x_mark_enc.repeat_interleave(num_stations, dim=0)
        phase_source = mask
        if phase_source is not None:
            phase_source = phase_source.repeat_interleave(num_stations, dim=0)
        else:
            phase_source = x_mark_enc
        cycle_index = self._cycle_index(phase_source, x_enc.shape[0], x_enc.device)
        prediction = self.model(x_enc, cycle_index)
        return prediction.reshape(
            batch_size, num_stations, prediction.shape[1], num_variables
        ).permute(0, 2, 1, 3).contiguous()
