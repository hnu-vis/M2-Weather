import numpy as np
import torch
from pathlib import Path
from torch import nn


def _resolve_pretrained_path(model_id):
    """Prefer an existing directory or cached Hub snapshot without networking."""
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        return local_path

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(model_id, local_files_only=True))
    except LocalEntryNotFoundError:
        # A fresh installation may not have the model yet. In that case only,
        # retain the standard Hub download behavior.
        return model_id


class Model(nn.Module):
    """Moirai probabilistic inference exposed as long-term forecasting."""


    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "long_term_forecast":
            raise ValueError("Moirai is exposed through long_term_forecast.")
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError as exc:
            raise ImportError("Moirai requires uni2ts.") from exc
        model_id = getattr(
            configs, "moirai_model_path", "Salesforce/moirai-2.0-R-small"
        )
        pretrained_path = _resolve_pretrained_path(model_id)
        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(pretrained_path),
            prediction_length=configs.pred_len,
            context_length=configs.seq_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.pred_len = configs.pred_len

    def forecast(self, x_enc):
        # ``Moirai2Forecast.predict`` creates its tensors on ``self.device``.
        # The outer experiment already moves this module to CUDA, so keep the
        # forecast module there.  Only the public NumPy input/output boundary
        # crosses the CPU; the expensive transformer inference stays on GPU.
        # Calling ``.to`` on the outer plain ``nn.Module`` moves the nested
        # parameters but does not update Lightning's internal device tracker;
        # invoke it on the forecast module itself once to keep tensors aligned.
        if self.model.device != x_enc.device:
            self.model.to(x_enc.device)
        outputs = []
        for channel in range(x_enc.shape[-1]):
            output = self.model.predict(x_enc[..., channel].detach().cpu().numpy())
            output = np.mean(output, axis=1)
            outputs.append(torch.as_tensor(output, dtype=x_enc.dtype, device=x_enc.device))
        return torch.stack(outputs, dim=-1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"Moirai expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch_size, length, num_stations, num_variables = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1, 3).reshape(
            batch_size * num_stations, length, num_variables
        )
        prediction = self.forecast(x_enc)
        return prediction.reshape(
            batch_size, num_stations, self.pred_len, num_variables
        ).permute(0, 2, 1, 3).contiguous()
