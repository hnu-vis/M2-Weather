import torch
from torch import nn


class Model(nn.Module):
    """Time-MoE pretrained inference exposed as long-term forecasting."""


    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "long_term_forecast":
            raise ValueError("TimeMoE is exposed through long_term_forecast.")
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("TimeMoE requires transformers.") from exc
        from utils.transformers_compat import (
            patch_dynamic_cache_seen_tokens,
            patch_legacy_generation_methods,
        )
        patch_dynamic_cache_seen_tokens()
        model_id = getattr(configs, "timemoe_model_path", "Maple728/TimeMoE-50M")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        patch_legacy_generation_methods(self.model)
        self.pred_len = configs.pred_len
        configured_chunk_size = int(
            getattr(configs, "foundation_chunk_size", 0) or 0
        )
        # Generation cache memory grows with the number of independent
        # station/variable series.  Bound it explicitly for Global-scale data.
        self.inference_chunk_size = configured_chunk_size or 1024

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        normalized = x_enc - means
        stdev = torch.sqrt(torch.var(normalized, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = normalized / stdev
        prediction = self.model.generate(
            normalized, max_new_tokens=self.pred_len
        )[:, -self.pred_len:]
        return prediction.to(x_enc.device) * stdev + means

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"TimeMoE expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch_size, length, num_stations, num_variables = x_enc.shape
        x_enc = x_enc.permute(0, 2, 3, 1).reshape(
            batch_size * num_stations * num_variables, length
        )
        prediction = torch.cat(
            [
                self.forecast(x_enc[start:start + self.inference_chunk_size])
                for start in range(0, x_enc.shape[0], self.inference_chunk_size)
            ],
            dim=0,
        )
        return prediction.reshape(
            batch_size, num_stations, num_variables, self.pred_len
        ).permute(0, 3, 1, 2).contiguous()
