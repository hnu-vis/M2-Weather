from torch import nn


class Model(nn.Module):
    """Sundial pretrained inference exposed as long-term forecasting."""


    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "long_term_forecast":
            raise ValueError("Sundial is exposed through long_term_forecast.")
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("Sundial requires transformers.") from exc
        from utils.transformers_compat import (
            patch_dynamic_cache_seen_tokens,
            patch_legacy_generation_methods,
        )
        patch_dynamic_cache_seen_tokens()
        model_id = getattr(configs, "sundial_model_path", "thuml/sundial-base-128m")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        patch_legacy_generation_methods(self.model)
        self.pred_len = configs.pred_len

    def forecast(self, x_enc):
        output = self.model.generate(
            x_enc, max_new_tokens=self.pred_len, num_samples=20
        )
        return output.mean(dim=1).to(x_enc.device)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"Sundial expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch_size, length, num_stations, num_variables = x_enc.shape
        x_enc = x_enc.permute(0, 2, 3, 1).reshape(
            batch_size * num_stations * num_variables, length
        )
        prediction = self.forecast(x_enc)
        return prediction.reshape(
            batch_size, num_stations, num_variables, self.pred_len
        ).permute(0, 3, 1, 2).contiguous()
