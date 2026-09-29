import torch
from torch import nn


class Model(nn.Module):
    """Hugging Face Timer zero-shot inference exposed as long-term forecasting."""


    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "long_term_forecast":
            raise ValueError("Timer is exposed through long_term_forecast.")
        try:
            from transformers import AutoModelForCausalLM
            from transformers.cache_utils import DynamicCache
        except ImportError as exc:
            raise ImportError("Timer requires transformers.") from exc

        class TimerDynamicCache(DynamicCache):
            """Restore the cache method expected by Timer's HF remote code."""

            def get_usable_length(self, _new_seq_length, layer_idx=0):
                return self.get_seq_length(layer_idx)

        self.cache_class = TimerDynamicCache
        model_id = getattr(configs, "timer_model_path", None) or "thuml/timer-base-84m"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model.requires_grad_(False)
        self.pred_len = configs.pred_len
        configured_chunk_size = int(
            getattr(configs, "foundation_chunk_size", 0) or 0
        )
        # Each station/variable is forecast independently.  Chunking this
        # flattened dimension avoids CUDA grid limits on continent-scale
        # batches without changing any individual prediction.
        self.inference_chunk_size = configured_chunk_size or 4096

    @torch.no_grad()
    def forecast(self, x_enc):
        batch_size, context_length = x_enc.shape
        if context_length == 0:
            raise ValueError("Timer requires at least one context point.")
        token_len = self.model.config.input_token_len
        padding = (-context_length) % token_len
        if padding:
            prefix = x_enc[:, :1].expand(-1, padding)
            model_input = torch.cat((prefix, x_enc), dim=1)
        else:
            model_input = x_enc

        # Timer's remote implementation performs RevIN with ``torch.std`` but
        # does not protect against a zero standard deviation.  Constant
        # station windows (most often calm wind or saturated RH) consequently
        # produce NaNs.  Normalize here with a finite scale and disable the
        # remote RevIN so every univariate series remains independent and
        # finite.
        location = model_input.mean(dim=-1, keepdim=True)
        scale = model_input.std(dim=-1, keepdim=True, unbiased=False)
        scale = torch.where(scale > 1e-5, scale, torch.ones_like(scale))
        model_input = (model_input - location) / scale
        attention_mask = torch.ones(
            model_input.shape[0], model_input.shape[1] // token_len,
            dtype=torch.long, device=model_input.device,
        )
        try:
            past_key_values = self.cache_class(config=self.model.config)
        except TypeError:
            past_key_values = self.cache_class()
        predictions = []
        generated_length = 0
        while generated_length < self.pred_len:
            remaining = self.pred_len - generated_length
            outputs = self.model(
                input_ids=model_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                max_output_length=remaining,
                revin=False,
            )
            next_token = outputs.logits[:, :remaining]
            predictions.append(next_token)
            generated_length += next_token.shape[1]
            if generated_length < self.pred_len:
                model_input = next_token
                past_key_values = outputs.past_key_values
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(
                            attention_mask.shape[0],
                            next_token.shape[1] // token_len,
                        ),
                    ],
                    dim=1,
                )
        prediction = torch.cat(predictions, dim=1)[:, :self.pred_len]
        return prediction * scale + location

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"Timer expects [B,L,N,V], got {tuple(x_enc.shape)}.")
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
