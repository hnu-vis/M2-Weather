import torch
from torch import nn
from pathlib import Path
import math
from layers.TimerXL_Transformer import TimerBlock, TimerLayer
from layers.TimerXL_SelfAttention import AttentionLayer, TimeAttention


class Model(nn.Module):
    """
    Timer-XL: Long-Context Transformers for Unified Time Series Forecasting 

    Paper: https://arxiv.org/abs/2410.04803
    
    GitHub: https://github.com/thuml/OpenLTM
    
    Citation: @article{liu2024timer,
        title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
        author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
        journal={arXiv preprint arXiv:2410.04803},
        year={2024}
    }
    """
    def __init__(self, configs):
        super().__init__()
        if getattr(configs, "task_name", "long_term_forecast") != "long_term_forecast":
            raise ValueError("TimerXL is exposed through long_term_forecast.")

        default_path = (
            Path(__file__).resolve().parents[1]
            / "pretrained_checkpoints" / "Timer-XL" / "checkpoint.pth"
        )
        configured_path = getattr(configs, "timerxl_model_path", None)
        self.checkpoint_path = Path(configured_path).expanduser() if configured_path else default_path
        state_dict = None
        if self.checkpoint_path.is_file():
            try:
                payload = torch.load(
                    self.checkpoint_path, map_location="cpu", weights_only=True, mmap=True
                )
            except TypeError:
                payload = torch.load(self.checkpoint_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise TypeError(f"Unsupported Timer-XL checkpoint type: {type(payload).__name__}")
            for key in ("state_dict", "model_state_dict", "model"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
            state_dict = {
                (key[7:] if key.startswith("module.") else key): value
                for key, value in payload.items()
            }
            required = {"embedding.weight", "head.weight"}
            missing = required.difference(state_dict)
            if missing:
                raise KeyError(f"Timer-XL checkpoint misses required keys: {sorted(missing)}")
            layer_ids = {
                int(key.split(".")[2]) for key in state_dict
                if key.startswith("blocks.attn_layers.")
            }
            configs.input_token_len = state_dict["embedding.weight"].shape[1]
            configs.d_model = state_dict["embedding.weight"].shape[0]
            configs.output_token_len = state_dict["head.weight"].shape[0]
            configs.e_layers = max(layer_ids) + 1
            configs.d_ff = state_dict["blocks.attn_layers.0.conv1.weight"].shape[0]
            configs.n_heads = state_dict[
                "blocks.attn_layers.0.attention.inner_attention.attn_bias.emb.weight"
            ].shape[1]
        else:
            raise FileNotFoundError(f"Timer-XL checkpoint not found: {self.checkpoint_path}")

        defaults = {
            "input_token_len": 96, "output_token_len": 96, "d_model": 1024,
            "d_ff": 2048, "n_heads": 8, "e_layers": 8, "dropout": 0.1,
            "activation": "relu", "output_attention": False, "covariate": False,
            "flash_attention": False, "use_norm": True,
        }
        for name, value in defaults.items():
            if not hasattr(configs, name):
                setattr(configs, name, value)

        self.input_token_len = configs.input_token_len
        self.output_token_len = configs.output_token_len
        self.pred_len = configs.pred_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        self.output_attention = configs.output_attention
        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(
                            True, attention_dropout=configs.dropout,
                            output_attention=self.output_attention, d_model=configs.d_model,
                            num_heads=configs.n_heads, covariate=configs.covariate,
                            flash_attention=configs.flash_attention,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff, dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.head = nn.Linear(configs.d_model, self.output_token_len)
        self.use_norm = configs.use_norm
        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)

    def forecast(self, x, x_mark, _y_mark):
        if x.shape[1] == 0:
            raise ValueError("Timer-XL requires at least one context point.")
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        B, _, C = x.shape
        # [B, C, L]
        x = x.permute(0, 2, 1)
        # The pretrained projection consumes non-overlapping 96-point tokens.
        # Left-padding normalized values with zero preserves the most recent
        # observations and represents missing history at the series mean.
        padding = (-x.shape[-1]) % self.input_token_len
        if padding:
            x = torch.nn.functional.pad(x, (padding, 0))
        # [B, C, N, P]
        x = x.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x.shape[2]
        # [B, C, N, D]
        embed_out = self.embedding(x)
        # [B, C * N, D]
        embed_out = embed_out.reshape(B, C * N, -1)
        embed_out, attns = self.blocks(embed_out, n_vars=C, n_tokens=N)
        # [B, C * N, P]
        dec_out = self.head(embed_out)
        # [B, C, N * P]
        dec_out = dec_out.reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out


    def forecast_training(self, x_enc, x_mark_enc=None, x_mark_dec=None):
        """Run the native one-pass training forecast on canonical input."""
        if x_enc.ndim != 4:
            raise ValueError(f"Timer-XL expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch_size, length, num_stations, num_variables = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1, 3).reshape(
            batch_size * num_stations, length, num_variables
        )
        prediction = self.forecast(x_enc, x_mark_enc, x_mark_dec)
        if self.output_attention:
            prediction, attentions = prediction
            prediction = prediction[:, -self.output_token_len:, :]
            prediction = prediction[:, :self.pred_len, :]
            prediction = prediction.reshape(
                batch_size, num_stations, prediction.shape[1], num_variables
            ).permute(0, 2, 1, 3).contiguous()
            return prediction, attentions
        prediction = prediction[:, -self.output_token_len:, :]
        prediction = prediction[:, :self.pred_len, :]
        return prediction.reshape(
            batch_size, num_stations, prediction.shape[1], num_variables
        ).permute(0, 2, 1, 3).contiguous()

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"Timer-XL expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch_size, length, num_stations, num_variables = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1, 3).reshape(
            batch_size * num_stations, length, num_variables
        )
        context = x_enc
        predictions = []
        last_attentions = None
        steps = math.ceil(self.pred_len / self.output_token_len)
        for _ in range(steps):
            output = self.forecast(context, x_mark_enc, x_mark_dec)
            if self.output_attention:
                output, last_attentions = output
            next_token = output[:, -self.output_token_len:, :]
            predictions.append(next_token)
            if len(predictions) < steps:
                context = torch.cat(
                    [context[:, self.input_token_len:, :], next_token], dim=1
                )
        prediction = torch.cat(predictions, dim=1)[:, :self.pred_len, :]
        prediction = prediction.reshape(
            batch_size, num_stations, self.pred_len, num_variables
        ).permute(0, 2, 1, 3).contiguous()
        if self.output_attention:
            return prediction, last_attentions
        return prediction
