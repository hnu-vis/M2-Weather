from layers.DUET_Linear import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from layers.DUET_Attention import (
    Mahalanobis_mask,
    Encoder,
    EncoderLayer,
    FullAttention,
    AttentionLayer,
)
import torch


class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        self.linear_head = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout)
        )

    def forward(self, input):
        # x: [batch_size, seq_len, n_vars]
        if self.CI:
            channel_independent_input = rearrange(input, "b l n -> (b n) l 1")

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            temporal_feature = rearrange(
                reshaped_output, "(b n) l 1 -> b l n", b=input.shape[0]
            )

        else:
            temporal_feature, L_importance = self.cluster(input)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")
        if self.n_vars > 1:
            changed_input = rearrange(input, "b l n -> b n l")
            channel_mask = self.mask_generator(changed_input)

            channel_group_feature, attention = self.Channel_transformer(
                x=temporal_feature, attn_mask=channel_mask
            )

            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, "b n d -> b d n")
        output = self.cluster.revin(output, "denorm")
        return output, L_importance


class Model(nn.Module):
    """Time-Series-Library entry retaining the DUET network and loss."""

    def __init__(self, configs):
        super().__init__()
        # DUET sees one station at a time, with enc_in weather variables. The
        # unrelated spatial-model num_features option defaults to 1.
        native_channels = configs.enc_in
        for name in ("enc_in", "dec_in", "c_out"):
            setattr(configs, name, native_channels)
        defaults = {"CI": True, "fc_dropout": 0.2, "hidden_size": 256, "num_experts": 4, "noisy_gating": True, "k": 1, "output_attention": False}
        for name, value in defaults.items():
            if not hasattr(configs, name):
                setattr(configs, name, value)
        self.model = DUETModel(configs)
        self.auxiliary_loss = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 4:
            raise ValueError(f"DUET expects [B,L,N,V], got {tuple(x_enc.shape)}.")
        batch_size, length, num_stations, num_variables = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1, 3).reshape(
            batch_size * num_stations, length, num_variables
        )
        output, importance_loss = self.model(x_enc)
        self.auxiliary_loss = importance_loss if self.training else None
        return output.reshape(
            batch_size, num_stations, output.shape[1], num_variables
        ).permute(0, 2, 1, 3).contiguous()
