import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.task_name != "long_term_forecast":
            raise ValueError("Autoformer supports long_term_forecast in this repository.")
        # This adapter runs Autoformer independently for each station. Its
        # native channels are the per-station variables declared by enc_in;
        # num_features belongs to spatial models and defaults to 1.
        native_channels = configs.enc_in
        for name in ("enc_in", "dec_in", "c_out"):
            setattr(configs, name, native_channels)
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if configs.task_name == "long_term_forecast":
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_enc.shape[0], self.pred_len, x_enc.shape[2]],
            device=x_enc.device,
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if x_enc.ndim != 4 or x_dec.ndim != 4:
            raise ValueError("Autoformer expects x_enc and x_dec as [B,L,N,V].")
        batch_size, length, num_stations, num_variables = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1, 3).reshape(
            batch_size * num_stations, length, num_variables
        )
        if x_mark_enc is not None:
            if x_mark_enc.shape[-1] == 0:
                x_mark_enc = None
            else:
                x_mark_enc = x_mark_enc.repeat_interleave(num_stations, dim=0)
        if x_mark_dec is not None:
            if x_mark_dec.shape[-1] == 0:
                x_mark_dec = None
            else:
                x_mark_dec = x_mark_dec.repeat_interleave(num_stations, dim=0)
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out = dec_out[:, -self.pred_len:, :]
        return dec_out.reshape(
            batch_size, num_stations, self.pred_len, num_variables
        ).permute(0, 2, 1, 3).contiguous()
